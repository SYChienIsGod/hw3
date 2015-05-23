#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#include <iostream>
#include <string>
#include <fstream>
#include <map>
#include <vector>
#include <stdio.h>
#include <sstream>
#include <cstdlib>
#include <time.h> 
#include <random>
#include <algorithm>
#include <chrono>
extern "C"
{
   #include <cblas.h>
}
#include "fastexp.h"

#define FLT 0
#define DBL 1

#define PRECISION FLT

#if PRECISION == DBL
#define GEMV cblas_dgemv
#define ACTIVATION_TYPE double
#define GER cblas_dger
#elif PRECISION == FLT
#define GEMV cblas_sgemv
#define ACTIVATION_TYPE float
#define GER cblas_sger
#endif

#define AT ACTIVATION_TYPE

using namespace std;

static map<string, int> wordMapping;
static vector<int> wordCount;
static vector<vector<int>*> sentencesRaw;
static vector<vector<int>*> sentences;
static vector<pair<int,int>> word2CIMap;

// msCIBW msWws msWss msWscWsi
long long msBackward = 0;
long long msForward = 0;
long long msCIBW = 0;
long long msWws = 0;
long long msWss = 0;
long long msWscWsi = 0;


int nextMappingId = 0;
int debug;

struct Config {
    int NStates;    // The dimension of the state vector
    int NWords;     // The number of (unique) words
    int NSentences; // The number of sentences
    int NTokens;    // The number of tokens in the text
    int NMaxSent;   // The maximum sentence length
    int NSentenceLength; // The length of each sentence after transformation
    double randMin; // unused
    double randMax; // Normally distributed initial weights: variance
    int NLosses;    // How many losses to be included in the averaging process
    int NEpochs;    // How many training epochs
    int NClasses;   // How many classes (for the decomposition of the Softmax Layer)
    int NIndices;   // How many indices (that is: words) at maximum per class
    int Overlap;    // Overlap of the batches
};
struct SentState {    
    int N; // length of the sentence
    AT * s; // The states
    AT * idx;   // The index probability outputs
    AT * cls;   // The class probability outputs
    int * idx_t; // The true index output
    int * cls_t; // The true class output
    AT * s_ds; // The derivative of the activation function (i.e. f'(x)=[1-f(x)]f(x) for f(x)=sigmoid(x))
    AT * delta; // The backward pass derivatives
    AT * p_idx; // Holds the derivatives dL/dIdx
    AT * p_cls; // Holds the derivatives dL/dCls
    int * w; // Word ids of the sentence (pointer into labels)
    AT * dL_dWsi;
    AT * dL_dWsc;
    AT * dL_dWss;
    double totalLoss;
    int sentenceId;
};
struct RNN {
    Config * config;
    AT * W_ss;
    AT * W_ws;
    AT * W_si;  // s_t -> i_t (index vector)
    AT * W_sc;  // s_t -> c_t (class vector)
    SentState * ss;
    AT * s0;
    double lr;
    double * lastLosses;
    int lossPtr;
    double avgLoss;
    int * labels;
    int * classBoundaries;
};
void printMatrix(AT * A, int M, int N) {
    for(int k = 0; k < M; k++) {
        for(int l = 0; l < N; l++) {
            printf("%d|%d=%0.2f, ",k,l,A[k*N+l]);
        }
        printf("\n");
    }
}

void printWss(RNN * rnn) {
    printMatrix(rnn->W_ss,rnn->config->NStates,rnn->config->NStates);
}

bool hasNaN(AT * A, int M, int N) {
    for(int k = 0; k < M; k++) {
        for(int l = 0; l < N; l++) {
            if(std::isnan(A[k*N+l]))
                return true;
        }
    }
    return false;
}

double fRand(double fMin, double fMax) {
    double f = (double)rand() / RAND_MAX;
    return fMin + f * (fMax - fMin);
}

void randomInitMatrix(AT * matrix, int m, int n, double min, double max) {
    default_random_engine generator;
    normal_distribution<AT> distribution(0.0,max);
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            matrix[i*n+j] = (AT) fRand(min,max);//distribution(generator);//fRand(min,max);
        }
    }
}

void buildMatrices(RNN * rnn) {
    AT * wss = new AT[rnn->config->NStates*rnn->config->NStates];
    AT * wws = new AT[rnn->config->NWords*rnn->config->NStates];
    AT * wsi = new AT[rnn->config->NIndices*rnn->config->NStates];
    AT * wsc = new AT[rnn->config->NClasses*rnn->config->NStates];
    randomInitMatrix(wss,rnn->config->NStates,rnn->config->NStates,rnn->config->randMin,rnn->config->randMax);
    randomInitMatrix(wws,rnn->config->NWords,rnn->config->NStates,rnn->config->randMin,rnn->config->randMax);
    randomInitMatrix(wsi,rnn->config->NIndices,rnn->config->NStates,rnn->config->randMin,rnn->config->randMax);
    randomInitMatrix(wsc,rnn->config->NClasses,rnn->config->NStates,rnn->config->randMin,rnn->config->randMax);
    rnn->W_ss=wss;
    rnn->W_ws=wws;
    rnn->W_sc=wsc;
    rnn->W_si=wsi;
    AT * s0 = new AT[rnn->config->NStates];
    for(int i=0;i<rnn->config->NStates;i++) {
        s0[i]=0.0;
    }
    rnn->s0=s0;
    rnn->lastLosses = new double[rnn->config->NLosses];
    for(int i=0;i<rnn->config->NLosses;i++) {
        rnn->lastLosses[i]=0.0;
    }
    rnn->lossPtr = 0;
    rnn->avgLoss = 0.0;
}

AT * getZeroMatrix(AT * matrix, int m, int n) {
    if (matrix==NULL)
        matrix = new AT[m*n];
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            matrix[i*n+j] = (AT) 0.0;
        }
    }
    return matrix;
}

void initSentState(RNN * rnn) {
    SentState * ss = new SentState;
    ss->N = 0;
    AT * s = getZeroMatrix(NULL,rnn->config->NStates,rnn->config->NMaxSent);
    AT * idx = getZeroMatrix(NULL,rnn->config->NIndices,rnn->config->NMaxSent);   // The index probability outputs
    AT * cls = getZeroMatrix(NULL,rnn->config->NClasses,rnn->config->NMaxSent);   // The class probability outputs
    ss->s=s;
    ss->idx = idx;
    ss->cls = cls;
    ss->s_ds = getZeroMatrix(NULL,rnn->config->NStates,rnn->config->NMaxSent);
    ss->dL_dWss = getZeroMatrix(NULL,rnn->config->NStates, rnn->config->NStates);
    ss->dL_dWsc = getZeroMatrix(NULL,rnn->config->NClasses, rnn->config->NStates);
    ss->dL_dWsi = getZeroMatrix(NULL,rnn->config->NIndices, rnn->config->NStates);
    ss->delta = getZeroMatrix(NULL,rnn->config->NStates,rnn->config->NMaxSent);
    ss->p_idx = getZeroMatrix(NULL,rnn->config->NIndices,rnn->config->NMaxSent);
    ss->p_cls = getZeroMatrix(NULL,rnn->config->NClasses,rnn->config->NMaxSent);
    rnn->ss = ss;
    ss->w = new int[rnn->config->NMaxSent];
    ss->cls_t = new int[rnn->config->NMaxSent];
    ss->idx_t = new int[rnn->config->NMaxSent];
}

void setClassIndexWordVectorsForSentence(RNN * rnn, int sId) {
    rnn->ss->N = sentences[sId]->size();
    int labelOffset = sId*rnn->config->NSentenceLength*3;
    rnn->ss->w = &rnn->labels[labelOffset];
    rnn->ss->idx_t = &rnn->labels[labelOffset+2*rnn->config->NSentenceLength+1];
    rnn->ss->cls_t = &rnn->labels[labelOffset+rnn->config->NSentenceLength+1];
}

void printVector(AT * v, int n) {
    for(int i = 0; i < n; i++) {
        printf(" %0.3f", v[i]);
    }
    printf("\n");
}

void checkMatricesForNans(string out, RNN * rnn) {
    bool wssNan = hasNaN(rnn->W_ss,rnn->config->NStates,rnn->config->NStates);
    bool wsiNan =  hasNaN(rnn->W_si,rnn->config->NIndices,rnn->config->NStates);
    bool wscNan = hasNaN(rnn->W_sc,rnn->config->NClasses,rnn->config->NStates);
    bool wwsNan = hasNaN(rnn->W_ws,rnn->config->NStates,rnn->config->NWords);
    if(wssNan || wsiNan || wscNan || wwsNan)
        printf("%s Wss: %d, Wsi: %d, Wsc: %d, Wws: %d\n",out.data(),wssNan,wsiNan,wscNan,wwsNan);
}

void updateStates(RNN * rnn) {
    //if(debug>0) printf("Updating states.\n");   
    bool useReLU = false;
    SentState * ss = rnn->ss;
    ss->totalLoss = 0.0;
    for(int i = 0; i < rnn->ss->N-1; i++) {
        AT * prevState = rnn->s0;
        if(i>0) prevState = &ss->s[(i-1)*rnn->config->NStates];
        AT * currWord   = &rnn->W_ws[rnn->ss->w[i]*rnn->config->NStates];//&ss->x[i*rnn->config->NStates];
        AT * currState  = &ss->s[i*rnn->config->NStates];
        AT * currStateD = &ss->s_ds[i*rnn->config->NStates];
        AT tmpSum;
        GEMV(CblasRowMajor, CblasNoTrans, rnn->config->NStates, 
                rnn->config->NStates, 1.0, rnn->W_ss, rnn->config->NStates, 
                prevState, 1, 0.0, currState, 1);
        for(int k = 0; k < rnn->config->NStates; k+=8) {
            currState[k+0] = 1.0 / (1.0 + fasterexp (-currState[k+0]-currWord[k+0]));
            currStateD[k+0] = (1-currState[k+0])*currState[k+0];
            currState[k+1] = 1.0 / (1.0 + fasterexp (-currState[k+1]-currWord[k+1]));
            currStateD[k+1] = (1-currState[k+1])*currState[k+1];
            currState[k+2] = 1.0 / (1.0 + fasterexp (-currState[k+2]-currWord[k+2]));
            currStateD[k+2] = (1-currState[k+2])*currState[k+2];
            currState[k+3] = 1.0 / (1.0 + fasterexp (-currState[k+3]-currWord[k+3]));
            currStateD[k+3] = (1-currState[k+3])*currState[k+3];
            currState[k+4] = 1.0 / (1.0 + fasterexp (-currState[k+4]-currWord[k+4]));
            currStateD[k+4] = (1-currState[k+4])*currState[k+4];
            currState[k+5] = 1.0 / (1.0 + fasterexp (-currState[k+5]-currWord[k+5]));
            currStateD[k+5] = (1-currState[k+5])*currState[k+5];
            currState[k+6] = 1.0 / (1.0 + fasterexp (-currState[k+6]-currWord[k+6]));
            currStateD[k+6] = (1-currState[k+6])*currState[k+6];
            currState[k+7] = 1.0 / (1.0 + fasterexp (-currState[k+7]-currWord[k+7]));
            currStateD[k+7] = (1-currState[k+7])*currState[k+7];
        }
        AT * currClsOut = &ss->cls[i*rnn->config->NClasses];
        GEMV(CblasRowMajor, CblasNoTrans, rnn->config->NClasses, 
                rnn->config->NStates, 1.0, rnn->W_sc, rnn->config->NStates, 
                currState, 1, 0.0, currClsOut, 1);
        AT denominator = 0.0;
        for(int k = 0; k < rnn->config->NClasses; k+=8) {
            currClsOut[k+0] = fasterexp(currClsOut[k+0]);
            denominator += currClsOut[k+0];
            currClsOut[k+1] = fasterexp(currClsOut[k+1]);
            denominator += currClsOut[k+1];
            currClsOut[k+2] = fasterexp(currClsOut[k+2]);
            denominator += currClsOut[k+2];
            currClsOut[k+3] = fasterexp(currClsOut[k+3]);
            denominator += currClsOut[k+3];
            currClsOut[k+4] = fasterexp(currClsOut[k+4]);
            denominator += currClsOut[k+4];
            currClsOut[k+5] = fasterexp(currClsOut[k+5]);
            denominator += currClsOut[k+5];
            currClsOut[k+6] = fasterexp(currClsOut[k+6]);
            denominator += currClsOut[k+6];
            currClsOut[k+7] = fasterexp(currClsOut[k+7]);
            denominator += currClsOut[k+7];
        }
        if(denominator == 0.0) {printf("computeSoftmaxClass: Denominator is 0.0, t=%d, sent=%d",i,ss->sentenceId);printVector(currClsOut,rnn->config->NClasses);exit(1);}
        for(int k = 0; k < rnn->config->NClasses; k+=8) {
            currClsOut[k+0] = currClsOut[k+0]/denominator;
            currClsOut[k+1] = currClsOut[k+1]/denominator;
            currClsOut[k+2] = currClsOut[k+2]/denominator;
            currClsOut[k+3] = currClsOut[k+3]/denominator;
            currClsOut[k+4] = currClsOut[k+4]/denominator;
            currClsOut[k+5] = currClsOut[k+5]/denominator;
            currClsOut[k+6] = currClsOut[k+6]/denominator;
            currClsOut[k+7] = currClsOut[k+7]/denominator;
        }    
        int maxClass = ss->cls_t[i];
        int A = rnn->classBoundaries[maxClass*2];
        int B = rnn->classBoundaries[maxClass*2+1];
        AT * currIdxOut = &ss->idx[i*rnn->config->NIndices];
        
        GEMV(CblasRowMajor, CblasNoTrans, B-A+1, 
                rnn->config->NStates, 1.0, &rnn->W_si[A*rnn->config->NStates], rnn->config->NStates, 
                currState, 1, 0.0, currIdxOut+A, 1);    
        currIdxOut = &ss->idx[i*rnn->config->NIndices];
        denominator = 0.0;
        for(int k = A; k <= B; k++) {
            currIdxOut[k] = fasterexp(currIdxOut[k]);
            denominator += currIdxOut[k];
        }
        if(denominator == 0.0) {printf("computeSoftmaxIndex: Denominator is 0.0, t=%d, sentenceId=%d\n",i, ss->sentenceId);checkMatricesForNans("computeSoftmaxIndex: ", rnn);exit(1);}
        for(int k = A; k <= B; k++) {
            currIdxOut[k] = currIdxOut[k]/denominator;
        }
        ss->totalLoss += -fasterlog2(currIdxOut[ss->idx_t[i]])-fasterlog2(currClsOut[maxClass]);
    }
    ss->totalLoss/=(double)(ss->N-1);
}

void computeClassIndexForwardPass(RNN * rnn, int sentenceId) {
    rnn->ss->sentenceId = sentenceId;
    setClassIndexWordVectorsForSentence(rnn,sentenceId);
    updateStates(rnn);
    //computeClassIndexSoftmax(rnn);
    //computeClassIndexError(rnn);
}

void update_Wsc_Wsi(RNN * rnn) {
    SentState * ss = rnn->ss;
    for(int i = 0; i < ss->N-1; i++) {
        int maxClass = ss->cls_t[i];
        int A = rnn->classBoundaries[maxClass*2];
        int B = rnn->classBoundaries[maxClass*2+1];
        AT * currState = &ss->s[i*rnn->config->NStates];
        //AT * current_dL_dCls = &ss->p_cls[i*rnn->config->NClasses];
        AT * current_dL_dCls = &ss->cls[i*rnn->config->NClasses];
        AT * current_dL_dIdx = &ss->idx[i*rnn->config->NIndices+A];
        GER(CblasRowMajor,B-A+1,rnn->config->NStates,-rnn->lr,
                current_dL_dIdx,1,currState,1,&rnn->W_si[A*rnn->config->NStates],rnn->config->NStates);
        GER(CblasRowMajor,rnn->config->NClasses,rnn->config->NStates,-rnn->lr,
                current_dL_dCls,1,currState,1,rnn->W_sc,rnn->config->NStates);
    }
}

void compute_dL_dWss(RNN * rnn) {
    /* 
     void cblas_dger(const enum CBLAS_ORDER order, const int M, const int N,
                const double alpha, const double *X, const int incX,
                const double *Y, const int incY, double *A, const int lda);
     */
    SentState * ss = rnn->ss;
    AT * currS;
    for(int i = 1; i<ss->N-1; i++) {
        currS = &ss->s[(i-1)*rnn->config->NStates];
        AT * currDelta = &ss->delta[i*rnn->config->NStates];
        GER(CblasRowMajor,rnn->config->NStates,rnn->config->NStates,-rnn->lr,currDelta,1,currS,1,rnn->W_ss,rnn->config->NStates);
    }
}

void compute_dL_dWws(RNN * rnn) {
    SentState * ss = rnn->ss;
    for(int i = 0; i<ss->N-1; i++) {
        AT * currDelta = &ss->delta[i*rnn->config->NStates];
        for(int j = 0; j < rnn->config->NStates; j+=8) {
            rnn->W_ws[j+0+ss->w[i]*rnn->config->NStates]-=rnn->lr*currDelta[j+0];
            rnn->W_ws[j+1+ss->w[i]*rnn->config->NStates]-=rnn->lr*currDelta[j+1];
            rnn->W_ws[j+2+ss->w[i]*rnn->config->NStates]-=rnn->lr*currDelta[j+2];
            rnn->W_ws[j+3+ss->w[i]*rnn->config->NStates]-=rnn->lr*currDelta[j+3];
            rnn->W_ws[j+4+ss->w[i]*rnn->config->NStates]-=rnn->lr*currDelta[j+4];
            rnn->W_ws[j+5+ss->w[i]*rnn->config->NStates]-=rnn->lr*currDelta[j+5];
            rnn->W_ws[j+6+ss->w[i]*rnn->config->NStates]-=rnn->lr*currDelta[j+6];
            rnn->W_ws[j+7+ss->w[i]*rnn->config->NStates]-=rnn->lr*currDelta[j+7];
        }
        for(int j = ((int)floor(rnn->config->NStates/8))*8; j < rnn->config->NStates; j++) {            
            rnn->W_ws[j+ss->w[i]*rnn->config->NStates]-=rnn->lr*currDelta[j];
        }
    }    
}

void computeClassIndexBackwardPass(RNN * rnn) {
    auto start = std::chrono::high_resolution_clock::now();
    auto elapsed = std::chrono::high_resolution_clock::now() - start;
    msCIBW += std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count()/1000;
    SentState * ss = rnn->ss;
    for(int i = ss->N-2; i >= 0; i--) {
        int trueClass = ss->cls_t[i];
        int A = rnn->classBoundaries[trueClass*2];
        int B = rnn->classBoundaries[trueClass*2+1];
        AT * currDelta = &ss->delta[i*rnn->config->NStates];
        AT * prevDelta = &ss->delta[(i+1)*rnn->config->NStates];
        AT * currDeriv = &ss->s_ds[i*rnn->config->NStates];
        int cls_true = ss->cls_t[i];
        AT * currentClsProb     = &ss->cls[i*rnn->config->NClasses];
        currentClsProb[cls_true] -= 1; 
        int idx_true = ss->idx_t[i];
        AT * currentIdxProb = &ss->idx[i*rnn->config->NIndices];
        currentIdxProb[idx_true] -= 1;
        for(int k = 0; k < rnn->config->NStates; k+=8) {
            currDelta[k+0]=0.0;
            currDelta[k+1]=0.0;
            currDelta[k+2]=0.0;
            currDelta[k+3]=0.0;
            currDelta[k+4]=0.0;
            currDelta[k+5]=0.0;
            currDelta[k+6]=0.0;
            currDelta[k+7]=0.0;
        }
        if(i==ss->N-2) {
            /*
             void cblas_dgemv(const enum CBLAS_ORDER order,
                 const enum CBLAS_TRANSPOSE TransA, const int M, const int N,
                 const double alpha, const double *A, const int lda,
                 const double *X, const int incX, const double beta,
                 double *Y, const int incY);
             
             */
            GEMV(CblasRowMajor, CblasTrans, B-A+1, rnn->config->NStates, 1.0, &rnn->W_si[A*rnn->config->NStates], rnn->config->NStates, &currentIdxProb[A], 1, 0.0, currDelta, 1);
            //GEMV(CblasRowMajor, CblasTrans, rnn->config->NIndices, rnn->config->NStates, 1.0, rnn->W_si, rnn->config->NStates, current_dL_dIdx, 1, 0.0, currDelta, 1);
            GEMV(CblasRowMajor, CblasTrans, rnn->config->NClasses, rnn->config->NStates, 1.0, rnn->W_sc, rnn->config->NStates, currentClsProb, 1, 1.0, currDelta, 1);
        } else {
            GEMV(CblasRowMajor, CblasTrans, B-A+1, rnn->config->NStates, 1.0, &rnn->W_si[A*rnn->config->NStates], rnn->config->NStates, &currentIdxProb[A], 1, 0.0, currDelta, 1);
            //GEMV(CblasRowMajor, CblasTrans, rnn->config->NIndices, rnn->config->NStates, 1.0, rnn->W_si, rnn->config->NStates, current_dL_dIdx, 1, 0.0, currDelta, 1);
            GEMV(CblasRowMajor, CblasTrans, rnn->config->NClasses, rnn->config->NStates, 1.0, rnn->W_sc, rnn->config->NStates, currentClsProb, 1, 1.0, currDelta, 1);
            GEMV(CblasRowMajor, CblasTrans, rnn->config->NStates, rnn->config->NStates, 1.0, rnn->W_ss, rnn->config->NStates, prevDelta, 1, 1.0, currDelta, 1);
        }        
        for(int j = 0; j < rnn->config->NStates; j+=8) {
            currDelta[j+0] = currDelta[j+0]*currDeriv[j+0];
            currDelta[j+1] = currDelta[j+1]*currDeriv[j+1];
            currDelta[j+2] = currDelta[j+2]*currDeriv[j+2];
            currDelta[j+3] = currDelta[j+3]*currDeriv[j+3];
            currDelta[j+4] = currDelta[j+4]*currDeriv[j+4];
            currDelta[j+5] = currDelta[j+5]*currDeriv[j+5];
            currDelta[j+6] = currDelta[j+6]*currDeriv[j+6];
            currDelta[j+7] = currDelta[j+7]*currDeriv[j+7];
        }
        //for(int j = (int)floor(rnn->config->NStates/8)*8; j < rnn->config->NStates; j+=1) {
        //   currDelta[j] = currDelta[j]*currDeriv[j];
        //}
                
    }
}

void computeClassIndexDerivatives(RNN * rnn) {
    //compute_dL_dIdx(rnn);
    //compute_dL_dCls(rnn);
    auto start = std::chrono::high_resolution_clock::now();
    computeClassIndexBackwardPass(rnn);
    auto elapsed = std::chrono::high_resolution_clock::now() - start;
    msCIBW += std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
    
    
    
    start = std::chrono::high_resolution_clock::now();
    compute_dL_dWws(rnn);
    elapsed = std::chrono::high_resolution_clock::now() - start;
    msWws += std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
    start = std::chrono::high_resolution_clock::now();
    compute_dL_dWss(rnn);
    elapsed = std::chrono::high_resolution_clock::now() - start;
    msWss += std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
    //compute_dL_dWsi(rnn);
    //compute_dL_dWsc(rnn); 
     start = std::chrono::high_resolution_clock::now();   
    update_Wsc_Wsi(rnn);
    elapsed = std::chrono::high_resolution_clock::now() - start;
    msWscWsi += std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
}


double computeMatrixNorm(double * A, int N) {
    double norm = 0.0;
    for(int i = 0; i < N; i++) {
        norm+=A[i]* A[i];// - A[i]*0.0001;
        //printf("A[%d]=%0.3f, dA[%d]=%0.3f ",i,A[i],i,dA[i]);
    }
    return norm;
}

void updateLossStatistics(RNN * rnn) {
    double tmpN = (double)rnn->config->NLosses;
    rnn->lastLosses[rnn->lossPtr] = rnn->ss->totalLoss;
    double tmp = 0.0;
    for(int i = 0; i < rnn->config->NLosses; i++) {
        tmp+=rnn->lastLosses[i];
    }
    rnn->avgLoss=tmp/tmpN;
    rnn->lossPtr++;
    if(rnn->lossPtr==rnn->config->NLosses) {
        rnn->lossPtr=0;
    }
}

void processClassIndexSent(RNN * rnn, int sentId) {
    auto start = std::chrono::high_resolution_clock::now();
    
    computeClassIndexForwardPass(rnn,sentId);
    
    auto elapsed = std::chrono::high_resolution_clock::now() - start;
    msForward += std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
    
    start = std::chrono::high_resolution_clock::now();
    
    computeClassIndexDerivatives(rnn);
    
    elapsed = std::chrono::high_resolution_clock::now() - start;
    msBackward += std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
    
    updateLossStatistics(rnn);
}

double getTotalLoss(RNN * rnn, bool useClassIndex) {
    double totalLoss = 0.0;
    for(int i = 0; i < rnn->config->NSentences; i++) {
        computeClassIndexForwardPass(rnn,i);
        totalLoss+=rnn->ss->totalLoss;
    }
    return totalLoss/(double)rnn->config->NSentences;
}

void epoch(RNN * rnn, int k, bool useClassIndex) {    
    auto start = std::chrono::high_resolution_clock::now();
    vector<int> sentOrder;
    for(int i = 0; i < rnn->config->NSentences; i++) {
        sentOrder.push_back(i);
    }
    random_shuffle(sentOrder.begin(),sentOrder.end());
    int j = 0;
    msBackward = 0;
    msForward = msCIBW = msWws = msWss = msWscWsi = 0;// msCIBW msWws msWss msWscWsi
    for(auto const &i: sentOrder) {
        processClassIndexSent(rnn,i);
        if((j+1)%20 == 0)
            printf("Epoch %d: %d/%d - Loss~%0.3f - %0.2fW/s\r",k,j+1,rnn->config->NSentences,rnn->avgLoss,(double)(j+1)*rnn->config->NSentenceLength/(double)(msForward+msBackward)*1000000.0);
        j++;
    }
    double totalLoss = getTotalLoss(rnn,useClassIndex);
    auto elapsed = std::chrono::high_resolution_clock::now() - start;
    long long totalTime = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
    printf("                                                     \r");
    printf("Epoch %d: LR=%0.4f T=%0.2fs Loss=%0.3f TFw=%lld TBw=%lld TCIBW=%lld TWws=%lld TWss=%lld TWscWsi=%lld\n",k,rnn->lr,
        (double)totalTime/1000000,totalLoss,msForward,msBackward,msCIBW,msWws,msWss,msWscWsi);
}

void readFile(string fileName, Config * cfg) {
    int sentenceEndId = nextMappingId++;
    wordMapping.insert(make_pair("</s>",sentenceEndId));
    wordCount.push_back(0);
    ifstream f;
    f.open(fileName.data());
    string word;
    string line;
    int tokens = 0;
    int maxLen = 0;
    if(f==NULL) {
        printf("File %s could not be opened.\n", fileName.data());
    }
    while(getline(f,line)) {
        istringstream iss(line);
        vector<int> * sent = new vector<int>();
        while(iss >> word) {
            //printf("Read word: %s\n", word.data());
            int id = 0;
            if (wordMapping.count(word) == 0) {
                id = nextMappingId++;
                wordMapping.insert(make_pair(word,id));
                wordCount.push_back(1);
            } else {
                id = wordMapping.at(word);
                wordCount[id] = wordCount[id] + 1;
            }
            tokens++;
            sent->push_back(id);
            
        }
        sent->push_back(sentenceEndId);
        if (sent->size() > maxLen) maxLen = sent->size();
        sentencesRaw.push_back(sent);
        //printf("Switching Lines...\n");
    }    
    wordCount[0] = sentencesRaw.size();
    cfg->NTokens = tokens+sentencesRaw.size(); // The sentence endings are not counted above
    cfg->NWords = (int) wordMapping.size();
    cfg->NMaxSent = maxLen;
}

void transformSentences(int sentLength = 30, int overlap = 5) {    
    int leftInCurrentSent = sentLength;
    int currentSentIdx = 0;
    vector<int> * currentSent = new vector<int>();
    vector<int> * nextSent;
    for(auto const &itraw : sentencesRaw) {
        currentSentIdx=0;
        while(currentSentIdx != itraw->size()) {
            if(leftInCurrentSent ==0) {
                leftInCurrentSent = sentLength-overlap;
                nextSent = new vector<int>();
                for(int i = 0;i < overlap; i++) {
                    nextSent->push_back(currentSent->at(sentLength-overlap+i));
                }
                sentences.push_back(currentSent);
                currentSent = nextSent;
            } 
            currentSent->push_back(itraw->at(currentSentIdx));
            currentSentIdx++;
            leftInCurrentSent--;
        }
    }
    // Make sure that the last sentence is padded with 0s so that they fit in a matrix
    while(currentSent->size() < sentLength) {
        currentSent->push_back(0);
    }
    sentences.push_back(currentSent);
}

int computeClasses(vector<int> &wordCounts, int NClasses, vector<pair<int,int>> &word2CI, int * &classBoundaries) {
    
    vector<int> indices(wordCounts.size());
    for(int i = 0; i != indices.size(); i++) {
        indices[i] = i;
    }
    
    sort(indices.begin(), indices.end(), [&wordCounts] (int idxA, int idxB) {return wordCounts[idxA] > wordCounts[idxB];});
        
    word2CI.resize(wordCounts.size());
    
    double totalCount = 0;
    for(auto &i : wordCounts) totalCount+=((double)i);
    
    double countPerClass = (double)totalCount/(double)NClasses;
    int nInCurrentClass = 0;
    int currentClass = 0;
    int wordCountsBinned = 0;
    classBoundaries = new int[NClasses * 2];
    classBoundaries[0] = 0;
    for(int i = 0; i < indices.size(); i++) {
        word2CI[indices[i]].first = currentClass;
        word2CI[indices[i]].second = nInCurrentClass;
        wordCountsBinned+=((double)wordCounts[indices[i]]);
        nInCurrentClass++;
        if(wordCountsBinned>=(1+currentClass)*countPerClass) {   
            classBoundaries[currentClass * 2+1] = nInCurrentClass-1;         
            // If we are in the last class, all words have to go inside...
            if(currentClass != NClasses-1) {
                currentClass++;   
                classBoundaries[currentClass * 2] = nInCurrentClass;    
            }
        }
    }
    classBoundaries[NClasses * 2-1] = nInCurrentClass-1;
    return nInCurrentClass;
}

int * packWords(vector<vector<int>*> &sentences, vector<pair<int,int>> & word2CI) {
    return NULL;
}

int * packSentences(vector<vector<int>*> &sentences, vector<pair<int,int>> & word2CI) {
    int NSentences = sentences.size();
    int sentenceLength = sentences[0]->size();
    int * labelStore = new int[NSentences*sentenceLength*3];
    int * currLine;
    for(int i = 0; i < NSentences; i++) {
        currLine = &labelStore[i*sentenceLength*3];
        for(int k = 0; k < sentenceLength; k++) {
            currLine[k] = sentences[i]->at(k);
            currLine[k+sentenceLength] = word2CI[sentences[i]->at(k)].first;
            currLine[k+2*sentenceLength] = word2CI[sentences[i]->at(k)].second;
        }
    }
    return labelStore;
}

void cblasTest();


int main(int argc, char**argv) {
    srand(123);
    
    RNN * rnn = new RNN;
    Config * cfg = new Config;
    rnn->config=cfg;
    cfg->NStates = 104; //100
    cfg->randMin = -0.1;
    cfg->randMax = 0.1;
    cfg->NLosses = 300;
    cfg->NClasses = 104;
    cfg->Overlap = 0;
    rnn->lr = 0.1;
    rnn->config->NEpochs = 15;
    debug = 0;
    readFile("/home/jan/NTU/MLDS/hw3/data/training_clean_1e5.txt",cfg);
    printf("Read data from file.\n");
    //readFile("/home/jan/NTU/MLDS/hw3/rnnc/RNNC/test.txt",cfg);
    cfg->NIndices = computeClasses(wordCount, cfg->NClasses, word2CIMap,rnn->classBoundaries);
    //cfg->NMaxSent =20; // Needs to be set after "read file"
    cfg->NSentenceLength = 60;
    if(cfg->NStates % 8 != 0) {
        printf("Error! The number of states must be divisible by 8, but was set to %d.\n",cfg->NStates);
        exit(1);
    }
    if(cfg->NClasses % 8 != 0) {
        printf("Error! The number of classes must be divisible by 8, but was set to %d.\n",cfg->NClasses);
        exit(1);
    }
    
    transformSentences(cfg->NSentenceLength,cfg->Overlap);
    rnn->labels = packSentences(sentences, word2CIMap);
    printf("Transformed and packed sentences.\n");
    //sentences = sentencesRaw;
    if(debug > 2) {
        for(auto const &itmap : wordMapping ) {
            printf("%d\t| C=%d,\tI=%d,\t#=%d, %s\n", itmap.second, word2CIMap[itmap.second].first,word2CIMap[itmap.second].second,wordCount[itmap.second], itmap.first.data());
        }
        for(auto const &itvec : sentencesRaw) {
            for(auto const &itv : *itvec) {
                printf("%d ",itv);
            }
            printf("\n");
        }
        for(auto const &itvec : sentences) {
            for(auto const &itv : *itvec) {
                printf("%d ",itv);
            }
            printf("\n");
        }
        for(int k = 0; k < cfg->NClasses*2; k++)
            printf("%d: %d  ",k,rnn->classBoundaries[k]);
        printf("\n");
    }    
        
    cfg->NSentences = sentences.size();
    printf("States=%d, Vocabulary=%d, Tokens=%d, Sentences=%d, Longest Sentence=%d, NClasses=%d, NIndices=%d\n", 
        cfg->NStates, cfg->NWords, cfg->NTokens, cfg->NSentences, rnn->config->NMaxSent, rnn->config->NClasses,
        rnn->config->NIndices);
    buildMatrices(rnn);
    initSentState(rnn);
    printf("Finished initialising matrices.\n");
    double totalLoss = 0.0;
    time_t timerStart;
    time (&timerStart);
    for(int i = 0; i < cfg->NSentences; i++) {
        //computeClassIndexForwardPass(rnn,i); //computeClassIndexForwardPass(rnn,i); computeForwardPass
        totalLoss+=rnn->ss->totalLoss;
    }
    time_t timerEnd;
    time (&timerEnd);
    printf("Total loss over all sentences: %0.3f (computed in %0.2fs)\n",totalLoss/(double)cfg->NSentences,difftime(timerEnd,timerStart)/(double)cfg->NSentences);
   
    for(int j = 1; j < rnn->config->NEpochs+1; j++) {
        epoch(rnn,j,true);
        if(j > 10) {
            rnn->lr *= 0.867;
        }
    }
    
    return 0;
}

void cblasTest() {
    double * A = new double[12];
    double * B = new double[12];
    double * C = new double[16];
    for(int i = 0; i < 12; i++) {
        A[i]=i;
        C[i]=-1;
    }
    for(int i = 0; i < 12; i++) {
        B[i]=i;
    }
    for(int i = 0; i < 12; i++) {
        printf("A[%d]=%0.0f\t",i,A[i]);
    }
    printf("\n");
    for(int i = 0; i < 12; i++) {
        printf("B[%d]=%0.0f\t",i,B[i]);
    }
    printf("\n");
    /*
     void cblas_dgemm(const enum CBLAS_ORDER Order, const enum CBLAS_TRANSPOSE TransA,
                 const enum CBLAS_TRANSPOSE TransB, const int M, const int N,
                 const int K, const double alpha, const double *A,
                 const int lda, const double *B, const int ldb,
                 const double beta, double *C, const int ldc);
     */
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, 4, 4, 3, 1.0, A, 3, B, 3, 0.0, C, 4);
    
    for(int i = 0; i < 16; i++) {
        printf("C[%d]=%0.0f\t",i,C[i]);
    }
    printf("\n");
}