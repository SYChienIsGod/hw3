# -*- coding: utf-8 -*-
"""
Created on Sun May 17 09:38:55 2015

@author: jan
"""

import cPickle
import numpy
with open('data/test.cpk','rb') as f:
    testData = cPickle.load(f)
with open('data/test_data_model_4.txt','rb') as f:
    i = 0
    j = 0
    correctAnswers = 0
    for line in f:    
        if i%5 == 0:
            res = numpy.zeros(shape=(5,))
        res[i%5] = float(line)
        if i%5 == 4:
            ans = numpy.argmax(res)
            if ans == testData[j][1]:
                correctAnswers+=1
            j+=1
        i+=1

print '#Correct Answers: {} ({:0.2f}%)'.format(correctAnswers,correctAnswers/1040.0*100)