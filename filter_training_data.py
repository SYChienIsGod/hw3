# -*- coding: utf-8 -*-

import os
import re

# Reads all the training data, filters and saves it to one file.

trainingFolder = 'data/Holmes_Training_Data'
trainingSentences = 'data/training_sents.txt'

def processFile(fileName):
    textMode = False
    emptyLines = 0
    sents = list()
    paragraph = ''
    with open(fileName,'rb') as f:
        for line in f:
            #if line.startswith('*END*THE SMALL PRINT! FOR PUBLIC DOMAIN ETEXTS*'):
            #    textMode = True
            #    continue
            if '*END*THE SMALL PRINT!' in line:
                textMode = True
                continue
            if not textMode:
                continue
            if line.isupper(): # Skip over headings
                continue
            if len(line.strip()) == 0: # Count empty lines to find paragraphs
                emptyLines+=1
                if paragraph != '':
                    sents.append(paragraph)
                    paragraph = ''
                continue
            if isNumber(line): # Chapter beginnings etc.
                continue
            paragraph+=line.strip() + ' '
            
    if not textMode:
        print 'Error: Did not find text mode marker in file {}'.format(fileName)
    return sents
        
def isNumber(line):
    if not line.isupper(): # Seems not to be a heading (e.g. chapter)
        return False
    if line.startswith('CHAPTER'):
        return True
    if line.isdigit():
        return True
    if re.match(r'(CHAPTER\s)?[IVX]*',line)!=None:
        return True
    return False
    

files = os.listdir(trainingFolder)

sents = list()

for f in files:
    sents.append(processFile(trainingFolder + '/'+f))
    
with file(trainingSentences,'wb') as f:
    for sentences in sents:
        for sent in sentences:
            f.write(sent+'\n')