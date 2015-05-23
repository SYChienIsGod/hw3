# Homework 3
## Assumed file layout
`hw3`
Python scripts.

`hw3/data`
Training/testing data.

`hw3/data/Holmes_Training_Data`
The original Project Gutenberg data.

`hw3/rnnc`
The C++ implementation.

## Preprocessing
First, the data is preprocessed with a python script that extracts the relevant information (i.e. deletes the Project Gutenberg headers) and puts it all into one file:

`python filter_training_data.py`

Afterwards, the shell script (slightly modified from the suggestion in the facebook group) is used:

`./pre.sh < training_sents.txt > training_clean.txt`

## C++ Implementation

The C++ implementation cannot not yet be build with the accompanied make file (I use netbeans at that has a strange make file implementation). It can compete in language modelling performance with RNNLM but is still a little slower. Parameters have to be set in the C++ file.

