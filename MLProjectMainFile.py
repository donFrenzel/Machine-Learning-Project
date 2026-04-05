### Program for ML Project ###
### Don Frenzel, Rohan Castillo, Lasaro Morell ###

###Imports.  
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from collections import Counter
import sklearn
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

###Alternatively, using biopython might be better, especially since fastaframes didn't separate the columns correctly.  
from Bio import SeqIO

### Take fasta file and basically turn it into the ideal pandas dataframe that can be used to actually prep the data for the ML that's gonna happen.
### NOTE LABEL '1' is toxic, LABEL '0' is nontoxic.  
def fastaConverter(filename):
    dataList = []
    ###Opens it as a handle which is just slightly easier to do and then uses the SeqIO part of biopython to sort through it and give a more modifiable result.  
    with open(filename,"r") as handle:
        for entry in SeqIO.parse(handle, "fasta"):
             dataList.append({
                    'label': entry.id[8],
                    'sequence': str(entry.seq),
                    'length': len(entry.seq)})
        #figure out how to split the id into the boolean var and the word 'peptide'

    dataframe = pd.DataFrame(dataList)
    return dataframe

###Testing purposes only.  
trainingData = fastaConverter('Train.fasta')
sequences = trainingData.sequence

### Import stuff/methods to extract features from the sequence data.

### 3 Different Groups of Features Required.  1 Deep Learning Architecture also Required (DeepAmp preferred).

### Could actually group into feature 'groups' and analyze different feature groups within the peptide to create a measure of whether they are poisonous or not.

### First goal will be to group based on physiochemical properties of Amino Acids.  Not sure how one would go about doing this.  

### Second is a group based on the alphabetical properties of the given peptides: Examples of such features are amino acid composition(AAC)(percentage of 
### each of standard 20 amino acids
### present in the sequence), occurrence (count of each amino acid within the sequence), and bi-gram (frequency of two adjacent amino acids, pair or dipeptide, in the sequence).
### so far, the first has been computed and loaded into a csv.  

### Create functions for getting AAC (Count of a given amino acid within the sequence and then divide it by the length of the sequence, do for all and output as a 
### dict that can then be converted into a dataframe itself.  Returns a dataframe where each column is the normalized count of each amino acid.  
def getAAC(sequence,desiredColumns):
    seqList = list(sequence)
    seqLen = len(sequence)
    counts = Counter(seqList)  ##outputs as dict type.  Convert to tuples and sort by key.

    ##normalize the data and then make it a dict at the same time.  
    counts = {key: count / seqLen for key, count in counts.items()}

    ## puts the counts data into a new dataframe and reindexes by the desired columns. 
    countsDF = pd.DataFrame([counts]).reindex(columns=desiredColumns)
    countsDF = countsDF.fillna(0)

    return countsDF

    
### Ocurrence: piggyback off of AAC by just grabbing the pure counts of each; output on the side; count whether an amino acid from the library
### is present using boolean variables.  Returns a pandas dataframe where each column is an amino acid.  
def getOCC(sequence,desiredColumns):
    occDict = {}
    seqList = list(sequence)

    ### does basic bucketing; if no bucket exists, creates a bucket, but only if in sequence.  Counts binary wise.  
    for aa in seqList:
        if aa not in occDict:
            occDict[aa]=1
    ###now convert to a dataframe
    occDF = pd.DataFrame([occDict]).reindex(columns=desiredColumns)
    occDF = occDF.fillna(0).astype(int)
    return occDF


### Bi-Gram: Get bi-gram or 2-gram sequences of each of the Amino Acids in the peptide, i.e. 2 character.  Can use count vectorizer in scikit learn for this, then count
### and normalize the presence of each of the 2-grams within the sequence.  Returns pandas dataframe that represents a matrix where each [i,j] is a bigram.  
def getBiGram(sequence,desiredColumns):
    ###Get bi-grams present in sequence.
    aas = 'ACDEFGHIKLMNPQRSTVWY'
    
    ##Gets all aas as a dictionary.  
    aasDict = {aa: i for i, aa in enumerate(aas)}

    ##list of all possible bigrams
    retMatrix = np.zeros((20,20))

    ### loop through in range of the sequence-1 (num of bigram pairs that can exist) and then increase the row, column value which is of the pair in the matrix by 1(for the sake of count)
    ### total number of permutations given by the sequence.  seq is of length 28, zB, so the number of pairs is 27, as the number of bigrams is always L-1 for a set of chars L.
    ### it grabs the row and column values, row being i and column being the next i to form a pair, which it then searches the value of in aas dict to get the row/column values for.  
    for i in range(len(sequence)-1):
        row = aasDict[sequence[i]]
        column = aasDict[sequence[i+1]]
        retMatrix[row,column]+=1

    totalPairs =  len(sequence)-1      

    ### Divide all vals of the retMatrix by the total pairs to normalize it.  
    retMatrix = retMatrix/totalPairs

    ##return matrix as a numpy dataframe with the labels intact.  
    retDF = pd.DataFrame(retMatrix, columns = desiredColumns, index = desiredColumns)
    return retDF

###KEEP THIS; NECESSARY as it is the basic, alphabetical order.  
allColumns = ['A','C','D','E','F','G','H','I','K','L','M','N','P','Q','R','S','T','V','W','Y']

aacTest = getAAC(sequences.iloc[2], allColumns)
print(aacTest)
        
occTest = getOCC(sequences.iloc[2],allColumns)
print(occTest)


bigramTest = getBiGram(sequences.iloc[2],allColumns)
print(bigramTest)

### Third is grouping based on PLMs which can generate numeric encoding of proteins.  (Look into PLM's).  

### Construct a number of features using a personal approach, iFeature, and another one.  Once those three features are set, use trainingDataLabels to confirm.
