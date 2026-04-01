### Program for ML Project ###
### Don Frenzel, Rohan Castillo, Lasaro Morell ###

###Imports.  
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os


####Get iFeature running in here.  

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
print(trainingData)

### Import stuff/methods to extract features from the sequence data.

### 3 Different Groups of Features Required.  1 Deep Learning Architecture also Required (DeepAmp preferred).

### Could actually group into feature 'groups' and analyze different feature groups within the peptide to create a measure of whether they are poisonous or not.

### 3 Groups of features as determined by the papers.  iFeature can be used.  Look into use.

### First goal will be to group based on physiochemical properties of Amino Acids.  Not sure how one would go about doing this.  

### Second is a group based on the alphabetical properties of the given peptides: Examples of such features are amino acid composition(AAC)(percentage of 
### each of standard 20 amino acids
### present in the sequence), occurrence (count of each amino acid within the sequence), and bi-gram (frequency of two adjacent amino acids, pair or dipeptide, in the sequence).
### so far, the first has been computed and loaded into a csv.  

### Create functions for getting AAC (Count of a given amino acid within the sequence and then divide it by the length of the sequence, do for all and output as a 
### dict that can then be converted into a dataframe itself.  

### Ocurrence: piggyback off of AAC by just grabbing the pure counts of each; output on the side; count whether an amino acid from the library
### is present using boolean variables.  All zeroes vector (use np.zeroes()), where it is a single row, each column is a separate amino acid.  

### Bi-Gram: Get bi-gram or 2-gram sequences of each of the Amino Acids in the peptide, i.e. 2 character.  Can use count vectorizer in scikit learn for this, then count
### and normalize the presence of each of the 2-grams within the sequence.  

### Third is grouping based on PLMs which can generate numeric encoding of proteins.  (Look into PLM's).  

### Construct a number of features using a personal approach, iFeature, and another one.  Once those three features are set, use trainingDataLabels to confirm.

