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

### Second is a group based on the alphabetical properties of the given peptides: Examples of such features are amino acid composition(AAC), occurrence, and bi-gram.

### Third is grouping based on PLMs which can generate numeric encoding of proteins.  (Look into PLM's).  

### Construct a number of features using a personal approach, iFeature, and another one.  Once those three features are set, use trainingDataLabels to confirm.
