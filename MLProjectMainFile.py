### Program for ML Project ###
### Don Frenzel, Rohan Castillo, Lasaro Morell ###

###Imports.  
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

###Alternatively, using biopython might be better, especially since fastaframes didn't separate the columns correctly.  
from Bio import SeqIO

### Take fasta file and basically turn it into the ideal pandas dataframe that can be used to actually prep the data for the ML that's gonna happen. 
def fastaConverter(filename):
    dataList = []
    ###Opens it as a handle which is just slightly easier to do and then uses the SeqIO part of biopython to sort through it and give a more modifiable result.  
    with open(filename,"r") as handle:
        for entry in SeqIO.parse(handle, "fasta"):
             dataList.append({
                    'id': entry.id,
                    'sequence': str(entry.seq),
                    'length': len(entry.seq)})
        #figure out how to split the id into the boolean var and the word 'peptide'

    dataframe = pd.DataFrame(dataList)

    return dataframe

trainingData = fastaConverter('Train.fasta')
print(trainingData)
