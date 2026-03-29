### Program for ML Project ###
### Don Frenzel, Rohan Castillo, Lasaro Morell ###

###Imports.  
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

###Alternatively, using biopython might be better, especially since fastaframes didn't separate the columns correctly.  
from Bio import SeqIO

def fastaConverter(filename):
    dataList = []
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
