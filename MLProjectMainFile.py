### Program for ML Project ###
### Don Frenzel, Rohan Castillo, Lasaro Morell ###

###Imports.  
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

###Fastaframes is like pandas but specific to bioInformatics for dataframe loading. 
from fastaframes import to_df
from fastaframes import fasta_to_entries, entries_to_fasta

###Alternatively, using biopython might be better, especially since fastaframes didn't separate the columns correctly.  


trainFrame = to_df("Train.fasta")
#print(trainFrame)
print(trainFrame.head())

##trainFrame does hold all of the data but the issue is that it hasn't been separated at all to match our needs for the confirmed 1/0 classification since it isn't isolated.  
for entry in fasta_to_entries("Train.fasta"):
    print(entry.protein_id, entry.protein_sequence, entry.unique_identifier)
'''
# Filter and write back
entries = [e for e in fasta_to_entries("Train.fasta") if e.organism_name == "Homo sapiens"]
entries_to_fasta(entries, output_file="human_only.fasta")
'''
