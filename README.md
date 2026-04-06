#---Program for ML Project---#
#---Don Frenzel, Rohan Castillo, Lasaro Morell---#

###NOTES FOR DON: Run on Python311.  Tensorflow only works until 3.13.  Should not be a problem otherwise

###Imports.  
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

###Imports for tensorflow/Deep Learning (used primarily for CNN)
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from tensorflow.keras import layers
from keras.utils import to_categorical
plt.style.use('fivethirtyeight')

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
                    'label': int(entry.id[8]),
                    'sequence': str(entry.seq),
                    'length': len(entry.seq)})
        #figure out how to split the id into the boolean var and the word 'peptide'

    dataframe = pd.DataFrame(dataList)
    return dataframe

###Load data in, training/testing.  Split later for validation.  
trainingData = fastaConverter('Train.fasta')
testingData = fastaConverter('test.fasta')

###Important for functions, essentially acts as a library for amino acids.  
allColumns = ['A','C','D','E','F','G','H','I','K','L','M','N','P','Q','R','S','T','V','W','Y']

###Testing purposes only.  
sequences = trainingData.sequence

## Import stuff/methods to extract features from the sequence data.

## 3 Different Groups of Features Required.

## Could actually group into feature 'groups' and analyze different feature groups within the peptide to create a measure of whether they are poisonous or not.

## First goal will be to group based on physiochemical properties of Amino Acids.  Not sure how one would go about doing this.  [Rohan]

### Second is a group based on the alphabetical properties of the given peptides: Examples of such features are amino acid composition(AAC)(percentage of 
### each of standard 20 amino acids
### present in the sequence), occurrence (count of each amino acid within the sequence), and bi-gram (frequency of two adjacent amino acids, pair or dipeptide, in the sequence).
### so far, the first has been computed and loaded into a csv.  [Don][Fulfilled]

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
def getBigram(sequence,desiredColumns):
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

    ### Divide all vals of the retMatrix by the total pairs to normalize it.  Creates good nparray.  
    retMatrix = retMatrix/totalPairs

    ##return matrix as a numpy dataframe with the labels intact.  
    #retDF = pd.DataFrame(retMatrix, columns = desiredColumns, index = desiredColumns)
    return retMatrix

#KEEP THIS; NECESSARY as it is the basic, alphabetical order.  
#print(sequences.iloc[2])
#aacTest = getAAC(sequences.iloc[2], allColumns)
#print(aacTest)
        
#occTest = getOCC(sequences.iloc[2],allColumns)
#print(occTest)

#Bigrams returning now as nparrays.
#bigramTest = getBigram(sequences.iloc[2],allColumns)
#print(bigramTest)

###Load in Data for CNN - LOADED EARLIER INTO trainingData & testingData vars, respectively.  Data Must be converted.  Menu might be good perhaps?
###Merge eventually with getBigram.  
def bigramSeqs(data, desiredColumns):
    cSeqs = []
    for seq in data.sequence:
        #convSeq is returned as an nparray.
        convSeq = getBigram(seq, desiredColumns)
        ###works properly, at least returns.  
        cSeqs.append(convSeq)
    ###results in stack of size (n, m, m) for n mxm matrices, channel depth must be explicit.  
    result = np.stack(cSeqs,axis=0)

    ###Adds the extra 1 dimension necessary for CNN to work.  
    result = np.expand_dims(result, axis=-1) 
    return result

### Third is grouping based on PLMs which can generate numeric encoding of proteins.  (Look into PLM's).  


### Now, convert all sequences to biGramSeqs and then all labels to np vectors; make sure dimensions are good to work with tensorflow.  
xTrain = bigramSeqs(trainingData, allColumns)
xTest = bigramSeqs(testingData, allColumns)

yTrain = (trainingData.label).to_numpy()
yTrain = np.expand_dims(yTrain, axis=-1)#expands dim by 1 to represent channel depth

yTest = (testingData.label).to_numpy()
yTest = np.expand_dims(yTest,axis=-1)#expands dim by 1 to represent channel depth


###classification array; pos 0 is label 0, pos 1 is label 1.  
classification = ['Nontoxic','Toxic']

### convert labels into set of 2 nums for the neural net imput; basically [x,y] where each is a binary indicator of whether the class is present or not.  
yTrain_oneHot = to_categorical(yTrain)
yTest_oneHot = to_categorical(yTest)

### since data is autonormalized by function, there's no need to renormalize it.  

### DON NOTES: Now that features have been gotten based off of the sequence, I am going to implement a Convolutional Neural Network to process the bigram data,
###            since its output is a matrix.  Seems convenient and fulfills the Deep Learning Req.  

### Step 1: Format/Prep the data from the Bigram into a pandas dataframe in which the matrix of a sequence is stored along with its original labels.  Since we are testing
###         whether they are poisonous or not.  Format labels to fit.  Will be using tensorflow for this.  

### Step 2: Program the CNN and then run the data through in order to classify it properly.  

### Step 3: Output the raw data; use for prediction.  

###CNN Implementation

model = Sequential()

###Create first layer
model.add(layers.Input(shape=(20, 20, 1)))
model.add(Conv2D(32,(3,3),activation='relu'))

###Create Pooling layer
#model.add(MaxPooling2D(pool_size=(2,2)))

###Create second convolution layer
model.add(Conv2D(32,(3,3),activation='relu'))

###Create Second Pooling Layer
#model.add(MaxPooling2D(pool_size=(2,2)))

###Create Flattening Layer (reduces dimensionality to a linear array)
model.add(Flatten())

###Create a layer with 500 neurons.
model.add(Dense(1000,activation='relu'))

###Create a dropout layer
model.add(Dropout(0.5))

###Second layer of neurons
model.add(Dense(500,activation='relu'))

###
model.add(Dropout(0.5))

model.add(Dense(250,activation='relu'))

model.add(Dense(2,activation='softmax'))

###Compile the model
model.compile(loss = 'categorical_crossentropy',
              optimizer = 'adam',
              metrics = ['accuracy'])
###train le model

hist = model.fit(xTrain,yTrain_oneHot,
                 batch_size=256,
                 epochs = 10,
                 validation_split=0.2)


testAcc = model.evaluate(xTest, yTest_oneHot)[1]
print(testAcc)

###Accuracy Visualization of Convolutional Neural Network Model.  
plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])
plt.axhline(y=testAcc, color='gold', linestyle='-')
plt.title('Model Accuracy, CNN')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train','Val','Test'],loc='upper left')
plt.show()

###Test acc of model with a value:
testPept = [['GEEELQENQELIRKSN']]
testPept = pd.DataFrame(testPept,columns=['sequence'])

testPeptReg = bigramSeqs(testPept, allColumns)

predictions = model.predict(testPeptReg)
print(predictions)

##For testing purposes only; picks one datapoinert and rolls with it.  
n = 5063
print(trainingData.sequence.iloc[n])
print(yTrain[n][0])
print('The peptide class is: ', classification[yTrain[n][0]])
print('The binary label is:',yTrain_oneHot[n])
