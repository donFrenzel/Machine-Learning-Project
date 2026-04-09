#---Program for ML Project---#
#---Don Frenzel, Rohan Castillo, Lasaro Morell---#

###NOTES FOR DON: Run on Python311.  Tensorflow only works until 3.13.  Should not be a problem otherwise

###Imports.  
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools
from collections import Counter


plt.style.use('fivethirtyeight')

###Convolutional Neural Network Imports PyTorch
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import TensorDataset, DataLoader

import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim

###Define Neural network Class
class convNeuralNet(nn.Module):
    def __init__(self):
        super().__init__()

        ###define first convolution, initial input layer is 1.  3 by 3 windows.  256 Filters.s
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3)

        ###Pooling layer: 
        self.pool1 = nn.MaxPool2d(2,2)

        ###Second convolutional layer and pooling layer.  
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3)
        #self.pool2 = nn.MaxPool2d(2,2)

        #self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=2)
        #self.pool3 = nn.MaxPool2d(2,2)

        self.flatten = nn.Flatten()

        self.dropout = nn.Dropout(p=0.3)

        self.fc1 = nn.Linear(in_features=2048, out_features=1000)
        self.fc2 = nn.Linear(in_features=1000, out_features=500)
        self.fc3 = nn.Linear(in_features=500, out_features=250)

        ###Add linear output layer; needs to end in 2 values.  
        self.out = nn.Linear(in_features=250, out_features=2)

    ###Forward Pass makes sure that it can go through
    def forward(self, x):
        x = F.relu(self.conv1(x))##runs x, img batch through first convolution.

        x = F.relu(self.conv2(x))##runs x, img batch through first convolution.

        x = self.pool1(x)

        #x = self.pool2(x)

        #x = F.relu(self.conv3(x))##runs x, img batch through first convolution.
        #x = self.pool3(x)
        
        x = self.flatten(x)

        x = F.relu(self.fc1(x))
        x = self.dropout(x)

        x = F.relu(self.fc2(x))
        x = self.dropout(x)

        x = F.relu(self.fc3(x))
        x = self.dropout(x)

        x = self.out(x)
        return x


###Set device to GPU if possible but make sure it can be set to CPU if need be
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print('Running on: ',device)

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


### CNN function takes input of training & testing data rewritten in Pytorch:
def CNN(trainingData, testingData):
    ### Now, convert all sequences to biGramSeqs and then all labels to np vectors; make sure dimensions are good to work with tensorflow.  Since data is
    ### autonormalized by function pipeline, there's no need to renormalize it.  
    classification = ['Nontoxic','Toxic']
    batchSize = 8
    
    xTrain = bigramSeqs(trainingData, allColumns)
    xTest = bigramSeqs(testingData, allColumns)

    yTrain = (trainingData.label).to_numpy()
    yTrain = np.expand_dims(yTrain, axis=-1)#expands dim by 1 to represent channel depth

    yTest = (testingData.label).to_numpy()
    yTest = np.expand_dims(yTest,axis=-1)#expands dim by 1 to represent channel depth

    ###run ToTensor on all of these.
    #print(len(xTrain))
    #print(len(xTest))
    #print(xTrain[5062], yTrain[5062])

    ###Need to pair training and test with their labels.
    ###convert to tensors.
    xTrainTensor =torch.from_numpy(xTrain)
    yTrainTensor = torch.from_numpy(yTrain)
    xTestTensor = torch.from_numpy(xTest)
    yTestTensor = torch.from_numpy(yTest)

    trainSet = TensorDataset(xTrainTensor, yTrainTensor)
    testSet = TensorDataset(xTestTensor, yTestTensor)
    
    #print("Matrix", trainSet[0][0], "\nLabels", trainSet[0][1]) #WORKS PROPERLY
    print("Matrix", testSet[0][0], "\nLabels", testSet[0][1])
    
    ###Split training set into training and validation.  Make validation roughly same amount in training data as test data.  1126/6387 or ~17.6%
    trainSet, valSet = torch.utils.data.random_split(trainSet, [5261, 1126])
    #print("Num, batches in trainingSet: ", int(5261/batchSize)) ##Will run in 657 batches total.  NOTE: Batches help with splitting should it run on a GPU instead.  
    #print("Num, batches in validationSet: ", int(1126/batchSize))##Will run in 140 batches total.

    ###Loaders load the data into proper datasets for the neural network to learn off of.  
    trainLoader = torch.utils.data.DataLoader(trainSet, batch_size = batchSize, shuffle = True, num_workers = 0)
    valLoader = torch.utils.data.DataLoader(valSet, batch_size = batchSize, shuffle = True, num_workers = 0)
    testLoader = torch.utils.data.DataLoader(testSet, batch_size = batchSize, shuffle = True, num_workers = 0)
    
    net = convNeuralNet()
    print(net.to(device))

    ###Loads training data and labels into the model.  Shows training info.  
    for i, data in enumerate(trainLoader):
        inputs, labels = data[0].to(device),data[1].to(device)
        ###Needed to change order due to issues; pyTorch expects order [batch, channels, height, width] and I have [Batch, Height, width, channels].
        ###Additionally, was processed as double so converted to float. 
        inputs = inputs.permute(0, 3, 1, 2).float()  
        print(f'input shape: {inputs.shape}')
        print(f'after network shape: {net(inputs).shape}')
        break


    ###Def loss function and optimizer.  Using Cross-Entropy Loss and Adam optimization. 
    criterion = nn.CrossEntropyLoss()  
    optimizer = optim.Adam(net.parameters(), lr=0.0001)

    ###Training loop:
    epochs = 10

    ###Track Accuracy and Loss over epochs
    accVOverEpochs = []
    accTOverEpochs = []
    for epochIndex in range(epochs):
        print("Epoch :",epochIndex+1)


        net.train(True) #set training mode
        running_loss = 0.0
        running_accuracy = 0.0
        avgAccTOverEpoch = []

        #iterate over dataloader to train the data in the epoch.
        for batch_index, data in enumerate(trainLoader):
            inputs, labels = data[0].to(device), data[1].to(device)
            inputs = inputs.permute(0, 3, 1, 2).float()
            labels = labels.squeeze() # Removes dimensions of size 1

            optimizer.zero_grad() #reset gradients.

            outputs = net(inputs) #shape: [batch_size, 2] grab highest value and look at index
            correct = torch.sum(labels == torch.argmax(outputs, dim=1)).item()
            running_accuracy += correct/batchSize

            loss = criterion(outputs, labels) ##utilizes loss function
            running_loss += loss.item()
            loss.backward() #backpropagates to learn model and then takes an optimizer's step
            optimizer.step()
            
            if batch_index % 200 == 199:
                avgLossOverBatches = running_loss/200
                avgAccOverBatches = (running_accuracy/200)*100
                print(f'Batch {batch_index+1}, Loss: {avgLossOverBatches:.3f}, Accuracy: {avgAccOverBatches:.1f}%')
                avgAccTOverEpoch.append(avgAccOverBatches)
                #print(avgAccTOverEpoch)
                #reset running accuracy and loss back to zero for start of next batch
                running_accuracy = 0.0
                running_loss = 0.0

        accTAvg = sum(avgAccTOverEpoch)/len(avgAccTOverEpoch)
        accTOverEpochs.append(accTAvg)
            
        print()

        ###Validation step:  Very similar to training data
        net.train(False) #disable training
        runningLossV = 0.0
        runningAccuracyV = 0.0

        for i, data in enumerate(valLoader):
            inputsV, labelsV = data[0].to(device), data[1].to(device)
            inputsV = inputsV.permute(0, 3, 1, 2).float()
            labelsV = labelsV.squeeze()

            with torch.no_grad():
                outputsV = net(inputsV)
                correctV = torch.sum(labelsV == torch.argmax(outputsV, dim=1)).item()
                runningAccuracyV += correctV/batchSize
                lossV = criterion(outputsV, labelsV)
                runningLossV = lossV.item()
                
        avgLossVOverBatches = runningLossV/len(valLoader)
        avgAccVOverBatches = (runningAccuracyV/len(valLoader))*100

        ###append data 3 times to match the batch data
        accVOverEpochs.append(avgAccVOverBatches)

        print(f'Validation Loss: {avgLossVOverBatches:.3f}, Validation Accuracy: {avgAccVOverBatches:.1f}%')


        print('*****************************************************')
        print()
        
    
    print("TRAINING COMPLETE!!!")

    print("BEGIN TESTING\n")

    ###Start testing; gets number of correct classifications and total classifications
    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for i, data in enumerate(testLoader):
            seq, labels = data[0],data[1]
            seq = seq.permute(0, 3, 1, 2).float() 
            # calculate outputs by running images through the network
            outputs = net(seq)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted.flatten() == labels.flatten()).sum().item()

    testAcc = 100* correct // total

    ###Test prints 
    print(f"Final Correct: {correct}")
    print(f"Final Total: {total}")
    print(f"Label Shape: {labels.shape}")
    print(f"Predicted Shape: {predicted.shape}")

    ###Prints total accuracy that will be present later on graph.  
    print(f'Accuracy of the network on the test set: {testAcc} %')
    ###Now that it's done, you can grab the data from the epochs and plot them like before:

    plt.plot(accTOverEpochs)
    plt.plot(accVOverEpochs)
    plt.axhline(y=testAcc, color='gold', linestyle='--')
    plt.title('Model Accuracy, CNN')
    plt.ylabel('Accuracy [%]')
    plt.xlabel('Epochs')
    plt.legend(['Train','Val','Test'],loc='lower right')
    plt.show()
###Create basic text interface:

###Choice of feature Eval + brief explanation of each ML method before a confirmation.  Once "confirmed, runs the method, then outputs the output of the function.
### Idea is for it to be a user menu; mostly for demo use during presentation of the project.  



CNN(trainingData,testingData)
