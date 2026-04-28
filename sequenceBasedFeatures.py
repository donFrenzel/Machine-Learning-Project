#---Program for ML Project---#
#---Don Frenzel, Rohan Castillo, Lasaro Morell---#

###NOTES FOR DON: Run on Python311.  Tensorflow only works until 3.13.  Should not be a problem otherwise - THIS IS FOR INITIAL VERSION
###               It may now run on up-to-date versions of Python, such as 3.14, and is no longer restricted as it was previously due to use of PyTorch.  

###Imports.  
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools
from collections import Counter

###Convolutional Neural Network Imports PyTorch
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

###Imports for AdaBoost Pipeline
import sklearn
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import cross_val_score



###Define Neural network Class
class convNeuralNet(nn.Module):
    def __init__(self):
        super().__init__()

        ###define first convolution, initial input layer is 1.  3 by 3 windows.  256 Filters.s
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, padding=1)
        ###Batch normalization recenters and rescales data
        self.bn1 = nn.BatchNorm2d(64)
        ###Second convolutional layer.  128 Filters from 64
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        ###Pooling layer, pools into several pooled maps.  
        self.pool1 = nn.MaxPool2d(2,2)

        ###Dropout basically is the chance that any given value will be dropped out of the data.  
        self.dropout_spatial = nn.Dropout2d(0.2)

        ### Linear Layer; linears are fully connected and perform final classification
        self.fc1 = nn.Linear(128 * 10 * 10, 512)
        ###Batch normalization
        self.bn_fc1 = nn.BatchNorm1d(512)
        ### Second linear layer.  
        self.fc2 = nn.Linear(512, 128)

        ###Larger dropout before output.  
        self.dropout = nn.Dropout(p=0.5)

        ###Add linear output layer; needs to end in 2 values.
        self.out = nn.Linear(in_features=128, out_features=2)

    ###Forward Pass makes sure that it can go through
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))##runs batch through first convolution.
        x = self.pool1(F.relu(self.bn2(self.conv2(x)))) ###'' second convolution
        x = self.dropout_spatial(x)

        ###Flattening layer; basically takes the whole thing and just makes it a singular 1 dimensional for compatibility with the full 
        x = torch.flatten(x, 1)

        x = F.relu(self.bn_fc1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
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
    countsDF = countsDF.fillna(0).astype(int) #Handles NaN data.  

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
    occDF = occDF.fillna(0).astype(int) #Handles NaN values.  
    return occDF

###OCCSeqs and AAC Seqs basically do the same thing as bigramSeqs; loop through data and convert, then output as a df.  In essence, they are the same function.  
def OCCSeqs(data, desiredColumns):
    ###Def list for concatenation
    oSeqs = []
    ###
    for seq in data.sequence:
        convSeq = getOCC(seq, desiredColumns)
        oSeqs.append(convSeq)

    oSeqsOutput = pd.concat(oSeqs, ignore_index=True)
    ###rets dataframe of converted sequences.  
    return oSeqsOutput

def AACSeqs(data, desiredColumns):
    aSeqs = []
    for seq in data.sequence:
        convSeq = getAAC(seq, desiredColumns)
        aSeqs.append(convSeq)
    aSeqsOutput = pd.concat(aSeqs, ignore_index=True)
    return aSeqsOutput



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

###Create a function which checks whether extra amino acids are present (U,X,Z)
def checkUXZ(data):
    countU = 0
    countX = 0
    countZ= 0
    ###Tests ocurrence of u, x, and z and totals them.  
    for i, seq in enumerate(data):
        occurrence = getOCC(seq, allColumns)
        if 'u' in occurrence.columns:
            countU+=1
        if 'x' in occurrence.columns:
            countX+=1
        if 'z' in occurrence.columns:
            countZ+=1
    ###Print out the counts.
    print(countU,countX,countZ)

###Check if U, X, or Z are present in any of the documents
print('Counts respectively of U, X, & Z amino acids in training and testing data:')
checkUXZ(sequences)
checkUXZ(testingData.sequence)

### Third is grouping based on PLMs which can generate numeric encoding of proteins.  (Look into PLM's).  


### CNN function takes input of training & testing data rewritten in Pytorch:
def CNN(trainingData, testingData, numEpochs):
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

    ###Need to pair training and test with their labels.
    ###convert to tensors.
    xTrainTensor =torch.from_numpy(xTrain)
    yTrainTensor = torch.from_numpy(yTrain)
    xTestTensor = torch.from_numpy(xTest)
    yTestTensor = torch.from_numpy(yTest)

    ###Creates the training and test Set
    trainSet = TensorDataset(xTrainTensor, yTrainTensor)
    testSet = TensorDataset(xTestTensor, yTestTensor)
    
    ###Split training set into training and validation.  Make validation roughly same amount in training data as test data.  80/20 split.  
    trainSet, valSet = torch.utils.data.random_split(trainSet, [5110, 1277])

    ###Loaders load the data into proper datasets for the neural network to learn off of.  
    trainLoader = torch.utils.data.DataLoader(trainSet, batch_size = batchSize, shuffle = True, num_workers = 0)
    valLoader = torch.utils.data.DataLoader(valSet, batch_size = batchSize, shuffle = True, num_workers = 0)
    testLoader = torch.utils.data.DataLoader(testSet, batch_size = batchSize, shuffle = True, num_workers = 0)

    ###Defines the net for use and also prints it to the device 
    net = convNeuralNet()
    print(net.to(device))
    
    ###Def loss function and optimizer.  Using Cross-Entropy Loss and Stochastic Gradient Descent optimization. Weight decay added also to make sure to avoid overfititng. 
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)  
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum = 0.9, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5) 

    ###Training loop:
    epochs = numEpochs

    ###Track Accuracy and Loss over epochs
    accVOverEpochs = []
    accTOverEpochs = []
    ###Training/Validation loop
    for epochIndex in range(epochs):
        print("Epoch :",epochIndex+1)

        
        net.train(True) #set training mode
        ###Reset epoch values
        running_loss = 0.0
        running_accuracy = 0.0
        avgAccTOverEpoch = []
        epochTotal = 0

        #iterate over dataloader to train the data in the epoch.
        for batch_index, data in enumerate(trainLoader):
            inputs, labels = data[0].to(device), data[1].to(device)
            inputs = inputs.permute(0, 3, 1, 2).float() ###NECESSARY; MUST BE REORDERED AS SUCH TO AVOID ERROR
            labels = labels.view(-1).long() # Ensures proper shape to avoid error. 

            #reset gradients.
            optimizer.zero_grad() 

            #shape: [batch_size, 2] grab highest value and look at index
            outputs = net(inputs)

            ###Gets the sum of the correct outputs & keep tabs on the general accuracy.  
            correct = torch.sum(labels == torch.argmax(outputs, dim=1)).item() 
            running_accuracy += correct/batchSize 

            ### Calculate loss function and calculates loss value
            loss = criterion(outputs, labels) 
            running_loss += loss.item()

            ###backpropagates to learn model and then takes an optimizer's step
            loss.backward() 
            optimizer.step()

            epochTotal += labels.size(0)

            ###Checks when the batches reach 200 so that it can print regular updates on the batch loss and batch accuracy within the epoch.  
            if batch_index % 200 == 199:
                avgLossOverBatches = running_loss/200
                avgAccOverBatches = (running_accuracy/200)*100
                print(f'Batch {batch_index+1}, Loss: {avgLossOverBatches:.3f}, Accuracy: {avgAccOverBatches:.1f}%')
                avgAccTOverEpoch.append(avgAccOverBatches)

                ###reset running accuracy and loss back to zero for start of next batch
                running_accuracy = 0.0
                running_loss = 0.0
        ###Gets the average sum of the batch accuracies so that it acts as the epoch's general accuracy for the charting at the end.  
        accTAvg = sum(avgAccTOverEpoch)/len(avgAccTOverEpoch)
        accTOverEpochs.append(accTAvg)
        
        print()

        ###Validation step:  Very similar to training data
        net.train(False)  ### Disable training.  
        runningLossV = 0.0
        runningAccuracyV = 0.0

        ###Load through data in the same way tothe 
        for i, data in enumerate(valLoader):
            inputsV, labelsV = data[0].to(device), data[1].to(device)
            inputsV = inputsV.permute(0, 3, 1, 2).float()
            labelsV = labelsV.view(-1).long() 

            ###Without calculating the gradient; saves memory.  Then grabs the accuracy values for the validation set by running them through that epoch's model version.  
            with torch.no_grad():
                outputsV = net(inputsV)

                ###Records num correct, running accuracy, the loss, and the running loss for the validation set.  
                correctV = torch.sum(labelsV == torch.argmax(outputsV, dim=1)).item()
                runningAccuracyV += correctV/batchSize
                lossV = criterion(outputsV, labelsV)
                runningLossV = lossV.item()
        ###Gets the average loss and accuracy over the batches   .    
        avgLossVOverBatches = runningLossV/len(valLoader)
        avgAccVOverBatches = (runningAccuracyV/len(valLoader))*100

        ###Appends the avg accuracy for the validation set to the list of them for data viz later.  
        accVOverEpochs.append(avgAccVOverBatches)
        scheduler.step() ###advance the scheduler

        print(f'Validation Loss: {avgLossVOverBatches:.3f}, Validation Accuracy: {avgAccVOverBatches:.1f}%')


        print('*****************************************************')
        print()
        
    
    print("TRAINING COMPLETE!!!")

    print("BEGIN TESTING\n")

    ###Start testing; gets number of correct classifications and total classifications.  Resets correct var.  
    correct = 0
    total = 0
    #### since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for i, data in enumerate(testLoader):
            seq, labels = data[0],data[1]
            seq = seq.permute(0, 3, 1, 2).float() 
            #### calculate outputs by running images through the network
            outputs = net(seq)
            #### the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted.flatten() == labels.flatten()).sum().item()

    ###Gets total test accuracy
    testAcc = 100* correct // total

    ###Test prints 
    print(f"Final Correct: {correct}")
    print(f"Final Total: {total}")
    print(f"Label Shape: {labels.shape}")
    print(f"Predicted Shape: {predicted.shape}")

    ###Prints total accuracy that will be present later on graph.  
    print(f'Accuracy of the network on the test set: {testAcc} %')
    
    ###Now that it's all done and over, grab the data from the epochs and plot them like before:
    plt.plot(accTOverEpochs)
    plt.plot(accVOverEpochs)
    plt.axhline(y=testAcc, color='gold', linestyle='--')
    plt.ylim(50, 100)
    plt.title('Model Accuracy, CNN')
    plt.ylabel('Accuracy [%]')
    plt.xlabel('Epochs')
    plt.legend(['Train','Val','Test'],loc='lower right')
    plt.show()

###Adaptive Boosting Implementation

def AdaptiveBoostingModel(trainingData, testingData):
    ### Def the columns in use/USE STANDARD COLS; NONSTANDARD NONPRESENT
    allColumns = ['A','C','D','E','F','G','H','I','K','L','M','N','P','Q','R','S','T','V','W','Y']

    ###Define training and testing values:
    xTrain = trainingData[['sequence']]
    yTrain = trainingData['label']

    xTest = testingData[['sequence']]
    yTest = testingData['label']

    ###USE OCCSeqs() and AACSeqs()
    ###Use these to construct the pipeline for both Ocurrence (getOCC) and for Amino Acid Composition (AAC)
    preprocessor = ColumnTransformer(transformers = [
        ('OCC', FunctionTransformer(OCCSeqs, kw_args={'desiredColumns': allColumns}),['sequence']),
        ('AAC', FunctionTransformer(AACSeqs, kw_args={'desiredColumns': allColumns}),['sequence'])])

    ###Define the AdaBoost Pipeline for output; note to self, clf just means classifier.  Bad acronym.
    adaBoostPipeline = Pipeline(steps=[('prep', preprocessor),('clf',AdaBoostClassifier())])

    ###Train the classifier model
    adaBoostPipeline.fit(xTrain, yTrain)

    ### Test it against the test set.  
    yPredicted = adaBoostPipeline.predict(xTest)

    ###Try k-fold cross validation too;
    crossValScores = cross_val_score(adaBoostPipeline, xTrain, yTrain, cv=5)

    crossValAvgScore = crossValScores.mean()



    ###Print various scores.
    accuracyScore = accuracy_score(yTest, yPredicted)
    reportCard = classification_report(yTest,yPredicted)
    print("\nAccuracy: ", accuracyScore)
    print("\nFULL REPORT:\n",reportCard)
    print("Cross Validation Score: ", crossValAvgScore)




###Create basic text interface:

###Choice of feature Eval + brief explanation of each ML method before a confirmation.  Once "confirmed, runs the method, then outputs the output of the function.
### Idea is for it to be a user menu; mostly for demo use during presentation of the project.  
userDecision = input("\nWhich would you like to run?  \n(0)CNN on BiGram Decomposition \n(1)AdaBoost on Occurrence and Amino Acid Composition?\n")
userDecision = int(userDecision)
if userDecision == 0:
    epochs = 20
    CNN(trainingData, testingData, epochs)
if userDecision == 1:
    AdaptiveBoostingModel(trainingData, testingData)
    
