from time import gmtime, strftime

import csv
import statistics
import numpy as np
import lib.createCrossValidation as crossValid
import lib.utils as utils

import sys

# Use this setting for running on a local unix machine
SEED = 7
FOLD = 10

# Use this setting in Eclipse:
# python3.7 multiVariateRegression.py 50 50 ../data/questions-words-prime1-comcap.csv KFold
#
# or for Stratified 
# python3.7 multiVariateRegression.py 50 50 ../data/questions-words-prime.12771NoHeader.csv Stratified
# Set running environment as specified by users
if len(sys.argv) < 5:
    print ("Usage " + sys.argv[0] + " <dimension> <epochs> <analogy_file> <KFold|Stratifed>")
    exit(-1)
vectorSize = int(sys.argv[1])
epochs = int(sys.argv[2])
analogyFile = sys.argv[3]
STRATIFIED = KFOLD = False

print("Glove Dimension=" + sys.argv[1] + ", Epochs=" + sys.argv[2] + ", analogy file=" + sys.argv[3] + " (" + sys.argv[4] + ") Cosine Distance")

if sys.argv[4] == "KFold":
  KFOLD = True
elif sys.argv[4] == "Stratified":
  STRATIFIED = True
else:
  print("Please specifiy KFold or Stratified")
  exit(-1)

# Use these two settings for running experiments
modelFile = "../../models/glove.6B." + str(vectorSize) + "d.clean.txt"

# load the model using panda
gloveModel, size = utils.loadGloveWithPanda(modelFile)

# save current time to ensure all folds has the same time (so it is easier to identify)
exp_time = strftime("%Y-%m-%d %H:%M:%S", gmtime())

np.random.seed(SEED)  # ensure experiment can be replicated and set random_state=None to use np's random state

"""# Load Analogy Data"""
# METHOD 2, SPLIT THEN EXPAND

analogies = []
csvFile = analogyFile

# read the analogy file - without header
with open(csvFile, newline='') as csvfile:
  reader = csv.DictReader(csvfile, fieldnames=['a', 'b', 'c', 'd', 'label'])
    
  for row in reader:    
    analogies.append(row)
    
# split into 10 folds, depending on whether it is stratified or kfold 
if KFOLD == True:  
  trainAllFolds, testAllFolds = crossValid.splitKFold(analogies, FOLD) 
elif STRATIFIED == True:
  # now that we have the index, get the analogies for the given index
  #trainAllFolds, testAllFolds = crossValid.splitStratified(analogy_label, FOLD)
  trainAllFolds, testAllFolds = crossValid.splitStratified(analogies, FOLD)
else:
  print("Can only split using KFold or Stratified")

# dimension of input and the ense layer
inputDim = 2 * vectorSize
outputDim = vectorSize
denseLayerSize = vectorSize
np.random.seed(SEED) #To ensure the experiment can be replicated

# Start 10 CROSS VALIDATION
foldCount = 1
accuracies=[]
MSE_train=[]
MSE_test=[]

for trainIndex, testIndex in zip(trainAllFolds, testAllFolds):
  nnModel = utils.createNNModel(inputDim, outputDim, denseLayerSize)
  print(nnModel.summary())
  print("fold", foldCount)
  
  analogiesTrain = analogiesTest = []    
  analogyTrain, XabTrain, XacTrain, yTrain = utils.convertRowsToNNInputVector(gloveModel, analogies, trainIndex)
  print("Original data length for training", len(trainIndex))
  print("Data size after permutation      ", len(analogyTrain))
  print("Shape of the training set(ab): ", XabTrain.shape, " train(ac): ", XacTrain.shape)
  
  nnModel.fit([XabTrain, XacTrain], yTrain, epochs=epochs, verbose=0)

  scores = nnModel.evaluate([XabTrain, XacTrain], yTrain)  # batch_size=5
  print(scores)
  print('MSE_on_train:', scores[0])
  MSE_train.append(scores[0])
  
  analogyTest, XabTest, XacTest, yTest = utils.convertRowsToNNInputVector(gloveModel, analogies, testIndex)
  scores = nnModel.evaluate([XabTest, XacTest], yTest)  # batch_size=5
  print('MSE_on_test:', scores[0], "\n")
  MSE_test.append(scores[0])  

  print("Original data length for testing", len(testIndex))
  print("Data size after permutation      ", len(analogyTest))
  print("Shape of the test set(ab): ", XabTest.shape, " test(ac): ", XacTest.shape)
  
  # SAVING MODEL
  # model_filename = 'mlr-' + exp_time + '-epochs' + str(epochs) + '-vectorSize' + str(vectorSize) + '_fold' + str(fold_count) + 'questions-words-prime.h5'
  # nnModel.save(model_filename)
  # print ("================================================\n")
  # print (model_filename)

  # EVALUATE THE NETWORK FOR EACH TEST SET
  # Evaluate the network using each dataset 
  total=0
  correct=0  

  #for each analogy in the permuted data set
  for row in analogyTest:
    #1. dAnswer is the 'd' predicted by the network
    dAnswer = utils.getRegressionAnswer(nnModel, row, gloveModel, vectorSize) 
  
    correctness = "Incorrect"   
    #if it is the right answer, then increment the count
    if (dAnswer == row['d']):
      correctness = "1"
      correct+=1     
      
    print ("Analogy=" + row['a'] + ":" + row['b'] + "::" + row['c'] + ":" + row['d'] + " =class " + row['label'] + ", Regression=", correctness, "[" + dAnswer, "]", flush=True)    
    total+=1

  accuracy=(correct/total) * 100 
  accuracies.append(accuracy)
  print("Accuracy for fold", foldCount, "is", accuracy, "\% (" + str (correct) + "/" + str(total) +")")     
  foldCount+=1
 
#end each fold

print("Average MSE_on_train", statistics.mean(MSE_train), "\% ($\pm$", statistics.stdev(MSE_train), ")")
print("Average MSE_on_test", statistics.mean(MSE_test), "\% ($\pm$", statistics.stdev(MSE_test), ")")

if len(accuracies)>1: 
  print("Average Accuracy, ", statistics.mean(accuracies), "\% ($\pm$", statistics.stdev(accuracies), ")")
else:
  print("Average Accuracy, ", statistics.mean(accuracies), "\%")  
