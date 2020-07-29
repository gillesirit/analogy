import random
import numpy as np
#26th Aoril - to create 10 fold cross validation
#The Stratified split from Keras is unreliable, as it used only half of the data for test

# data which has been permuted is removed, so the total no of data is now only 12771 instead of 19544
#NO_OF_CLASSESS=14 #no of classes in the dataset

from sklearn.model_selection import KFold

DEBUG=False
SEED = 7

def findClassSize(data):
  classSize=[]
  classNo=int(data[0]['label'])
  classCount=0
    
  #go through each row to find out how many entries in each class and how many classes there are  
  for line in data:
    #if it is a new class      
    if classNo != int(line['label']):
      classNo+=1
  
      if classNo>1:
        classSize.append(classCount)
        classCount=0
    classCount+=1

  classSize.append(classCount)
  return classSize
  
def splitKFold(data,fold):
  kfold = KFold(n_splits=fold, shuffle=True, random_state=SEED)
  testAllFolds=[0 for x in range(fold)]
  trainAllFolds=[0 for x in range(fold)]  
  
  #create the second dimension - there's probably a better way, but I don't know how  
  for index in range (0, fold):
    testAllFolds[index]=[]
    trainAllFolds[index]=[]
    
  index=0
  for trainIndex, testIndex in kfold.split(data):
      trainAllFolds[index]=trainIndex
      testAllFolds[index]=testIndex
      index+=1
  
  return trainAllFolds, testAllFolds

#Use KFold to split data equally in each class
#Can work with 1 or multiple classes
#The data for training
def splitStratified(data,fold):

  classSize=findClassSize(data)  
  #print("classSize", classSize)
  
  #to store the index of the classes, for eg [0]506, [1] will be 506+4524=5030
  #so, it is easy to generate the random numbers
  #classIndex=[0 for x in range(NO_OF_CLASSESS)]
  noOfClasses=len(classSize)
  classIndex=[0 for x in range(noOfClasses+1)]
  testAllFolds=[0 for x in range(fold)]
  trainAllFolds=[0 for x in range(fold)]

  #generate the index location
  classIndex[0]=0
  for classNo in range (0,noOfClasses):  
    classIndex[classNo+1]+=classIndex[classNo]+ classSize[classNo]
    
  #print(classIndex)
  
  #create the second dimension - there's probably a better way, but I don't know how  
  for index in range (0,fold):
    testAllFolds[index]=[]
    trainAllFolds[index]=[]
  
  #make sure the random data is repeatable
  random.seed(SEED)  
  print("noOfClasses", noOfClasses)

  kfold = KFold(n_splits=fold)
  #generate the fold for each class
  for classNo in range(0,noOfClasses):
    foldNo=0
    for trainIndex, testIndex in kfold.split(data[classIndex[classNo]:classIndex[classNo+1]]):
      trainAllFolds[foldNo].extend(trainIndex + classIndex[classNo])
      testAllFolds[foldNo].extend(testIndex + classIndex[classNo])
      foldNo+=1 

  #for foldNo in range (0,fold):
    #print("trainIndex", trainAllFolds[foldNo])
    #print("testIndex", testAllFolds[foldNo])    
  
  return trainAllFolds,testAllFolds    
# 
# def splitStratified(classes,fold):
#   
#   totalData, classSize=findClassSize(classes)
#   #to store the index of the classes, for eg [0]506, [1] will be 506+4524=5030
#   #so, it is easy to generate the random numbers
#   #classIndex=[0 for x in range(NO_OF_CLASSESS)]
#   noOfClasses=len(classSize)
#   classIndex=[0 for x in range(noOfClasses)]
#   testAllFolds=[0 for x in range(fold)]
#   trainAllFolds=[0 for x in range(fold)]
#   available=[True for x in range(totalData)]
#   test_fraction=1/fold
#   
#   #generate the index location
#   classIndex[0]=classSize[0]
#   for classNo in range (1,noOfClasses):  
#     classIndex[classNo]+=classIndex[classNo-1]+ classSize[classNo]
#   
#   #create the second dimension - there's probably a better way, but I don't know how  
#   for index in range (0,fold):
#     testAllFolds[index]=[]
#     trainAllFolds[index]=[]
#   
#   #make sure the random date is repeatable
#   random.seed(SEED)  
#   print("noOfClasses", noOfClasses)
#   #for each class, select 10% of data
#   for fold_no in range (0,fold-1):  
#     for classNo in range(0,noOfClasses):    
#       #generates 10% of data for each class
#       dataCount=int(test_fraction*classSize[classNo])
#       dataGeneratedPerClass=0
#       
#       #GENERATE THE INDEX FOR EACH fold
#       #find the starting index for the class
#       startIdx=0    
#       if classNo!=0:
#         startIdx=classIndex[classNo-1]
#          
#       while(dataGeneratedPerClass<dataCount and any(available[startIdx:classIndex[classNo]])):
#         selected = random.randint(startIdx,classIndex[classNo]-1) #range is inclusive, so make sure we minus 1 from the end
#             
#         if(available[selected]): 
#           testAllFolds[fold_no].append(selected)           
#           available[selected]=False                                    
#           dataGeneratedPerClass+=1          
#           
#   #for the last fold, take what's available
#   for itemIdx in range(0,len(available)):
#     if(available[itemIdx]):
#       for classIdx in range (0,noOfClasses): #find which class
#         if itemIdx < classIndex[classIdx]:
#           classNo=classIdx+1
#           break
#       testAllFolds[fold-1].append(itemIdx)
#       available[itemIdx]=False
#       
#   #NOW GENERATE THE TRAINING SET FOR EACH FOLD, WHICH IS THE OTHER NINE FOLDS      
#   for foldIdx in range(0,fold):
#     #the training set is then everything else
#     for testFoldIdx in range(0, fold):
#       if foldIdx == testFoldIdx: # exclude the testAllFolds fold
#         continue        
#       
#       for itemIdx in range(0,len(testAllFolds[testFoldIdx])):  
#         trainAllFolds[foldIdx].append(testAllFolds[testFoldIdx][itemIdx])    
#     
#   if (DEBUG):
#     for foldIdx in range(0,fold):
#       print("testAllFolds-fold", foldIdx)
#       for itemIdx in range(0,len(testAllFolds[foldIdx])):
#         print(testAllFolds[foldIdx][itemIdx])
#       
#       print("trainAllFolds-fold", foldIdx)
#       for itemIdx in range(0,len(trainAllFolds[foldIdx])):
#         print(trainAllFolds[foldIdx][itemIdx])      
#   
#   return trainAllFolds,testAllFolds    
