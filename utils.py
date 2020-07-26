###  UTILITIES FOR ANALOGIES LOADING
import numpy as np
import csv
import random
import datetime
from datetime import date
from sklearn import metrics 
from sklearn.model_selection import train_test_split

#TO SPLIT FILES
import pandas as pd

#KERAS
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Conv2D
from keras.optimizers import Adam
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from keras.models import load_model

#CREATE A SPLIT TRAIN/TEST WITH 1st line a b c d label
def split(dataSet,testSize): #create 2 files train.csv and test.csv WE HAVE TO SHUFFLE
    data = pd.read_csv(dataSet)
    y = data.label
    X = data  #data.drop('label', axis=1)
    Xtrain, Xtest, ytrain, ytest = train_test_split(X, y,test_size=testSize,stratify=y)
    Xtrain.to_csv('train.csv',index=False)
    Xtest.to_csv('test.csv',index=False)

#LOADING GLOVEFILE
def loadGloveModel(gloveFile):
    #print ("Loading Glove Pretrained Model '"+gloveFile+'\'')
    with open(gloveFile, encoding="utf8" ) as f:
       content = f.readlines()
    model = {}
    for line in content:
        splitLine = line.split()
        word = splitLine[0]
        embedding = np.array([float(val) for val in splitLine[1:]])
        model[word] = embedding
        size=len(embedding)
    #print ("Done with loading ",len(model)," words loaded with",size," components each")
    return model, size

#CREATE THE STACK FOR 1 ANALOGY - NO PERMUTATION - a b c d are words
def convertCSVRow2Glove(gloveModel,row):
    a=gloveModel[row['a']]
    b=gloveModel[row['b']]
    c=gloveModel[row['c']]
    d=gloveModel[row['d']]
    return np.stack([a,b,c,d]).T

#CREATE A CLASS OF 8 PERMUT  - a b c d are vectors
def createClassOf8ImgWithClass(gloveModel, a, b, c, d): 
    im1=np.stack([a,b,c,d]).T
    im2=np.stack([a,c,b,d]).T
    im3=np.stack([c,d,a,b]).T
    im4=np.stack([c,a,d,b]).T
    im5=np.stack([b,a,d,c]).T
    im6=np.stack([b,d,a,c]).T
    im7=np.stack([d,b,c,a]).T
    im8=np.stack([d,c,b,a]).T
    return im1,im2,im3,im4,im5,im6,im7,im8

def createClassOf8ImgWithClassFromRowabcd(gloveModel, row):
    a=gloveModel[row['a']]
    b=gloveModel[row['b']]
    c=gloveModel[row['c']]
    d=gloveModel[row['d']]
    return createClassOf8ImgWithClass(gloveModel,a,b,c,d)

def createClassOf8ImgWithClassFromRowabcdSAT(gloveModel, row):
    a=gloveModel[row['a']]
    b=gloveModel[row['b']]
    c=gloveModel[row['c']]
    d=gloveModel[row['d']]
    label = row['e'] #we get the class
    im1,im2,im3,im4,im5,im6,im7,im8 = createClassOf8ImgWithClass(gloveModel,a,b,c,d)
    return im1,im2,im3,im4,im5,im6,im7,im8, int(label)


def createClassOf8ImgWithClassFromRowbacd(gloveModel, row):
    a=gloveModel[row['a']]
    b=gloveModel[row['b']]
    c=gloveModel[row['c']]
    d=gloveModel[row['d']]
    return createClassOf8ImgWithClass(gloveModel,b,a,c,d)

def createClassOf8ImgWithClassFromRowcbad(gloveModel, row):
    a=gloveModel[row['a']]
    b=gloveModel[row['b']]
    c=gloveModel[row['c']]
    d=gloveModel[row['d']]
    return createClassOf8ImgWithClass(gloveModel,c,b,a,d)

def extend(gloveModel, size, dataset):  #dataset is a csv file of valid analogies
    with open(dataset, newline='') as csvfile:
        X, y = [], []
        reader = csv.DictReader(csvfile) 
        count = 0
        for row in reader:
            abcd=createClassOf8ImgWithClassFromRowabcd(gloveModel,row)
            bacd=createClassOf8ImgWithClassFromRowbacd(gloveModel,row)
            cbad=createClassOf8ImgWithClassFromRowcbad(gloveModel,row)
            for im in abcd:
                X.append(np.array(im))
                y.append(1)
                count+=1
        
            for im in bacd: 
                X.append(np.array(im))
                y.append(0)
                count+=1
            
            for im in cbad: 
                X.append(np.array(im))
                y.append(0)
                count+=1
    X = np.array(X) 
    X = X.reshape(X.shape[0], size, 4, 1)
    y=np.array(y)
    
    return X, y  

def extendSAT(gloveModel, size, SAT):  #dataset is the csv file of valid and non valid analogies
    with open(SAT, newline='') as csvfile:
        X, y = [], []
        reader = csv.DictReader(csvfile) 
        countInDataset = 0
        countNumberOfExamples = 0
        for row in reader:
            im1,im2,im3,im4,im5,im6,im7,im8, label=createClassOf8ImgWithClassFromRowabcdSAT(gloveModel,row)
            for im in [im1,im2,im3,im4,im5,im6,im7,im8]:  #we get only 8 permuts
                X.append(np.array(im))
                y.append(label)
                countNumberOfExamples+=1
            countInDataset+=1
    X = np.array(X) 
    X = X.reshape(X.shape[0], size, 4, 1)
    y=np.array(y)
    print('initial SAT size: ',str(countInDataset), 'final SAT size: ',str(countNumberOfExamples))
    return X, y  


'''
def prepareDataAsIs(gloveModel,size,dataset): #do not add any permut - dataset with 1st line a b c d label
    with open(dataset, newline='') as csvfile:
        X, y = [], []
        reader = csv.DictReader(csvfile)
        count = 0
        for row in reader:
            im=convertCSVRow2Glove(gloveModel,row)
            count+=1
            X.append(np.array(im))
            y.append(int(row['label']))
        print('Size of dataset '+dataset+': ',count)
        X = np.array(X) 
        X = X.reshape(X.shape[0], size, 4, 1)
        y=np.array(y)
        return X, y   
'''

## DESIGNING NEURAL NETWORK MODELS
def createCNNModel(shape):
    model = Sequential()
    model.add(Conv2D(128, (1, 2), strides=(1,2), input_shape=shape))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (2, 2),strides=(2,2)))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation('relu'))
    #model.add(Dense(12,activation='relu'))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    #model.summary()
    model.compile(loss='binary_crossentropy', metrics=['accuracy'], optimizer='adam')
    return model

def createMLPModel(shape):
    model = Sequential()
    model.add(Dense(200,input_shape=shape))
    model.add(Activation('relu'))
    model.add(Dense(100))
    model.add(Activation('relu'))
    model.add(Dense(1, activation='sigmoid'))
    #model.summary()
    model.compile(loss='binary_crossentropy', metrics=['accuracy'], optimizer='rmsprop')
    return model

def saveModel(model,infoString):
    today = date.today()
    name = str(today)+'-'+infoString+'.h5'
    model.save(name)
    return name
### END UTILITIES