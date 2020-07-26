import numpy as np
import os,csv,datetime,random
from datetime import date
import pandas as pd
from pandas import read_csv

from sklearn import metrics 
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

from sklearn.metrics import average_precision_score, recall_score, precision_score, f1_score

###SHOULD BE STRATIFIED KFOLD

###  UTILITIES
import utils
####
#SHUFFLE THEN SPLIT THEN EXTEND (METHOD 2) - CREATE THE NEGATIVE EXAMPLES BY PERMUTATION - DATASET ONLY MADE OF VALID ANALOGIES
def doCrossValid(size,dataset,n,batch,epochsNumber):
    names=['a', 'b', 'c', 'd','label'] #no header on dataset
    data = read_csv(dataset, names=names, header=None)
    array = data.values
    X = array[:,0:5]
    #randomness versus determinism
    seed = 7
    np.random.seed(seed)
    kfold = KFold(n_splits=n,shuffle=True,random_state=7) #WE SHUFFLE THE DATASET
    cvscores = []
    i=1
    for train_index, test_index in kfold.split(X): #WE SPLIT
        # TRAINING SET
        train='train_'+str(i)+'.csv'
        pd.DataFrame(X[train_index],columns=names).to_csv(train) #train csv created
        X_train, y_train  = utils.extend(gloveModel,size,train)  #WE EXTEND
        print(X_train.shape)
        
        # TESTING SET
        test='test_'+str(i)+'.csv'
        pd.DataFrame(X[test_index],columns=names).to_csv(test) #test csv created
        X_test, y_test = utils.extend(gloveModel,size,test) #8 pos permut + 16 neg permut
        print(X_test.shape)
        
        #CREATE MODEL SAME STRUCTURE
        input_shape=(size,4,1) 
        NNmodel = utils.createCNNModel(input_shape)
        
        #TRAIN MODEL
        NNmodel.fit(X_train, y_train, epochs=epochsNumber, batch_size=batch, verbose=0)
        
        #TEST MODEL
        scores = NNmodel.evaluate(X_test, y_test, verbose=0) 
        cvscores.append(scores[1] * 100)
        i+=1
    print(dataset,": ",n," cross folds AVERAGE ACCURACY for ",epochsNumber," epochs: ",np.mean(cvscores)," and STANDARD DEVIATION: ",np.std(cvscores))                                                                                                                           
    
#SPECIAL CASE OF SAT WITH TURNEY NEGATIVE EXAMPLES  
def doCrossValidSAT(size,dataset,n,batch,epochsNumber):
    names=['a', 'b', 'c', 'd','e'] #no header on dataset
    data = read_csv(dataset, names=names, header=None)
    array = data.values
    X = array[:,0:5]
    #randomness versus determinism
    seed = 7
    np.random.seed(seed)
    kfold = KFold(n_splits=n,shuffle=True,random_state=7) #WE SHUFFLE THE DATASET
    cvscores = []
    precisionscores=[]
    recallscores=[]
    f1scores=[]
    i=1
    for train_index, test_index in kfold.split(X): #WE SPLIT
        # TRAINING SET
        train='train_'+str(i)+'.csv'
        pd.DataFrame(X[train_index],columns=names).to_csv(train) #train csv created
        X_train, y_train  = utils.extendSAT(gloveModel,size,train)  #WE EXTEND
        print(X_train.shape)
        
        # TESTING SET
        test='test_'+str(i)+'.csv'
        pd.DataFrame(X[test_index],columns=names).to_csv(test) #test csv created
        X_test, y_test = utils.extendSAT(gloveModel,size,test) 
        print(X_test.shape)
        
        #CREATE MODEL SAME STRUCTURE
        input_shape=(size,4,1) 
        NNmodel = utils.createCNNModel(input_shape)
        #NNmodel = utils.createMLPModel(input_shape)
        
        #TRAIN MODEL
        NNmodel.fit(X_train, y_train, epochs=epochsNumber, batch_size=batch, verbose=0)
        
        #TEST MODEL AND GET METRICS
        scores = NNmodel.evaluate(X_test, y_test, verbose=0)
        y_score = NNmodel.decision_function(X_test)
        precision, recall, _ = precision_recall_curve(y_test,y_score)
        cvscores.append(scores[1] * 100)
        precisionscores.append(precision)
        recallscores.append(recall)
        f1scores.append(2*precision*recall/(precision+recall))
        i+=1
        
    print(dataset,": ",n," cross-folds AVERAGE ACCURACY for ",epochsNumber," epochs: ",np.mean(cvscores)," and STANDARD DEVIATION: ",np.std(cvscores),'precision:', np.mean(precisionscores),'recall:', np.mean(recallscores),'f1:', np.mean(f1scores)) 

#HERE IS AN EXAMPLE OF AN INITIAL DATASET COMING FROM GOOGLE (class 5)
#THE GOOGLE DATASET PROPERLY SPLITTED ARE IN THE DATA FOLDER
#DATASET = '../data/GOOGLE/questions-words-prime5.csv'

FOLDS=10
BATCH=1000
#WE DECIDE WHICH FOLDER TO DEAL WITH - ALL DATA SHOULD BE INSIDE THE DATA FOLDER AS CSV

folder='../data/BATS-CSV/4_Lexicographic_semantics'
list_file=os.listdir(folder)
for file in list_file:
    DATASET=os.path.join(folder,file)
    print('START WITH ', DATASET )
    for GLOVEDIMENSION in [50]:
        print('************** START DIMENSION: ', str(GLOVEDIMENSION),' **************')
        gloveFile = "../../models/glove.6B."+str(GLOVEDIMENSION)+"d.clean.txt"
        gloveModel, size = utils.loadGloveModel(gloveFile) 
        input_shape=(size,4,1) 
        for epochsNumber in [5]:
            print('Epoch: '+str(epochsNumber)+' - dimension: '+str(GLOVEDIMENSION))
            #doCrossValidSAT(GLOVEDIMENSION,DATASET,FOLDS,BATCH,epochsNumber)
            doCrossValid(size,DATASET,FOLDS,BATCH,epochsNumber)
            print(DATASET,' : ************** END DIMENSION: ', str(GLOVEDIMENSION),' **************')
    print('END WITH ', DATASET )