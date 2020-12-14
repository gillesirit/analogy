import numpy as np
import csv, datetime, random, os
from datetime import date
import pandas as pd
from pandas import read_csv

from sklearn import metrics 
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, classification_report, f1_score, precision_score, recall_score

###  UTILITIES TO BE IMPORTED
import utils
###
#1- GOOGLE DATASET SHUFFLE THEN SPLIT THEN EXTEND (METHOD 2) - CREATE THE NEGATIVE EXAMPLES BY PERMUTATION - DATASET MADE OF VALID ANALOGIES
def doCrossValidGoogle(size,dataset,n,epochsNumber):
    names=['a', 'b', 'c', 'd','label'] #no header on dataset
    data = read_csv(dataset, names=names, header=None)
    array = data.values
    X = array[:,0:5]
    batch=len(X)
    #randomness versus determinism
    seed = 7
    np.random.seed(seed)
    kfold = KFold(n_splits=n,shuffle=True,random_state=7) #WE SHUFFLE THE DATASET
    cvscores = []
    
    #Slim - add precision recalls
    precisionScores=[]
    recallScores=[]
    f1Scores=[]
    
    i=1
    for train_index, test_index in kfold.split(X): #WE SPLIT
        # TRAINING SET
        train='train_'+str(i)+'.csv'
        pd.DataFrame(X[train_index],columns=names).to_csv(train) #train csv created
        X_train, y_train  = utils.extendGoogle(gloveModel,size,train)  #WE EXTEND
        print(X_train.shape)
        
        # TESTING SET
        test='test_'+str(i)+'.csv'
        pd.DataFrame(X[test_index],columns=names).to_csv(test) #test csv created
        X_test, y_test = utils.extendGoogle(gloveModel,size,test) #8 pos permut + 16 neg permut
        print(X_test.shape)
        
        #CREATE MODEL SAME STRUCTURE
        input_shape=(size,4,1) 
        NNmodel = utils.createCNNModel(input_shape)
        
        #TRAIN MODEL
        NNmodel.fit(X_train, y_train, epochs=epochsNumber, batch_size=batch, verbose=0)
        
        #TEST MODEL
        scores = NNmodel.evaluate(X_test, y_test, verbose=0) 
    
        #Slim - add precision recalls             
        predicted=NNmodel.predict(X_test)
        tn, fp, fn, tp = confusion_matrix(y_test,np.around(np.array(predicted), 0)).ravel()  
        print("tp=", tp, "fp=", fp, "tn=",tn, "fn=",fn)
  
        print(classification_report(y_test,np.around(np.array(predicted), 0)))  
 
        precisionScores.append(precision_score(y_test,np.around(np.array(predicted), 0), average='binary'))
        recallScores.append(recall_score(y_test,np.around(np.array(predicted), 0), average='binary'))
        f1Scores.append(f1_score(y_test,np.around(np.array(predicted), 0), average='binary'))        
        cvscores.append(scores[1] * 100)
        
        i+=1
       # NNmodel.save("Method2-folds: "+str(n)+"--"+str(epochsNumber)+"epochDim"+str(size)+"fold_"+str(i)+".h5")
        
    print(dataset,": ",n," cross folds AVERAGE ACCURACY for ",epochsNumber," epochs: ",np.mean(cvscores)," and STANDARD DEVIATION: ",np.std(cvscores))
    print("Average precision %.2f" % np.mean(precisionScores), "($\pm$ %.2f)" % np.std(precisionScores)) 
    print("Average recall    %.2f" % np.mean(recallScores), "($\pm$ %.2f)" % np.std(recallScores))                                       
    print("Average f1        %.2f" % np.mean(f1Scores), "($\pm$ %.2f)" % np.std(f1Scores))
                                                                                                                           
    
#2 - SAT DATASET WITH TURNEY NEGATIVE EXAMPLES  
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
    
    #Slim - add precision recalls
    precisionScores=[]
    recallScores=[]
    f1Scores=[]
    
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
        
        #TEST MODEL
        scores = NNmodel.evaluate(X_test, y_test, verbose=0) 

        #Slim - add precision recalls             
        predicted=NNmodel.predict(X_test)
        tn, fp, fn, tp = confusion_matrix(y_test,np.around(np.array(predicted), 0)).ravel()  
        print("tp=", tp, "fp=", fp, "tn=",tn, "fn=",fn)
  
        print(classification_report(y_test,np.around(np.array(predicted), 0)))  
 
        precisionScores.append(precision_score(y_test,np.around(np.array(predicted), 0), average='binary'))
        recallScores.append(recall_score(y_test,np.around(np.array(predicted), 0), average='binary'))
        f1Scores.append(f1_score(y_test,np.around(np.array(predicted), 0), average='binary'))       
        
        cvscores.append(scores[1] * 100)
        i+=1
        #NNmodel.save("SAT-Method2-folds: "+str(n)+"--"+str(epochsNumber)+"epochDim"+str(size)+"fold_"+str(i)+".h5")
    print(dataset,": ",n," cross folds AVERAGE ACCURACY for ",epochsNumber," epochs: ",np.mean(cvscores)," and STANDARD DEVIATION: ",np.std(cvscores)) 
    print("Average precision %.2f" % np.mean(precisionScores), "($\pm$ %.2f)" % np.std(precisionScores)) 
    print("Average recall    %.2f" % np.mean(recallScores), "($\pm$ %.2f)" % np.std(recallScores))                            
    print("Average f1        %.2f" % np.mean(f1Scores), "($\pm$ %.2f)" % np.std(f1Scores))

    
#3- DIFFVEC DATASET SHUFFLE THEN SPLIT THEN EXTEND (METHOD 2) - INITIAL DATASET ONLY MADE OF VALID ANALOGIES
def doCrossValidDiffVec(size,dataset,n,epochsNumber):
    names=['a', 'b', 'c', 'd','label'] #no header on dataset
    data = read_csv(dataset, names=names, header=None)
    array = data.values
    X = array[:,0:5]
    batch=len(X)
    #randomness versus determinism
    seed = 7
    np.random.seed(seed)
    kfold = KFold(n_splits=n,shuffle=True,random_state=7) #WE SHUFFLE THE DATASET
    cvscores = []
    
    #Slim - add precision recalls
    precisionScores=[]
    recallScores=[]
    f1Scores=[]
    
    i=1
    for train_index, test_index in kfold.split(X): #WE SPLIT
        # TRAINING SET
        train='train_'+str(i)+'.csv'
        pd.DataFrame(X[train_index],columns=names).to_csv(train) #train csv created
        X_train, y_train  = utils.extendDiffVec(gloveModel,size,train)  #WE EXTEND
        print(X_train.shape)
        
        # TESTING SET
        test='test_'+str(i)+'.csv'
        pd.DataFrame(X[test_index],columns=names).to_csv(test) #test csv created
        X_test, y_test = utils.extendDiffVec(gloveModel,size,test) #8 pos permut + 16 neg permut
        print(X_test.shape)
        
        #CREATE MODEL SAME STRUCTURE
        input_shape=(size,4,1) 
        NNmodel = utils.createCNNModel(input_shape)
        
        #TRAIN MODEL
        NNmodel.fit(X_train, y_train, epochs=epochsNumber, batch_size=batch, verbose=0)
        
        #TEST MODEL
        scores = NNmodel.evaluate(X_test, y_test, verbose=0) 
    
        #Slim - add precision recalls             
        predicted=NNmodel.predict(X_test)
        tn, fp, fn, tp = confusion_matrix(y_test,np.around(np.array(predicted), 0)).ravel()  
        print("tp=", tp, "fp=", fp, "tn=",tn, "fn=",fn)
  
        print(classification_report(y_test,np.around(np.array(predicted), 0)))  
 
        precisionScores.append(precision_score(y_test,np.around(np.array(predicted), 0), average='binary'))
        recallScores.append(recall_score(y_test,np.around(np.array(predicted), 0), average='binary'))
        f1Scores.append(f1_score(y_test,np.around(np.array(predicted), 0), average='binary'))        
        cvscores.append(scores[1] * 100)
        i+=1
        
    print(dataset,": ",n," cross folds AVERAGE ACCURACY for ",epochsNumber," epochs: ",np.mean(cvscores)," and STANDARD DEVIATION: ",np.std(cvscores))
    print("Average precision %.2f" % np.mean(precisionScores), "($\pm$ %.2f)" % np.std(precisionScores)) 
    print("Average recall    %.2f" % np.mean(recallScores), "($\pm$ %.2f)" % np.std(recallScores))                                       
    print("Average f1        %.2f" % np.mean(f1Scores), "($\pm$ %.2f)" % np.std(f1Scores))
                          
#GLOBAL VARIABLES - NO HEADER a b c d label in the dataset
#DATASET = '../data/GOOGLE/questions-words-prime5.csv'
FOLDER_DIFFVEC='../data/DIFFVEC/diffvecAnalogiesByClass/'
FOLDER_GOOGLE='../data/GOOGLE/'
DATASETDIFFVEC=os.listdir(FOLDER_DIFFVEC) 
DATASETGOOGLE = os.listdir(FOLDER_GOOGLE)
#FOR SAT DATASET: ASK PETER TURNEY

FOLDS=10
for FILE in DATASETGOOGLE:
    DATASET=FOLDER_GOOGLE+FILE
    if (os.stat(DATASET).st_size)< 30000: #TO AVOID TOO BIG FILES N(N_1)/2
        print('START WITH ', FILE )
        for GLOVEDIMENSION in [100]:
            print('************** START DIMENSION: ', str(GLOVEDIMENSION),' **************')
            gloveFile = "../GloveModels/glove.6B."+str(GLOVEDIMENSION)+"d.clean.txt"
            gloveModel, size = utils.loadGloveModel(gloveFile) 
            input_shape=(size,4,1) 
            for epochsNumber in [10]:
                print("GOOGLE METHOD: ",DATASET, ' Epoch: '+str(epochsNumber)+' - Glove dimension: '+str(GLOVEDIMENSION))
                doCrossValidGoogle(GLOVEDIMENSION,DATASET,FOLDS,epochsNumber)
                print("DIFFVEC METHOD: ",DATASET, ' Epoch: '+str(epochsNumber)+' - Glove dimension: '+str(GLOVEDIMENSION))
                doCrossValidDiffVec(GLOVEDIMENSION,DATASET,FOLDS,epochsNumber)
                print(DATASET,' : ************** END DIMENSION: ', str(GLOVEDIMENSION),' **************')
            print('END WITH ', DATASET )
