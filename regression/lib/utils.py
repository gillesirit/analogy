"""# Helper modules"""
import numpy as np
import pandas as pd
import csv
from collections import OrderedDict
from scipy.spatial import distance
import time

# KERAS
import keras
from keras.models import Model
from keras.layers import Dense, Input

# PANDA STYLE LOADING A FILE - GLOVEFILE IS A CSV FILE
def loadGloveWithPanda(gloveFile):
  model = pd.read_table(gloveFile, sep=" ", index_col=0, header=None, quoting=csv.QUOTE_NONE)
  #t = model.loc['a'].as_matrix()
  t = model.loc['a'].values
  size = len(t)
  return model, size

# START - UTILITIES FOR ANALOGIES LOADING 
def vec(w, model):
  return model.loc[w].values


def createVector(a, b):  # we get a vector of dim 2 * size
  vec = []
  vec.extend(a)
  vec.extend(b)
  return vec


def createDict(a, b, c, d, label):
  od = OrderedDict()
  od['a'] = a
  od['b'] = b
  od['c'] = c
  od['d'] = d
  od['label'] = label

  return od

# END - UTILITIES FOR ANALOGIES LOADING 
 
# CONVERT CSV ROW TO GLOVE
def convertCSV2GlovePos(gloveModel, row):

  a = vec(row['a'], gloveModel)
  b = vec(row['b'], gloveModel)
  c = vec(row['c'], gloveModel)
  d = vec(row['d'], gloveModel)
    
  # abc relationship | abc=createVector(a,b,c)
  ab = createVector(a, b)
  ac = createVector(a, c)  
  # acb relationship (same as abc)| acb=createVector(a,c,b)

  # cda relationship | cda=createVector(c,d,a)
  cd = createVector(c, d)
  ca = createVector(c, a)   
  # cad relationship (same as cda)| cad=createVector(c,a,d)
    
  # bad relationship | bad=createVector(b,a,d)
  ba = createVector(b, a)
  bd = createVector(b, d)  
  # bda relationship (same as bad) | bda=createVector(b,d,a)
    
  # dbc relationship | dbc=createVector(d,b,c)
  db = createVector(d, b)
  dc = createVector(d, c)    
  # dcb relationship (same as dbc) |  dcb=createVector(d,c,b)
    
  return ab, ac, d, cd, ca, b, ba, bd, c, db, dc, a
  
'''
works well for Google Dataset but not for Turney
'''
def createNNModel(inputDim, outputDim, denseLayerSize):

     # Define two inputs - one for ab and for ac.
  abInputLayer = Input(shape=(inputDim,), name='abInputLayer')
  acInputLayer = Input(shape=(inputDim,), name='acInputLayer')

  # Define the network that will calculate the relationship for ab layer
  abRelationship = Dense(denseLayerSize, kernel_initializer='normal', activation='relu')(abInputLayer)
  abRelationship = Model(inputs=abInputLayer, outputs=abRelationship)

  # Define the network that will calculate the relationship for ac layer
  acRelationship = Dense(denseLayerSize, kernel_initializer='normal', activation='relu')(acInputLayer)
  acRelationship = Model(inputs=acInputLayer, outputs=acRelationship)

  # Define the layer that takes the relationship from abRelationship and acRelationship
  # The ab and ac layers are concatenated
  abacRelationship = keras.layers.concatenate([abRelationship.output, acRelationship.output])

  hidden = Dense(denseLayerSize, kernel_initializer='normal', activation='relu')(abacRelationship)

  # Define the output layer that is connected to the abacRelationship
  #main_output = Dense(outputDim, activation='linear', name='main_output')(abacRelationship)
  main_output = Dense(outputDim, activation='linear', name='main_output')(hidden)

  # Define the nnModel with two inputs and one output
  nnModel = Model(inputs=[abRelationship.input, acRelationship.input], outputs=main_output)
  nnModel.compile(loss='mse', metrics=['mse'], optimizer='adam')

  return nnModel

def convertRowsToNNInputVector(gloveModel, analogies, indexes):
  Xab = []
  Xac = []
  y = []
 
  analogies_permuted = []
 
  for index in (indexes):
    #Need to add another subcsript row = analogies[index][0], instead of the previous row = analogies[index]
    #print("**** analogies[index]", analogies[index])
    row = analogies[index]
    #print("**** row", row)
    
    ab, ac, d, cd, ca, b, ba, bd, c, db, dc, a = convertCSV2GlovePos(gloveModel, row)     
    
    for im_ab in [ab, ac, cd, ca, ba, bd, db, dc]:
      Xab.append(np.array(im_ab))
                                    
    for im_ac in [ac, ab, ca, cd, bd, ba, dc, db]:
      Xac.append(np.array(im_ac))
 
    y.extend([d, d])
    y.extend([b, b])
    y.extend([c, c])
    y.extend([a, a])
 
    # Create 8 permutations of the analogy
    analogies_permuted.append(createDict(row['a'], row['b'], row['c'], row['d'], row['label']))    
    analogies_permuted.append(createDict(row['a'], row['c'], row['b'], row['d'], row['label']))
    analogies_permuted.append(createDict(row['c'], row['d'], row['a'], row['b'], row['label']))
    analogies_permuted.append(createDict(row['c'], row['a'], row['d'], row['b'], row['label']))   
   
    analogies_permuted.append(createDict(row['b'], row['a'], row['d'], row['c'], row['label']))
    analogies_permuted.append(createDict(row['b'], row['d'], row['a'], row['c'], row['label']))
    analogies_permuted.append(createDict(row['d'], row['b'], row['c'], row['a'], row['label']))
    analogies_permuted.append(createDict(row['d'], row['c'], row['b'], row['a'], row['label']))    
   
  return analogies_permuted, np.array(Xab), np.array(Xac), np.array(y)
  
'''
  Given a, b and c; create an input suitable as an input to the NN so we can find the answer for the missing word 'd'
'''  
def constructNNInput(a, b, c, vectorSize):
  equation_ab = np.array(createVector(a, b))
  equation_ab = np.array(equation_ab).reshape(1, 2 * vectorSize)
  
  equation_ac = np.array(createVector(a, c))
  equation_ac = np.array(equation_ac).reshape(1, 2 * vectorSize)

  list_of_2_arrays = [equation_ab, equation_ac]
  i_train = list_of_2_arrays
  
  return i_train

def constructNNInput2(a, b, c, vectorSize):
  equation_ab = np.array(createVector(a, b))
  equation_ab = np.array(equation_ab).reshape(1, 2 * vectorSize)
  
  equation_c = np.array(c)
  equation_c = np.array(equation_c).reshape(1, vectorSize)

  list_of_2_arrays = [equation_ab, equation_c]
  i_train = list_of_2_arrays
  
  return i_train

'''
  Find the word in gloveModel that is closest to targetVector
  time is about 0.25 seconds
'''
def findWordFromGloveCosineSimilarity(v, gloveModel):

  #start = time.time()
  u = np.expand_dims(gloveModel.values, 1)
  n = np.sum(u * v, axis=2)
  d = np.linalg.norm(u, axis=2) * np.linalg.norm(v, axis=1)
  results = n / d
  indices = np.argsort(np.squeeze(-results), axis=0)[:1]

  minIdx=indices[0]
  #print("cos time is", time.time() - start)

  return gloveModel.iloc[minIdx].name


'''
  Find the word in gloveModel that is closest to targetVector
  time is about 0.13 seconds
'''
def findWordFromGloveEuclideanDistance(targetVector, gloveModel):   
  start = time.time()
  diff = gloveModel.values - targetVector 
  delta = np.sum(diff * diff, axis=1)  
  minIdx = np.argmin(delta)
  
  print("Euclidean time is", time.time() - start)
  return gloveModel.iloc[minIdx].name

 
'''
  Use the nnModel to find the answer to a:b::c:x 
    nnModel = contain all words in the nnModel
    row   = the row that contains the four words of an anology - we will only use the first 3 columns (a:b:c)
    gloveModel = the model to be used
    vectorSize = the length of the vector used train the nnModel
'''
def getRegressionAnswer(nnModel, row, gloveModel, vectorSize):
  a = vec(row['a'], gloveModel)
  b = vec(row['b'], gloveModel)
  c = vec(row['c'], gloveModel)
        
  #get the value regressed
  dNN = nnModel.predict(constructNNInput(a, b, c, vectorSize))
  #dNN is unlikely to be a valid word in the model, so find one closest to it
  #dAnswer is the missing x  
  dAnswer = findWordFromGloveCosineSimilarity(dNN,gloveModel)  
  #dAnswer = findWordFromGloveEuclideanDistance(dNN,gloveModel)  
  return dAnswer
  

def getRegressionAnswer2(nnModel, row, gloveModel, vectorSize):
  a = vec(row['a'], gloveModel)
  b = vec(row['b'], gloveModel)
  c = vec(row['c'], gloveModel)
        
  #get the value regressed
  dNN = nnModel.predict(constructNNInput2(a, b, c, vectorSize))
  
  #dNN is unlikely to be a valid word in the model, so find one closest to it
  #dAnswer is the missing x  
  dAnswer = findWordFromGloveEuclideanDistance(dNN[0], gloveModel)  
  return dAnswer
