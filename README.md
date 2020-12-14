# Analogy
Contains Machine Learning approach for word analogies.


The GloVe embedding that we used can be found on https://nlp.stanford.edu/projects/glove/ as http://nlp.stanford.edu/data/glove.6B.zip.

We provide BATS, Google and DiffVec datasets a CSV.  These are the data we used to train and test.
Regarding SAT dataset, please kindly ask Peter Turney.

# Analogy classification 
In the python file 10CrossValidTrain.py, just indicate line 112 the data folder containing the CSV files you want to classify,
then run : python3.7 10CrossValidTest.py

# Analogy completion as a regression task
The program for running the neural network regression is multiVariateRegression.py and it expects four parameters:
- first param is number of Glove Vector Dimension (50)
- second param is number of epochs (100)
- third parameter is the filename of the analogy file (../../data/GOOGLE/questions-words-prime1-comcap.csv)
- fourth parameter is KFold or Stratified (if the analogy has more than one class, then use Stratified)
Example:
        python3.7 multiVariateRegression.py 50 100 ../../data/GOOGLE/questions-words-prime1-comcap.csv KFold

The analogy file is assumed to be in this location "../../models/" if your model is in a different location, then modify line 40 in multiVariateRegression
