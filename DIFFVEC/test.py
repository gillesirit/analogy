import pandas as pd

#UTILS
#LIST OF DISTINCT CLASSES FROM DATA
def buildClass(data):
    l=[]
    for i in range(len(data)):
        l.append(data.values[i][0])
    list=[]
    for i in range(len(l)):
        if l[i] not in list:
            list.append(l[i])
    return list

#CARDINAL OF A CLASS
def cardinal_class(data,cl):
    l=[]
    for i in range(len(data)):
        if (data.values[i][0] == cl):
            l.append(data.values[i][0])
    return len(l)

#def class_of():
#BUILD ANALOGIES
#def build_analogy(data,cl):
    
    
    
#END UTILS
    
    
    
    
    
    

#MAIN
#LOAD CSV
data=pd.read_csv("./DiffVec.csv",usecols=[0])
 
#BUILD CLASSES
set_of_classes=buildClass(data)
print(len(set_of_classes))

#COMPUTE CARDINALITY OF EACH CLASS
for cl in set_of_classes:
    print(cardinal_class(data,cl))



