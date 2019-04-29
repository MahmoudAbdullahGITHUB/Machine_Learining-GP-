import pandas as pd
from csv import reader
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix
import warnings
import numpy as np
from sklearn.metrics import accuracy_score



# Assign colum names to the dataset
#names = ['name','Ram','storage','screen size','OS','Front Camera','Rear Camera','price','rating']
names = ['name','Ram','storage','screen size','OS','Front Camera','Rear Camera','rating']

filename = 'seeds_dataset4.csv'

# Read dataset to pandas dataframe
irisdata = pd.read_csv(filename , names=names)
print("irisdata")
print( len(irisdata))
print(irisdata.head())

# Assign data from first four columns to X variable
X = irisdata.iloc[:,0:7]
print("X")
print( len(X))
print(X.head())

# Assign data from first fifth columns to y variable
#y = irisdata.select_dtypes(include=['rating'])
y = irisdata.iloc[:, 7:8]

print("y")
print( len(y))
print(y.head())

print("unique")
print( len(y.rating.unique()))
print(y.rating.unique())

print("sort")
unique = sorted(y.rating.unique())
print(unique)

# lables for each unique of rating
le = preprocessing.LabelEncoder()
y = y.apply(le.fit_transform)
print("y Lables")
print(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30)

print("X_train")
print(X_train)
print("X_test")
print(X_test)
print("y_train")
print(y_train)
print("y_test")
print(y_test)



#scale only for the testing data
scaler = StandardScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
print("X_trainS")
print(X_train)

X_test = scaler.transform(X_test)
print("X_testS")
print(X_test)


"""
MLPClassifier(activation='relu', alpha=1e-05, batch_size='auto',
              beta_1=0.9, beta_2=0.999, early_stopping=False,
              epsilon=1e-08, hidden_layer_sizes=(5, 2),
              learning_rate='constant', learning_rate_init=0.001,
              max_iter=200, momentum=0.9, n_iter_no_change=10,
              nesterovs_momentum=True, power_t=0.5, random_state=1,
              shuffle=True, solver='lbfgs', tol=0.0001,
              validation_fraction=0.1, verbose=False, warm_start=False)
"""



mlp = MLPClassifier(hidden_layer_sizes=(100, 100, 100), max_iter=200000 , learning_rate='constant' ,learning_rate_init=.01 )
mlp.fit(X_train, y_train.values.ravel())

print("mlp.fit")
print(mlp.fit(X_train, y_train.values.ravel()))

print("y_test")
print(y_test)
print("set(y_test)")
print(set(y_test))      # why this output ???

predictions = mlp.predict(X_test)
print("predictions")
print(predictions)

print("set(y_test )- set(predictions)")
print(set(y_test )- set(predictions))      # why this output ???


print("confusion_matrix")                       #### ????
print(confusion_matrix(y_test,predictions))
warnings.filterwarnings('ignore')
print("classification_report")
print(classification_report(y_test,predictions))

print("the Accurecy")
print(accuracy_score(y_test,predictions))


myTest = [2.0,1.0,8.0,5.0,1.0,5.0,8.0]
myTest = np.array(myTest).reshape(1,-1)
myPredection = mlp.predict(myTest)
indexPredection = myPredection[0]
print("index")
print(indexPredection)
print("myPrediction")
print(unique[indexPredection])


