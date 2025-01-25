import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.feature_selection import SequentialFeatureSelector
df =pd.read_csv("Bourse.dat", sep="|", names=['data'])
df = df['data'].str.split(n=20, expand=True)
df=df.astype(float)
df -= df.min()
df /= df.max()

data=df.to_numpy()
np.random.shuffle(data)
X=np.squeeze(np.delete(data,[20],axis=1))
y=np.array(data[:,20], dtype=np.int8)
Xtrain=X[:115]
Xtest=X[115:]
Ytrain=y[:115]
Ytest=y[115:]

nnModel=MLPClassifier(hidden_layer_sizes=(7),alpha=1e-4,solver='adam',max_iter=100000,n_iter_no_change=1000,verbose=False,learning_rate_init=0.1)#
nnModel.fit(Xtrain,Ytrain)
print("training score",nnModel.score(Xtrain,Ytrain))
print("validation score",nnModel.score(Xtest,Ytest))
nnModel.predict(Xtest)

"""# **feature selection**"""

sfs_selector = SequentialFeatureSelector(estimator=MLPClassifier(hidden_layer_sizes=(7),alpha=1e-4,solver='adam',max_iter=100000,n_iter_no_change=1000,verbose=False,learning_rate_init=0.1), n_features_to_select =3, direction ='backward')
sfs_selector.fit(X, y)
np.array([i for i in range(20)])[sfs_selector.get_support()]

"""# **# model validation with selected features**"""

featureList=[10, 11, 16]
scores=[]
for j in range(100):
  data=df.to_numpy()
  np.random.shuffle(data)
  X=np.delete(data,[f for f in range(21) if f not in featureList],axis=1)
  y=np.array(data[:,20], dtype=np.int8)
  Xtrain=X[:123]
  Xtest=X[123:]
  Ytrain=y[:123]
  Ytest=y[123:]
  nnModel=MLPClassifier(hidden_layer_sizes=(7),alpha=1e-4,solver='adam',max_iter=100000,n_iter_no_change=1000,verbose=False,learning_rate_init=0.1)#
  nnModel.fit(Xtrain,Ytrain)
  scores.append(nnModel.score(Xtest,Ytest))
print(np.sum(np.array(scores)))