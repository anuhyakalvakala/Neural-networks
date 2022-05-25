import tensorflow as tf
import numpy as np
from sklearn import datasets
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import KFold
from tensorflow.keras import models,layers

print("***** one hot encoder *****")

data = datasets.fetch_openml(data_id=1489)
enc = OneHotEncoder(sparse=False)
tmp = [[x] for x in data.target] 
ohe_target = enc.fit_transform(tmp)

print(" running with 10 folds")

kfolds = KFold(n_splits=10,shuffle=True,random_state=0)
kfolds

print("with no hidden layers")

test_fold_accuracy = []
for train, test in kfolds.split(data.data,ohe_target) :
    nn = models.Sequential()
    nn.add(layers.Dense(10,activation='relu',input_dim=5))
    nn.add(layers.Dense(2,activation="softmax"))
    nn.compile(optimizer="adam",loss ="categorical_crossentropy",metrics=["accuracy"])
    n_epochs = 100
    nn.fit(data.data.iloc[train],ohe_target[train],epochs=n_epochs)
    s = nn.evaluate(data.data.iloc[test],ohe_target[test])
    test_fold_accuracy.append(s[1])
    print("Fold",len(test_fold_accuracy),"Accuracy =",s[1])
   
print("\nTesting accuracy for all folds:",test_fold_accuracy)
no_hidden=np.mean(test_fold_accuracy)
print("\nAverage testing accuracy:",np.mean(test_fold_accuracy))

print("with less dense hidden layers")

test_fold_accuracy_less_dense = []
for train, test in kfolds.split(data.data,ohe_target) :
    nn = models.Sequential()
    nn.add(layers.Dense(10,activation='relu',input_dim=5))
    nn.add(layers.Dense(7, activation='relu'))
    nn.add(layers.Dense(2,activation="softmax"))
    nn.compile(optimizer="adam",loss ="categorical_crossentropy",metrics=["accuracy"])
    n_epochs = 100
    nn.fit(data.data.iloc[train],ohe_target[train],epochs=n_epochs)
    s = nn.evaluate(data.data.iloc[test],ohe_target[test])
    test_fold_accuracy_less_dense.append(s[1])
    print("Fold",len(test_fold_accuracy_less_dense),"Accuracy =",s[1])
   
print("\nTesting accuracy for all folds:",test_fold_accuracy_less_dense)
less_dense=np.mean(test_fold_accuracy_less_dense)
print("\nAverage testing accuracy:",np.mean(test_fold_accuracy_less_dense))

print("with high dense hidden layers")

test_fold_accuracy_high_dense = []
for train, test in kfolds.split(data.data,ohe_target) :
    nn = models.Sequential()
    nn.add(layers.Dense(10,activation='relu',input_dim=5))
    nn.add(layers.Dense(15, activation='relu'))
    nn.add(layers.Dense(2,activation="softmax"))
    nn.compile(optimizer="adam",loss ="categorical_crossentropy",metrics=["accuracy"])
    n_epochs = 100
    nn.fit(data.data.iloc[train],ohe_target[train],epochs=n_epochs)
    s = nn.evaluate(data.data.iloc[test],ohe_target[test])
    test_fold_accuracy_high_dense.append(s[1])
    print("Fold",len(test_fold_accuracy_high_dense),"Accuracy =",s[1])
   
print("\nTesting accuracy for all folds:",test_fold_accuracy_high_dense)
high_dense=test_fold_accuracy_high_dense
print("\nAverage testing accuracy:",np.mean(test_fold_accuracy_high_dense))

print("with two hidden layers")

test_fold_accuracy_two_hidden = []
for train, test in kfolds.split(data.data,ohe_target) :
    nn = models.Sequential()
    nn.add(layers.Dense(10,activation='relu',input_dim=5))
    nn.add(layers.Dense(7, activation='relu'))
    nn.add(layers.Dense(3, activation='relu'))
    nn.add(layers.Dense(2,activation="softmax"))
    nn.compile(optimizer="adam",loss ="categorical_crossentropy",metrics=["accuracy"])
    n_epochs = 100
    nn.fit(data.data.iloc[train],ohe_target[train],epochs=n_epochs)
    s = nn.evaluate(data.data.iloc[test],ohe_target[test])
    test_fold_accuracy_two_hidden.append(s[1])
    print("Fold",len(test_fold_accuracy_two_hidden),"Accuracy =",s[1])
   
print("\nTesting accuracy for all folds:",test_fold_accuracy_two_hidden)
two_hidden=np.mean(test_fold_accuracy)
print("\nAverage testing accuracy two hidden layers:",np.mean(test_fold_accuracy_two_hidden))

print("*** Statistical significance***")

import scipy
no_hidden = scipy.stats.ttest_ind(test_fold_accuracy_high_dense,test_fold_accuracy)
less_dense = scipy.stats.ttest_ind(test_fold_accuracy_high_dense,test_fold_accuracy_less_dense)
high_dense = scipy.stats.ttest_ind(test_fold_accuracy_high_dense,test_fold_accuracy_two_hidden)

print("no hidden p value",no_hidden)
print("less dense p value",less_dense)
print("high dense p value",high_dense)
