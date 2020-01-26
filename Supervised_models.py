
# coding: utf-8

# In[1]:

import pandas as pd
import pandas as pd
import numpy as np



from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import MultinomialNB

from sklearn import metrics

from scipy.sparse import *
from scipy.io import mmread


# # Load Data

# In[2]:

train1 =  mmread('trainSingle_sparse.mtx').tocsr() 
test1 = mmread('testSingle_sparse.mtx').tocsr()

train2 = mmread('train2grams_sparse.mtx').tocsr()
test2 = mmread('test2grams_sparse.mtx').tocsr()

train3 = mmread('train3grams_sparse.mtx').tocsr()
test3 = mmread('test3grams_sparse.mtx').tocsr()



# # Define Data

# In[52]:

dsets = ['single', '2grams', '3grams']
trains = [train1, train2, train3]
tests = [test1, test2, test3]


names = ["LR", "RFC","NN"]
classifiers = [LogisticRegression(), RandomForestClassifier(), MultinomialNB()]


#prediction frame
predictions_valid = pd.DataFrame()
predictions_test = pd.DataFrame()

#error frame
index = pd.Series(index = ['brier_valid', 'brier_test', 'loss_valid', 'loss_test', 'pcc_valid', 'pcc_test'])
errors = pd.DataFrame(index)

for dset, train, test in zip(dsets, trains, tests):
    keep = list(set(range(train.shape[1]))-set([0, 1, 2, 3, 4]))
    X_train = train[:260618, keep]
    X_valid = train[260618:, keep]
    X_test = test[:, keep]

    y_train = (train[:260618,:].getcol(4).toarray()).astype(int).ravel()
    y_valid = (train[260618:,:].getcol(4).toarray()).astype(int).ravel()
    y_test = (test.getcol(4).toarray()).astype(int).ravel()
    
    for name, clf in zip(names, classifiers):
        clf.fit(X_train, y_train)
        valid_pred = clf.predict_proba(X_valid)
        test_pred = clf.predict_proba(X_test)
        
        loss_valid = metrics.log_loss(y_valid, valid_pred)
        loss_test = metrics.log_loss(y_test, test_pred)
        brier_valid = (sum((y_valid - valid_pred[:,1])**2))/X_valid.shape[0]
        brier_test = (sum((y_test - test_pred[:,1])**2))/test.shape[0]
        pcc_valid = clf.score(X_valid, y_valid)
        pcc_test = clf.score(X_test, y_test)
        
        predictions_valid[name + '_' + dset] = valid_pred[:,1]
        predictions_test[name + '_' + dset] = test_pred[:,1]
        
        errors[name + '_' + dset] = pd.Series([brier_valid, brier_test, loss_valid, loss_test, pcc_valid, pcc_test]).values


# In[67]:

predictions_valid.to_csv("valid_predictions.csv",  index = False)
predictions_test.to_csv("test_predictions.csv", index = False)
errors.to_csv("pred_errors.csv", index = False)


# In[51]:

clf.score(X_test, y_test)


# In[ ]:



