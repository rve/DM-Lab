
# coding: utf-8

# In[1]:

import pandas as pd
import seaborn as sns
import numpy as np
import random as rnd
import matplotlib as mp
import matplotlib.pyplot as plt


from collections import Counter

from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from imblearn.over_sampling import SMOTE


# In[57]:

train_=pd.read_csv('../DM-Lab/train_allcols.csv')
validate_=pd.read_csv('../DM-Lab/validate_allcols.csv')


# In[61]:

train = train_.query('SUB2 != 1')
validate = validate_.query('SUB2 != 1')
print (train['SUB2'].value_counts())


# In[63]:

retain_list = ['EMPLOY','GENDER','FREQ1','YEAR','EDUC','PSYPROB','PSOURCE','SERVSETA','DETCRIM',
               'REGION','NOPRIOR','DIVISION','DSMCRIT','ROUTE1','SUB1','AGE','IDU','SUB3','ROUTE3',
               'FREQ3','FRSTUSE3','FREQ2','FRSTUSE2']

train = train[train['SUB2'].isin([2,3,4,5,7,10])]
validate = validate[validate['SUB2'].isin([2,3,4,5,7,10])]

X_train = train[retain_list]
y_train = train["SUB2"]
X_validate = validate[retain_list]
y_validate = validate["SUB2"]
X_train.shape, X_validate.shape


# In[64]:

#one hot

# 1. INSTANTIATE
enc = preprocessing.OneHotEncoder()

# 2. FIT
enc.fit(X_train)


# In[65]:

# 3. Transform
X_train_enc = enc.transform(X_train).toarray()

X_train_enc.shape


# In[67]:

# 4. Transform test
X_val_enc = enc.transform(X_validate).toarray()

X_val_enc.shape


# In[68]:

# Random Forest

random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train_enc, y_train)
random_forest.score(X_train_enc, y_train)


# In[69]:

yp_rf = random_forest.predict(X_val_enc)
print (metrics.accuracy_score(yp_rf, y_validate))


# In[70]:

print (metrics.recall_score(y_validate, yp_rf, average='macro'))


# In[74]:

print (metrics.classification_report(y_validate, yp_rf))


# In[ ]:

from sklearn.grid_search import GridSearchCV
from sklearn.datasets import make_classification

# Build a classification task using 3 informative features
'''X, y = make_classification(n_samples=1000,
                           n_features=10,
                           n_informative=3,
                           n_redundant=0,
                           n_repeated=0,
                           n_classes=2,
                           random_state=0,
                           shuffle=False)'''


rfc = RandomForestClassifier(n_jobs=-1, max_features='sqrt', n_estimators=50, oob_score = True) 

param_grid = { 
    'n_estimators': [100, 200, 250],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth': [15, 20, 25],
    'min_samples_leaf': [10, 25, 50, 100],
    'bootstrap': [True, False],
}

CV_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv= 5)
CV_rfc.fit(X_train_enc, y_train)
print (CV_rfc.best_params_)


# In[ ]:



