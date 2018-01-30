
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


# In[57]:

train_=pd.read_csv('../train_allcols.csv')
validate_=pd.read_csv('../validate_allcols.csv')
characters=pd.read_csv('../week9/finalcharacters.csv')


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
X_char = character[retain_list]
y_char = character["SUB2"]
X_train.shape, X_validate.shape, X_char.shape


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
X_char_enc = enc.transform(X_char).toarray()

X_val_enc.shape


# In[68]:

# Random Forest

random_forest = RandomForestClassifier(n_estimators=200, max_depth=50)
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

yp_char = random_forest.predict_proba(X_char_enc)
print (yp_char)

