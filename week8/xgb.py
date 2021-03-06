
# coding: utf-8

# In[34]:


import pandas as pd
import numpy as np
import random as rnd

from sklearn.cross_validation import KFold, cross_val_score

# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier



train=pd.read_csv('../trainval.csv')
validate=pd.read_csv('../validate.csv')
#test=pd.read_csv('../testwDSM.csv')
kf = KFold(n_splits=10)

print train.shape 


# In[35]:


#train.describe()
train = train.sample(10000)
validate = validate.sample(6000)
train.shape, validate.shape, validate.head(2)


# In[36]:


#train = train.query('SUB1 <= 10').query('SUB2 <= 10')
#validate = validate.query('SUB1 <= 10').query('SUB2 <= 10')

drop_list = ['DSMCRIT', 'YEAR', 'STFIPS', 
             'ALCFLG', 'COKEFLG', 'MARFLG', 'HERFLG', 'METHFLG', 'OPSYNFLG', 'PCPFLG', 'HALLFLG', 'MTHAMFLG', 
             'AMPHFLG', 'STIMFLG', 'BENZFLG', 'TRNQFLG', 'BARBFLG', 'SEDHPFLG', 'INHFLG', 'OTCFLG', 'OTHERFLG', 
             'ALCDRUG', #'NUMSUBS'
             ]

X_train = train.drop(drop_list, axis=1)
Y_train = train["DSMCRIT"]
X_validate = validate.drop(drop_list, axis=1)
Y_validate = validate["DSMCRIT"]
#X_test  = test.drop(drop_list, axis=1)
X_train.shape, X_validate.shape, #X_test.shape



# In[37]:


print X_train.columns.tolist()


# In[38]:


#one hot
from sklearn import preprocessing

# 1. INSTANTIATE
enc = preprocessing.OneHotEncoder()

# 2. FIT
enc.fit(X_train)

# 3. Transform
onehotlabels = enc.transform(X_train).toarray()
#onehotlabels.shape
X_train = onehotlabels

onehotlabels = enc.transform(X_validate).toarray()
X_validate = onehotlabels

print X_train.shape

# XGBoost
import xgboost as xgb
xgb_boost = xgb.XGBClassifier(seed=1850, n_jobs=-1)
xgb_boost.fit(X_train, Y_train) 
l_acc_xgb = cross_val_score(xgb, X_train, Y_train, cv=5)
acc_xgb = round(np.mean(l_acc_xgb), 3)
l_acc_xgb = ['%.3f' % elem for elem in l_acc_xgb]

print 'xgboost'
print l_acc_xgb
print acc_xgb

