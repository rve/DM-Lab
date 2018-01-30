
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


# In[ ]:

da = pd.read_csv('../final.csv', sep=',')


# In[ ]:

USE_SUB2 = []

for row in da['SUB2']:
    if row == 1:
        USE_SUB2.append(0)
    else:
        USE_SUB2.append(1)
        


# In[ ]:

da['USE_SUB2'] = USE_SUB2


# In[ ]:

#for drop_list_suppl , we'll handle the missing values later
drop_list = ['YEAR', 'REGION', 'DIVISION', 'PRIMPAY', 'PRIMINC', 'DAYWAIT',
             'HLTHINS', 'CBSA10', 'STFIPS', 'CASEID', 'METHUSE', 
             'ALCFLG', 'COKEFLG', 'MARFLG', 'HERFLG', 'METHFLG', 'OPSYNFLG', 'PCPFLG', 'HALLFLG', 'MTHAMFLG', 
             'AMPHFLG', 'STIMFLG', 'BENZFLG', 'TRNQFLG', 'BARBFLG', 'SEDHPFLG', 'INHFLG', 'OTCFLG', 'OTHERFLG', 
             'ALCDRUG', ]
drop_list_suppl = ['FREQ2', 'FREQ3', 'FRSTUSE2', 'FRSTUSE3', 'ROUTE2', 'ROUTE3', ]

#howto deal with: priminc
t_df = da.drop(drop_list + drop_list_suppl, axis=1)
t_df['DETCRIM'].replace(to_replace=[-9], value = 0, inplace=True)
t_df['DETNLF'].replace(to_replace=[-9], value = 0, inplace=True)
t_df['IDU'].replace(to_replace=[-9], value = 0, inplace=True)
t_df.ix[t_df.GENDER.isin([1]), 'PREG'] = 2

pp_da = t_df
pp_da = pp_da[(pp_da >= 0).all(1)]
print (pp_da.shape)
print (pp_da.columns.tolist())


# In[ ]:

top6 = pp_da[pp_da['SUB2'].isin([2,3,4,5,7,10])]

none = pp_da[pp_da['SUB2'].isin([1])]
nonesample = none.sample(350000)

top7_scaled = pd.concat([top6, nonesample])


# In[ ]:

train_scaled, validation_scaled = np.split(top7_scaled.sample(frac=1), [int(.8*len(top7_scaled))])


# In[3]:

#load virtual characters
characters = pd.read_csv('finalcharacters.csv', sep=',')


# In[ ]:

characters = characters.drop(drop_list + drop_list_suppl, axis=1)


# In[7]:

USE_SUB2 = []

for row in characters['SUB2']:
    if row == 1:
        USE_SUB2.append(0)
    else:
        USE_SUB2.append(1)
        
characters['USE_SUB2'] = USE_SUB2


# In[8]:

X_train = train_scaled.drop(['SUB2','SUB3','NUMSUBS','USE_SUB2'], axis=1)
Y_train = train_scaled['USE_SUB2']
X_val = validation_scaled.drop(['SUB2','SUB3','NUMSUBS','USE_SUB2'], axis=1)
Y_val = validation_scaled['USE_SUB2']
X_c = characters.drop(['SUB2','SUB3','NUMSUBS','USE_SUB2'], axis=1)
Y_char = characters['USE_SUB2']


# In[9]:

#one hot

# 1. INSTANTIATE
enc = preprocessing.OneHotEncoder()

# 2. FIT
enc.fit(X_train)


# In[ ]:

# 3. Transform
X_train_enc = enc.transform(X_train).toarray()

X_train_enc.shape


# In[ ]:

# 4. Transform test
X_val_enc = enc.transform(X_val).toarray()
X_char = enc.transform(X_c).toarray()

print (X_val_enc.shape, X_char)


# In[ ]:

# Random Forest

random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train_enc, Y_train)
random_forest.score(X_train_enc, Y_train)


# In[ ]:

Y_pred_rf = random_forest.predict(X_val_enc)
metrics.accuracy_score(Y_pred_rf, Y_val)


# In[ ]:

metrics.recall_score(Y_val, Y_pred_rf, average='macro')


# In[ ]:

yp_char = random_forest.predict_proba(X_char)
print (yp_char)


# In[ ]:

Counter(Y_char)

