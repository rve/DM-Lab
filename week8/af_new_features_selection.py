
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

get_ipython().magic('matplotlib notebook')
sns.set(font_scale=1)
plt.style.use('ggplot')


# In[2]:

da = pd.read_csv('../final.csv', sep=',')


# In[4]:

#for drop_list_suppl , we'll handle the missing values later
drop_list = ['REGION', 'DIVISION', 'PRIMINC',
             'CBSA10', 'STFIPS', 'CASEID', 'METHUSE', 
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


# In[7]:

top6 = pp_da[pp_da['SUB2'].isin([2,3,4,5,7,10])]

none = pp_da[pp_da['SUB2'].isin([1])]
nonesample = none.sample(130000)

top7_scaled = pd.concat([top6, nonesample])


# In[8]:

X = top7_scaled.drop(['SUB2','SUB3','NUMSUBS'], axis=1)
y = top7_scaled['SUB2']


# In[9]:

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    stratify=y, 
                                                    test_size=0.20)


# In[10]:

#one hot

# 1. INSTANTIATE
enc = preprocessing.OneHotEncoder()

# 2. FIT
enc.fit(X_train)


# In[11]:

# 3. Transform
X_train_enc = enc.transform(X_train).toarray()

X_train_enc.shape


# In[12]:

# 4. Transform test
X_test_enc = enc.transform(X_test).toarray()

X_test_enc.shape


# In[51]:

# Random Forest

random_forest = RandomForestClassifier(n_estimators=100, max_depth=20, random_state=100)
random_forest.fit(X_train_enc, y_train)
random_forest.score(X_train_enc, y_train)


# In[52]:

yp_rf = random_forest.predict(X_test_enc)
acc_rf = metrics.accuracy_score(yp_rf, y_test)


# In[53]:

rec_rf = metrics.recall_score(y_test, yp_rf, average='macro')


# In[ ]:

print(acc_rf, rec_rf)


# In[54]:

metrics.classification_report(y_test, yp_rf)


# In[18]:

# Decision Tree

decision_tree = DecisionTreeClassifier(random_state=1)
decision_tree.fit(X_train_enc, y_train)
decision_tree.score(X_train_enc, y_train)


# In[19]:

yp_dt = decision_tree.predict(X_test_enc)
acc_dt = metrics.accuracy_score(yp_dt, y_test)


# In[20]:

rec_dt = metrics.recall_score(y_test, yp_dt, average='macro')


# In[ ]:

print(acc_dt, rec_dt)

