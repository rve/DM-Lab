
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
#from imblearn.over_sampling import SMOTE


# In[3]:

train_=pd.read_csv('../train_allcols.csv')
validate_=pd.read_csv('../validate_allcols.csv')


# In[6]:

'''train = train_.sample(10000)
validate = validate_.sample(3000)
train_.shape,'''


# In[7]:

USE_SUB2 = []

for row in train['SUB2']:
    if row == 1:
        USE_SUB2.append(0)
    else:
        USE_SUB2.append(1)
        
train['USE_SUB2'] = USE_SUB2


# In[8]:

USE_SUB2 = []

for row in validate['SUB2']:
    if row == 1:
        USE_SUB2.append(0)
    else:
        USE_SUB2.append(1)
        
validate['USE_SUB2'] = USE_SUB2


# In[10]:

# for use_sub2

'''X_train = train.drop(['USE_SUB2'], axis=1)
y_train = train['USE_SUB2']

from sklearn.feature_selection import SelectKBest, SelectPercentile
from sklearn.feature_selection import f_classif,chi2

Selector_f = SelectKBest(f_classif, k=10)
Selector_f.fit(X_train,y_train)

zipped = zip(X_train.columns.tolist(),Selector_f.scores_)
ans = sorted(zipped, key=lambda x: x[1])
for n,s in ans:    
    if 'FLG' in n: 
        pass
    else:
        print ('F-score: %3.2ft for feature %s' % (s,n))'''


# In[28]:

retain_list = ['EMPLOY','GENDER','FREQ1','YEAR','EDUC','PSYPROB','PSOURCE','SERVSETA','DETCRIM','MARSTAT', 'PRIMINC',
               'REGION','NOPRIOR','DIVISION','DSMCRIT','ROUTE1','SUB1','AGE','IDU','METHUSE']

retain_list_v2 = ['AGE', 'GENDER', 'RACE', 'ETHNIC', 'MARSTAT', 'EDUC', 'EMPLOY', 'DETNLF', 'PREG', 'VET', 'LIVARAG', 
                  'ARRESTS', 'SERVSETA', 'PSOURCE', 'DETCRIM', 'NOPRIOR', 'SUB1', 'ROUTE1', 'FREQ1', 'FRSTUSE1', 
                  'IDU', 'DSMCRIT', 'PSYPROB']

train = train[train['SUB2'].isin([1,2,3,4,5,7,10])]
validate = validate[validate['SUB2'].isin([1,2,3,4,5,7,10])]



X_train = train[retain_list]
y_train = train["USE_SUB2"]
X_validate = validate[retain_list]
y_validate = validate["USE_SUB2"]
X_train.shape, X_validate.shape


# In[44]:

print (sorted(retain_list))
print (sorted(retain_list_v2))


# In[29]:

#one hot

# 1. INSTANTIATE
enc = preprocessing.OneHotEncoder()

# 2. FIT
enc.fit(X_train)


# In[30]:

# 3. Transform
X_train_enc = enc.transform(X_train).toarray()

X_train_enc.shape


# In[31]:

# 4. Transform test
X_val_enc = enc.transform(X_validate).toarray()

X_val_enc.shape


# In[ ]:

print('====RANDOM FOREST====')


# In[34]:

# Random Forest

random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train_enc, y_train)


# In[35]:

yp_rf = random_forest.predict(X_val_enc)
print (metrics.accuracy_score(yp_rf, y_validate))


# In[36]:

print (metrics.recall_score(y_validate, yp_rf, average='macro'))


# In[27]:

print (metrics.classification_report(y_validate, yp_rf))


# In[19]:

from sklearn.grid_search import GridSearchCV
from sklearn.datasets import make_classification

rfc = RandomForestClassifier(n_jobs=-1, max_features='sqrt', n_estimators=50, oob_score = True) 

param_grid = { 
    'n_estimators': [100, 200, 250],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth': [15, 20, 25],
    'min_samples_leaf': [10, 25, 50, 100],
}

CV_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv= 5)
CV_rfc.fit(X_train_enc, y_train)
print (CV_rfc.best_params_)


# In[ ]:

print('====LOGISTIC REGRESSION====')


# In[21]:

#Logistic Regression

logreg = LogisticRegression()
logreg.fit(X_train_enc, y_train)


# In[46]:

yp_lr = logreg.predict(X_val_enc)
metrics.accuracy_score(yp_lr, y_validate)


# In[23]:

metrics.recall_score(y_validate, yp_lr, average='macro')


# In[45]:

print (metrics.classification_report(y_validate, yp_lr))


# In[ ]:



