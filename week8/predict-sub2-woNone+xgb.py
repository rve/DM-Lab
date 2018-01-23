
# coding: utf-8

# In[54]:

#visualization part
#import matplotlib.pyplot as plt
#import seaborn as sns


#basic libs
import pandas as pd
import numpy as np
import random as rnd

from sklearn.model_selection import cross_val_score

# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier



train_=pd.read_csv('../train_allcols.csv')
validate_=pd.read_csv('../validate_allcols.csv')
#test=pd.read_csv('../testwDSM.csv')

train_.shape, validate_.shape, #test.shape


# In[55]:

#train.describe()
train = train_#.sample(20000)
validate = validate_#.sample(10000)
train_.shape, #validate_.shape, validate.head(2)


# In[56]:

train = train.query('SUB2 != 1')
validate = validate.query('SUB2 != 1')

train = train[train.SUB2.isin([2,3,4,5,7,10])]
validate = validate[validate.SUB2.isin([2,3,4,5,7,10])]
print train.shape
print train['SUB2'].value_counts()


# In[57]:

# for sub2



X_train = train.drop(['SUB2'], axis=1)
Y_train = train["SUB2"]

from sklearn.feature_selection import SelectKBest, SelectPercentile
from sklearn.feature_selection import f_classif,chi2
#Selector_f = SelectPercentile(f_classif, percentile=25)
Selector_f = SelectKBest(f_classif, k=10)
Selector_f.fit(X_train,Y_train)

zipped = zip(X_train.columns.tolist(),Selector_f.scores_)
ans = sorted(zipped, key=lambda x: x[1])
for n,s in ans:    
    if 'FLG' in n: 
        pass
    else:
        print 'F-score: %3.2ft for feature %s' % (s,n)
                
#print X_train.columns.tolist()


# In[58]:

# get the sorted feature list
import sys
for n, s in ans:
    
    if 'FLG' in n: 
        pass
    else:
        sys.stdout.write('\'%s\',' % (n))


# In[59]:

#train = train.query('SUB1 <= 10').query('SUB2 <= 10')
#validate = validate.query('SUB1 <= 10').query('SUB2 <= 10')

drop_list = ['SUB2',  'ROUTE2', 'FREQ2', 'FRSTUSE2', 'SUB3', 'ROUTE3', 'FREQ3', 'FRSTUSE3', 'NUMSUBS'
             ]
retain_list = ['EMPLOY','GENDER','FREQ1','YEAR','EDUC','PSYPROB','PSOURCE','SERVSETA','DETCRIM',
               'REGION','NOPRIOR','DIVISION','DSMCRIT','ROUTE1','SUB1','AGE','IDU','SUB3','ROUTE3',
               'FREQ3','FRSTUSE3','FREQ2','FRSTUSE2']
X_train = train[retain_list]
Y_train = train["SUB2"]
X_validate = validate[retain_list]
Y_validate = validate["SUB2"]
#X_test  = test.drop(drop_list, axis=1)
X_train.shape, X_validate.shape, #X_test.shape



# In[ ]:

print X_train.columns.tolist()


# In[ ]:

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

X_train.shape, X_validate.shape


# In[ ]:

# Logistic Regression
logreg = LogisticRegression(C=1)
logreg.fit(X_train, Y_train)
#Y_pred = logreg.predict(X_test)
acc_log = round(logreg.score(X_validate, Y_validate) * 100, 2)
print acc_log


# In[ ]:

# Perceptron

perceptron = Perceptron()
perceptron.fit(X_train, Y_train)
#Y_pred = perceptron.predict(X_test)
acc_perceptron = round(perceptron.score(X_validate, Y_validate) * 100, 2)
print acc_perceptron


# In[ ]:

# Stochastic Gradient Descent

sgd = SGDClassifier()
sgd.fit(X_train, Y_train)
#Y_pred = sgd.predict(X_test)
acc_sgd = round(sgd.score(X_validate, Y_validate) * 100, 2)
print acc_sgd


# In[ ]:

# Random Forest (slow)

random_forest = RandomForestClassifier(n_estimators=100, max_depth=20)
random_forest.fit(X_train, Y_train)
#Y_pred = random_forest.predict(X_test)
random_forest.score(X_train, Y_train)
acc_random_forest = round(random_forest.score(X_validate, Y_validate) * 100, 2)
print acc_random_forest

#print cross_val_score(random_forest, X_validate, Y_validate)


# In[ ]:

# Stochastic Gradient Descent
sgd = SGDClassifier()
sgd.fit(X_train, Y_train)
#Y_pred = sgd.predict(X_test)
acc_sgd = round(sgd.score(X_validate, Y_validate) * 100, 2)
print acc_sgd


# In[ ]:

# Linear SVC
linear_svc = LinearSVC(C=1)
linear_svc.fit(X_train, Y_train)
#Y_pred = linear_svc.predict(X_test)
acc_linear_svc = round(linear_svc.score(X_validate, Y_validate) * 100, 2)
print acc_linear_svc


# In[ ]:

import xgboost as xgb
xgb_boost = xgb.XGBClassifier(seed=1850, n_jobs=-1)
xgb_boost.fit(X_train, Y_train) 
acc_xgb = round(xgb_boost.score(X_validate, Y_validate) * 100, 2)
print acc_xgb


# In[ ]:

print 'predict-top6-sub2-allcols-sample10000'
models = pd.DataFrame({
    'Model': [ 'Logistic Regression', 
              'Random Forest',  'Perceptron', 
              'Stochastic Gradient Decent', 'Linear SVC', 
               'XGBoost'],
    'Validate Set Score': [acc_log, 
              acc_random_forest,  acc_perceptron, 
              acc_sgd, acc_linear_svc, acc_xgb]
    })
print models.sort_values(by='Validate Set Score', ascending=False)

