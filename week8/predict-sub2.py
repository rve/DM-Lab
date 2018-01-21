
# coding: utf-8

# In[102]:

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

train.shape, validate.shape, #test.shape


# In[103]:

#train.describe()
train = train_.sample(20000)
validate = validate_.sample(6000)
train.shape, validate.shape, validate.head(2)


# In[104]:

#train = train.query('SUB1 <= 10').query('SUB2 <= 10')
#validate = validate.query('SUB1 <= 10').query('SUB2 <= 10')

drop_list = ['SUB2',  'ROUTE2', 'FREQ2', 'FRSTUSE2', 'SUB3', 'ROUTE3', 'FREQ3', 'FRSTUSE3', 'NUMSUBS'
             ]
retain_list = ['DETNLF','ETHNIC','DAYWAIT','RACE','VET','LIVARAG','PRIMINC','FRSTUSE1','HLTHINS','MARSTAT',
               'METHUSE','GENDER','EMPLOY','FREQ1','PSYPROB','YEAR','EDUC','PSOURCE','REGION',
               'NOPRIOR','SERVSETA','DETCRIM','DIVISION','DSMCRIT','ROUTE1','AGE','SUB1','IDU',]
X_train = train[retain_list]
Y_train = train["SUB2"]
X_validate = validate[retain_list]
Y_validate = validate["SUB2"]
#X_test  = test.drop(drop_list, axis=1)
X_train.shape, X_validate.shape, #X_test.shape



# In[105]:

print X_train.columns.tolist()


# In[106]:

from sklearn.feature_selection import SelectKBest, SelectPercentile
from sklearn.feature_selection import f_classif,chi2
#Selector_f = SelectPercentile(f_classif, percentile=25)
Selector_f = SelectKBest(f_classif, k=11)
Selector_f.fit(X_train,Y_train)

X_new = SelectKBest(chi2, k=11).fit_transform(X_train, Y_train)
zipped = zip(X_train.columns.tolist(),Selector_f.scores_)
ans = sorted(zipped, key=lambda x: x[1])
for n,s in ans:
     print 'F-score: %3.2ft for feature %s' % (s,n)


# In[107]:

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


# In[108]:

# Logistic Regression
logreg = LogisticRegression()
logreg.fit(X_train, Y_train)
#Y_pred = logreg.predict(X_test)
acc_log = round(logreg.score(X_validate, Y_validate) * 100, 2)
print acc_log


# In[109]:

# Perceptron

perceptron = Perceptron()
perceptron.fit(X_train, Y_train)
#Y_pred = perceptron.predict(X_test)
acc_perceptron = round(perceptron.score(X_validate, Y_validate) * 100, 2)
print acc_perceptron


# In[110]:

# Stochastic Gradient Descent

sgd = SGDClassifier()
sgd.fit(X_train, Y_train)
#Y_pred = sgd.predict(X_test)
acc_sgd = round(sgd.score(X_validate, Y_validate) * 100, 2)
print acc_sgd


# In[111]:

# Random Forest (slow)

random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, Y_train)
#Y_pred = random_forest.predict(X_test)
random_forest.score(X_train, Y_train)
acc_random_forest = round(random_forest.score(X_validate, Y_validate) * 100, 2)
print acc_random_forest

#print cross_val_score(random_forest, X_validate, Y_validate)


# In[112]:

from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier, RandomForestClassifier
#Adaboost 
ada_boost = AdaBoostClassifier(random_state=1851)
ada_boost.fit(X_train, Y_train) 

acc_ada = round(ada_boost.score(X_validate, Y_validate) * 100, 2)
print acc_ada


# In[113]:

# Stochastic Gradient Descent
sgd = SGDClassifier()
sgd.fit(X_train, Y_train)
#Y_pred = sgd.predict(X_test)
acc_sgd = round(sgd.score(X_validate, Y_validate) * 100, 2)
print acc_sgd


# In[114]:

# Linear SVC
linear_svc = LinearSVC()
linear_svc.fit(X_train, Y_train)
#Y_pred = linear_svc.predict(X_test)
acc_linear_svc = round(linear_svc.score(X_train, Y_train) * 100, 2)
print acc_linear_svc

import xgboost as xgb
xgb_boost = xgb.XGBClassifier(seed=1850, n_jobs=-1)
xgb_boost.fit(X_train, Y_train) 
acc_xgb = round(linear_svc.score(X_train, Y_train) * 100, 2)

# In[115]:

print 'predict-sub2-woflags-newsplit-sample20000'
models = pd.DataFrame({
    'Model': [ 'Logistic Regression', 
              'Random Forest',  'Perceptron', 
              'Stochastic Gradient Decent', 'Linear SVC', 
               'AdaBoost', 'XGBoost'],
    'Validate Set Score': [acc_log, 
              acc_random_forest,  acc_perceptron, 
              acc_sgd, acc_linear_svc, acc_ada, acc_xgb]
    })
models.sort_values(by='Validate Set Score', ascending=False)

