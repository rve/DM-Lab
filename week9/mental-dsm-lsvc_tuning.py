
# coding: utf-8

# In[1]:

#import matplotlib.pyplot as plt
#import seaborn as sns


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



train_=pd.read_csv('../train_allcols.csv')
validate_=pd.read_csv('../validate_allcols.csv')
#test=pd.read_csv('../testwDSM.csv')

train_.shape, validate_.shape, #test.shape


# In[2]:

train = train_.query('DSMCRIT > 13')
validate = validate_.query('DSMCRIT > 13')
#print train['DSMCRIT'].value_counts()
print train.shape


# In[3]:

#alcohol
#print train['DSMCRIT'].value_counts() / train['DSMCRIT'].count()
#print train['SUB1'].value_counts() / train['SUB1'].count()


# In[4]:

#train.query('SUB1 == 4')['DSMCRIT'].value_counts() / train.query('SUB1 == 4')['DSMCRIT'].count()


# In[6]:

train = pd.concat([train, validate])
#train.describe()
#train = train.sample(20000)
#validate = validate.sample(6000)
print train.shape, #validate.shape, #validate.head(3)


# In[3]:

#train = train.query('SUB1 <= 10').query('SUB2 <= 10')
#validate = validate.query('SUB1 <= 10').query('SUB2 <= 10')

drop_list = ['DSMCRIT',  #'NUMSUBS'
             ]
drop_list_select = ['RACE', 'PREG', 'ARRESTS', 'PSYPROB', 'DETNLF', 'ETHNIC', 'MARSTAT', 'GENDER', 'EDUC'
                   ,'LIVARAG', 'EMPLOY', 'SUB3']

retain_list = ['RACE','PCPFLG','PRIMINC','LIVARAG','BENZFLG','HLTHINS','GENDER','ROUTE3','PRIMPAY',
               'MARSTAT','PSYPROB','ROUTE2','EMPLOY','SUB2','FRSTUSE3','FREQ3','FRSTUSE2','OTHERFLG',
               'EDUC','FREQ2','FREQ1','YEAR',
               'PSOURCE','DETCRIM','DIVISION','REGION','NOPRIOR','NUMSUBS','ALCDRUG',
               'METHUSE','FRSTUSE1','AGE','COKEFLG','OPSYNFLG','IDU','SERVSETA','ROUTE1','MARFLG',
               'MTHAMFLG','HERFLG',
               'ALCFLG','SUB1']
X_train = train[retain_list]
#X_train = train.drop(drop_list + drop_list_select, axis=1)
Y_train = train["DSMCRIT"]
#X_validate = validate.drop(drop_list + drop_list_select, axis=1)
#Y_validate = validate["DSMCRIT"]
#X_test  = test.drop(drop_list, axis=1)
X_train.shape, #X_validate.shape, #X_test.shape



# In[6]:

print X_train.columns.tolist()

#one hot
from sklearn import preprocessing

# 1. INSTANTIATE
enc = preprocessing.OneHotEncoder()

# 2. FIT
enc.fit(X_train)

# 3. Transform
onehotlabels = enc.transform(X_train).toarray()
X_train = onehotlabels

#onehotlabels = enc.transform(X_validate).toarray()
#X_validate = onehotlabels

X_train.shape, #X_validate.shape


# In[9]:

#kfold
kf = 3

# Linear SVC
linear_svc = LinearSVC(dual=False)
linear_svc.fit(X_train, Y_train)
#Y_pred = linear_svc.predict(X_test)

l_acc_linear_svc = cross_val_score(linear_svc, X_train, Y_train, cv=kf)
acc_linear_svc = round(np.mean(l_acc_linear_svc), 3)
l_acc_linear_svc = ['%.3f' % elem for elem in l_acc_linear_svc]
print 'original score'
print l_acc_linear_svc
print acc_linear_svc


# In[ ]:

from sklearn.model_selection import GridSearchCV, cross_val_score
#Cs = np.logspace(-2, -1, 100)
paras = {'penalty':('l1', 'l2'), 'C': np.logspace(-2,-1,20), 
        'multi_class': ('crammer_singer', 'ovr'), }
clf = GridSearchCV(estimator=linear_svc, param_grid=paras,
                   n_jobs=-1)
clf.fit(X_train, Y_train)        

print 'tuning: Cs:'

print paras 

print 'best score'

print clf.best_score_                                  

print 'best parameter'

print clf.best_estimator_          
