
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

characters = pd.read_csv('chars.csv', sep=',')

train_.shape, validate_.shape, #test.shape


# In[2]:

characters.shape


# In[3]:

train = train_.query('DSMCRIT < 14')#.sample(20000)
validate = validate_.query('DSMCRIT < 14')#.sample(10000)
#print train['DSMCRIT'].value_counts()
print train.shape


# In[4]:

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

X_c = characters[retain_list]
Y_c = characters["DSMCRIT"]
#X_train = train.drop(drop_list + drop_list_select, axis=1)
Y_train = train["DSMCRIT"]
#X_validate = validate.drop(drop_list + drop_list_select, axis=1)
#Y_validate = validate["DSMCRIT"]
#X_test  = test.drop(drop_list, axis=1)
X_train.shape, #X_validate.shape, #X_test.shape



# In[5]:

#print X_train.columns.tolist()


# In[6]:

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


X_char = enc.transform(X_c).toarray()

print X_train.shape, X_char.shape#X_validate.shape


# In[7]:

#kfold
kf = 3


# In[8]:

# Logistic Regression
logreg = LogisticRegression(n_jobs=-1)
logreg.fit(X_train, Y_train)
#Y_pred = logreg.predict(X_test)

l_acc_log = cross_val_score(logreg, X_train, Y_train, cv=kf)
acc_log = round(np.mean(l_acc_log), 3)

l_acc_log = ['%.3f' % elem for elem in l_acc_log]
print l_acc_log
print acc_log


# In[9]:

# Random Forest (slow)

random_forest = RandomForestClassifier(n_estimators=200, max_depth=20, n_jobs=-1)
random_forest.fit(X_train, Y_train)
#Y_pred = random_forest.predict(X_test)

l_acc_random_forest = cross_val_score(random_forest, X_train, Y_train, cv=kf)
acc_random_forest = round(np.mean(l_acc_random_forest), 3)

l_acc_random_forest = ['%.3f' % elem for elem in l_acc_random_forest]
print l_acc_random_forest
print acc_random_forest


# In[ ]:

yp_char = logreg.predict_proba(X_char)
print yp_char
#print Y_c


# In[ ]:

yp_char = random_forest.predict_proba(X_char)
print yp_char

