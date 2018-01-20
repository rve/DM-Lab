
# coding: utf-8

# In[1]:


#import csv from the dataset
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


train=pd.read_csv('../train.csv')
validate=pd.read_csv('../validate.csv')
#test=pd.read_csv('../testwDSM.csv')

train.shape, validate.shape, #test.shape


# In[2]:


#train.describe()
train.shape


# In[3]:


#train = train.query('SUB1 <= 10').query('SUB2 <= 10')
#validate = validate.query('SUB1 <= 10').query('SUB2 <= 10')

drop_list = ['DSMCRIT', 'STFIPS', 
             'ALCFLG', 'COKEFLG', 'MARFLG', 'HERFLG', 'METHFLG', 'OPSYNFLG', 'PCPFLG', 'HALLFLG', 'MTHAMFLG', 
             'AMPHFLG', 'STIMFLG', 'BENZFLG', 'TRNQFLG', 'BARBFLG', 'SEDHPFLG', 'INHFLG', 'OTCFLG', 'OTHERFLG', 
             'ALCDRUG', 
             ]

X_train = train.drop(drop_list, axis=1)
Y_train = train["DSMCRIT"]
X_validate = validate.drop(drop_list, axis=1)
Y_validate = validate["DSMCRIT"]
#X_test  = test.drop(drop_list, axis=1)
X_train.shape, X_validate.shape, #X_test.shape



# In[4]:


print X_train.columns.tolist()


# In[5]:


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


# In[6]:


# Decision Tree

decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, Y_train)
#Y_pred = decision_tree.predict(X_test)
acc_decision_tree = round(decision_tree.score(X_validate, Y_validate) * 100, 2)
print acc_decision_tree
ts_acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 2)
print ts_acc_decision_tree
#print cross_val_score(decision_tree, X_validate, Y_validate)


# In[7]:


# Logistic Regression
logreg = LogisticRegression()
logreg.fit(X_train, Y_train)
#Y_pred = logreg.predict(X_test)
acc_log = round(logreg.score(X_validate, Y_validate) * 100, 2)
print acc_log
ts_acc_log = round(logreg.score(X_train, Y_train) * 100, 2)
print ts_acc_log
#print cross_val_score(logreg, X_validate, Y_validate)


# In[8]:


# Perceptron

perceptron = Perceptron()
perceptron.fit(X_train, Y_train)
#Y_pred = perceptron.predict(X_test)
acc_perceptron = round(perceptron.score(X_validate, Y_validate) * 100, 2)
print acc_perceptron
ts_acc_perceptron = round(perceptron.score(X_train, Y_train) * 100, 2)
print ts_acc_perceptron
#print cross_val_score(perceptron, X_validate, Y_validate)


# In[9]:


# Gaussian Naive Bayes

gaussian = GaussianNB()
gaussian.fit(X_train, Y_train)
#Y_pred = gaussian.predict(X_test)
acc_gaussian = round(gaussian.score(X_validate, Y_validate) * 100, 2)
print acc_gaussian
ts_acc_gaussian = round(gaussian.score(X_train, Y_train) * 100, 2)
print ts_acc_gaussian
#print cross_val_score(gaussian, X_validate, Y_validate)


# In[10]:


# Random Forest (slow)

print 'random forrest'
random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, Y_train)
#Y_pred = random_forest.predict(X_test)
random_forest.score(X_train, Y_train)
acc_random_forest = round(random_forest.score(X_validate, Y_validate) * 100, 2)
print acc_random_forest
ts_acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)
print ts_acc_random_forest

#print cross_val_score(random_forest, X_validate, Y_validate)


# In[11]:


# 2-nn
print '2-nn'
knn = KNeighborsClassifier(n_neighbors = 2)
knn.fit(X_train, Y_train)
#Y_pred = knn.predict(X_test)
acc_2nn = round(knn.score(X_validate, Y_validate) * 100, 2)
print acc_2nn
ts_acc_2nn = round(knn.score(X_train, Y_train) * 100, 2)
print ts_acc_2nn
#print cross_val_score(knn, X_validate, Y_validate)


# In[12]:


# Stochastic Gradient Descent

sgd = SGDClassifier()
sgd.fit(X_train, Y_train)
#Y_pred = sgd.predict(X_test)
acc_sgd = round(sgd.score(X_validate, Y_validate) * 100, 2)
print acc_sgd
ts_acc_sgd = round(sgd.score(X_train, Y_train) * 100, 2)
print ts_acc_sgd
#print cross_val_score(sgd, X_validate, Y_validate)


# In[13]:


# Support Vector Machines
print 'svm'

svc = SVC()
svc.fit(X_train, Y_train)
#Y_pred = svc.predict(X_test)
acc_svc = round(svc.score(X_validate, Y_validate) * 100, 2)
print acc_svc
ts_acc_svc = round(svc.score(X_train, Y_train) * 100, 2)
print ts_acc_svc


# In[14]:


print ' Linear SVC'
linear_svc = LinearSVC()
linear_svc.fit(X_train, Y_train)
#Y_pred = linear_svc.predict(X_test)
acc_linear_svc = round(linear_svc.score(X_train, Y_train) * 100, 2)
print acc_linear_svc
ts_acc_linear_svc = round(linear_svc.score(X_validate, Y_validate) * 100, 2)
print ts_acc_linear_svc


# In[17]:


print 'predict-dsm-woflags-newsplit-sample20000'
models = pd.DataFrame({
    'Model': ['Support Vector Machines', '2-NN', 'Logistic Regression', 
              'Random Forest', 'Naive Bayes', 'Perceptron', 
              'Stochastic Gradient Decent', 'Linear SVC', 
              'Decision Tree'],
    'Train Set Score': [ts_acc_svc, ts_acc_2nn, ts_acc_log, 
              ts_acc_random_forest, ts_acc_gaussian, ts_acc_perceptron, 
              ts_acc_sgd, ts_acc_linear_svc, ts_acc_decision_tree],
    'Validate Set Score': [acc_svc, acc_2nn, acc_log, 
              acc_random_forest, acc_gaussian, acc_perceptron, 
              acc_sgd, acc_linear_svc, acc_decision_tree]
    })
ans = models.sort_values(by='Validate Set Score', ascending=False)

print ans
print ans.values
