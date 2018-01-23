
# coding: utf-8

# In[1]:

import pandas as pd
df=pd.read_csv('../final.csv')
df.shape


# In[2]:

#for drop_list_suppl , we'll handle the missing values later
drop_list = ['REGION', 'DIVISION', 'PRIMPAY', 'PRIMINC', 'DAYWAIT',
             'HLTHINS', 'CBSA10', 'CASEID', 'YEAR', 'STFIPS', 
             'ALCFLG', 'COKEFLG', 'MARFLG', 'HERFLG', 'METHFLG', 'OPSYNFLG', 'PCPFLG', 'HALLFLG', 'MTHAMFLG', 
             'AMPHFLG', 'STIMFLG', 'BENZFLG', 'TRNQFLG', 'BARBFLG', 'SEDHPFLG', 'INHFLG', 'OTCFLG', 'OTHERFLG', 
             'ALCDRUG',]
drop_list_suppl = ['FREQ2', 'FREQ3', 'FRSTUSE2', 'FRSTUSE3', 'ROUTE2', 'ROUTE3', ]
                      
drop_list=[]
drop_list_suppl = []

#howto deal with: priminc
#add NOPRIOr freq1 frstuse1 route1 SERVSETA sub3 PSYPROB MARSTAT VET, PSOURCE DETCRIM METHUSE DETNLF IDU PREG
train_df = df.drop(drop_list + drop_list_suppl, axis=1)
train_df['DETCRIM'].replace(to_replace=[-9], value = 0, inplace=True)
train_df['DETNLF'].replace(to_replace=[-9], value = 0, inplace=True)
train_df['IDU'].replace(to_replace=[-9], value = 0, inplace=True)
train_df.ix[train_df.GENDER.isin([1]), 'PREG'] = 2
train_df.ix[train_df.SUB2.isin([1]), 'FREQ2'] = 0
train_df.ix[train_df.SUB2.isin([1]), 'FRSTUSE2'] = 0
train_df.ix[train_df.SUB2.isin([1]), 'ROUTE2'] = 0
train_df.ix[train_df.SUB3.isin([1]), 'FREQ3'] = 0
train_df.ix[train_df.SUB3.isin([1]), 'FRSTUSE3'] = 0
train_df.ix[train_df.SUB3.isin([1]), 'ROUTE3'] = 0


df3 = train_df
df3 = df3[(df3 >= 0).all(1)]
print df3.shape
print df3.columns.tolist()


# In[23]:

# for sub2

X_train = df3.drop(['SUB2'], axis=1)
Y_train = df3["SUB2"]

from sklearn.feature_selection import SelectKBest, SelectPercentile
from sklearn.feature_selection import f_classif,chi2
#Selector_f = SelectPercentile(f_classif, percentile=25)
Selector_f = SelectKBest(f_classif, k=10)
Selector_f.fit(X_train,Y_train)

zipped = zip(X_train.columns.tolist(),Selector_f.scores_)
ans = sorted(zipped, key=lambda x: x[1])
for n,s in ans:    
    if 'FLG' in n or '2' in n or '3' in n: 
        pass
    else:
        print 'F-score: %3.2ft for feature %s' % (s,n)
                
#print X_train.columns.tolist()


# In[24]:

# get the sorted feature list
import sys
for n, s in ans:
    
    if 'FLG' in n or '2' in n or '3' in n: 
        pass
    else:
        sys.stdout.write('\'%s\',' % (n))


# In[20]:

# for dsm

X_train = df3.drop(['DSMCRIT'], axis=1)
Y_train = df3["DSMCRIT"]

from sklearn.feature_selection import SelectKBest, SelectPercentile
from sklearn.feature_selection import f_classif,chi2
#Selector_f = SelectPercentile(f_classif, percentile=25)
Selector_f = SelectKBest(f_classif, k=10)
Selector_f.fit(X_train,Y_train)

zipped = zip(X_train.columns.tolist(),Selector_f.scores_)
ans = sorted(zipped, key=lambda x: x[1])
for n,s in ans:
     print 'F-score: %3.2ft for feature %s' % (s,n)
                
#print X_train.columns.tolist()


# In[21]:

# get the sorted feature list
import sys
for n, s in ans:
    sys.stdout.write('\'%s\',' % (n))


# In[14]:

from sklearn.model_selection import train_test_split
train, validate = train_test_split(df3, test_size=0.33, random_state=42)
train.shape, validate.shape, train.head(2)


# In[15]:

#df3.to_csv('../trainval.csv', index=False)

train.to_csv('../train_allcols.csv', index=False)
validate.to_csv('../validate_allcols.csv', index=False)

#train.to_csv('../train_abuse.csv', index=False)
#validate.to_csv('../validate_abuse.csv', index=False)
print 'done'

