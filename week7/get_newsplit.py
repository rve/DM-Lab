
# coding: utf-8

# In[2]:


#import csv from the dataset
import pandas as pd

df=pd.read_csv('../final.csv')

print df.shape


# In[3]:


#for drop_list_suppl , we'll handle the missing values later
drop_list = ['REGION', 'DIVISION', 'PRIMPAY', 'PRIMINC', 'DAYWAIT',
             'HLTHINS', 'CBSA10', 'CASEID',]
drop_list_suppl = ['FREQ2', 'FREQ3', 'FRSTUSE2', 'FRSTUSE3', 'ROUTE2', 'ROUTE3', 
                      ]


#howto deal with: priminc
#remove NOPRIOr freq1 frstuse1 route1 SERVSETA sub3 PSYPROB MARSTAT VET, PSOURCE DETCRIM METHUSE DETNLF IDU PREG
train_df = df.drop(drop_list + drop_list_suppl, axis=1)
train_df['DETCRIM'].replace(to_replace=[-9], value = 0, inplace=True)
train_df['DETNLF'].replace(to_replace=[-9], value = 0, inplace=True)
train_df['IDU'].replace(to_replace=[-9], value = 0, inplace=True)
train_df.ix[train_df.GENDER.isin([1]), 'PREG'] = 2

df3 = train_df
df3 = df3[(df3 >= 0).all(1)]
print df3.shape
print df3.columns.tolist()


# In[8]:


print df3['IDU'].describe()
#print df3.head()


# In[7]:


from sklearn.model_selection import train_test_split
train, validate = train_test_split(df3, test_size=0.33, random_state=42)
train.shape, validate.shape, train.head(2)


# In[9]:



df3.to_csv('../trainval.csv', index=False)
#train.to_csv('../train.csv', index=False)
#validate.to_csv('../validate.csv', index=False)
print 'done'

