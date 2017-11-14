
# coding: utf-8

# In[ ]:

import pandas as pd
import glob, os
 
os.chdir("c:/Users/raghav.sharma/Desktop/dmLab")
results = pd.DataFrame([])
 
for counter, file in enumerate(glob.glob("*.csv")):
    namedf = pd.read_csv(file)
    results = results.append(namedf)
 
results.to_csv('c:/Users/Raghav.sharma/Desktop/dmLab/combinedfile2.csv')


