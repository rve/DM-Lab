# Data Mining Lab - Mental Health Dataset

Data Mining Practical Course

## Getting Started


### Prerequisites


```
jupyter notebook
```

For lab machines  

```
wget link/to/anaconda.sh
bash anaconda.sh
```
Check the lab wiki (week 1) for how to connect `jupyter notebook` remotely.



For Google Clound

```
sudo apt-get install python-pip
sudo apt-get install unzip
```
You may also install `zsh` and `oh-my-zsh` for the auto completion.

### Setting up

Install dependencies
```
pip install -r requirements.txt
```

Download the csv datasets and concat them. 
```
bash utils/download.sh
bash utils/concat.sh 
```
Get the current preprocessed dataset (for Google Cloud):   
go to the folder of most recent week 
```
jupyter nbconvert --to python get_newsplit.ipynb
python get_newsplit.py
```

## Schedule
Week 0: Dataset Preparation   
Week 1-5: Descriptive Mining I-V   
Week 6-9: Predictive Mining I-IV   
Week 10-11: Final Presentation   

## Dataset

* [SAMHDA](http://datafiles.samhsa.gov/study-series/treatment-episode-data-set-admissions-teds-nid13518) - Treatment Episode Data Set: Admissions (TEDS-A)
* [DASIS](https://wwwdasis.samhsa.gov/dasis2/nmhss.htm) - National Mental Health Services Survey (N-MHSS)
* [DASIS](https://wwwdasis.samhsa.gov/dasis2/nssats.htm) - National Survey of Substance Abuse Treatment Services
(N-SSATS)

