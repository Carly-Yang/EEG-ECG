import pandas as pd
import numpy as np
import random

bis_data = pd.read_csv('100_new_bis.csv')
bis = pd.DataFrame()
bis["bis"] = bis_data
eeg_data = pd.read_csv('100_new_eeg.csv')
eeg = pd.DataFrame()
eeg["eeg"] = eeg_data

#delete nan and -1 values
eeg_df = eeg.dropna()
bis_df = bis.dropna()
bis_df=np.array(bis_df)
eeg_df=np.array(eeg_df)
eeg_df = eeg_df[eeg_df > 0]
bis_df = bis_df[bis_df > 0]
bis_df=pd.DataFrame(bis_df)
eeg_df=pd.DataFrame(eeg_df)
e_len=len(eeg_df)
b_len=len(bis_df)

if b_len > e_len:
    k = b_len - e_len    
    x = random.sample(range(b_len),k)
    bis_de = bis_df.drop(x)
    eeg_de = eeg_df
    bis_de=np.array(bis_de)
    bis_de = bis_de[bis_de > 0]
    bis_de=pd.DataFrame(bis_de)
    
elif b_len < e_len:
    k = e_len - b_len
    x = random.sample(range(e_len),k)
    eeg_de = eeg_df.drop(x)
    bis_de = bis_df
    eeg_de=np.array(eeg_de)
    eeg_de = bis_de[eeg_de > 0]
    eeg_de=pd.DataFrame(eeg_de)    
else :
    bis_de = bis_df
    eeg_de = eeg_df

h_con = pd.DataFrame()
h_con["eeg"] = eeg_de
h_con["bis"] = bis_de 
h_con.to_csv("100_step3.csv") 