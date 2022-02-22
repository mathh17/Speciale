
#%%
# Importing necessary packages
import os
import pandas as pd
#%%
path = r'C:\Users\MTG\OneDrive - Energinet.dk\Skrivebord\VI_projekt\Scripts\data\stations_data_dk2'

os.chdir(path)

temp_conc_data = pd.DataFrame(columns=['time'])
radi_conc_data = pd.DataFrame(columns=['time'])

iterations = 0
for file in os.listdir():
    df = pd.read_pickle(file)
    file_name = os.path.basename(file)
    if 'temp_mean_past1h' in df.columns:
        temp_conc_data = pd.merge(temp_conc_data,df[['time','temp_mean_past1h']],on='time',how='outer', suffixes=(['old','_{}'.format(file_name)]))
    if 'radia_glob_past1h' in df.columns:
        radi_conc_data = pd.merge(radi_conc_data,df[['time','radia_glob_past1h']],on='time',how='outer', suffixes=(['old','_{}'.format(file_name)]))
    iterations += 1



# %%
#temp_no_time = temp_conc_data
temp_conc_data['mean'] = temp_conc_data.iloc[:,1:12].sum(axis=1) / 11
radi_conc_data['mean'] = radi_conc_data.iloc[:,1:7].sum(axis=1) / 6
dk2_mean = pd.DataFrame()
dk2_mean['time'] = temp_conc_data['time']
dk2_mean['temp_mean_past1h'] = temp_conc_data['mean']
dk2_mean['radia_glob_past1h'] = radi_conc_data['mean']
dk2_mean.head()
# %%
