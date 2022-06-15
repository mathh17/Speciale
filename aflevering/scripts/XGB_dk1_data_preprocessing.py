#%%
import os
import numpy as np
import pandas as pd
import Holidays_calc as hc


#%%
#Calling the holiday function to build a column for if its a holiday or not
def holidays(df):
    holidays = []
    for i, row in df.iterrows():
        is_holiday = hc.get_date_type(row['time'])
        holidays.append(is_holiday)
    return holidays
def data_encoder(df): 
    df['time'] = pd.to_datetime(df['time'],format='%Y-%m-%dT%H:%M:%S', utc=True)
    df['is_holiday'] = holidays(df)
    return df

#%%
# read the files from the datafolder containing data fra DK1
# OBS THIS WILL HAVE TO BE CHANGED TO THE LOCATION YOU HAVE STORED THE WEATHER DATA IN ORDER FOR THE CODE TO RUN
home_path = r'PATH GOES HERE'
os.chdir(home_path)

temp_conc_data = pd.DataFrame(columns=['time'])
radi_conc_data = pd.DataFrame(columns=['time'])

# goes through all the files one by one adding them all together to create a Dataframe with one column for each station
for file in os.listdir():
    df = pd.read_pickle(file)
    file_name = os.path.basename(file)
    if 'temp_mean_past1h' in df.columns:
        temp_conc_data = pd.merge(temp_conc_data,df[['time','temp_mean_past1h']],on='time',how='outer', suffixes=(['old','_{}'.format(file_name)]))
    if 'radia_glob_past1h' in df.columns:
        radi_conc_data = pd.merge(radi_conc_data,df[['time','radia_glob_past1h']],on='time',how='outer', suffixes=(['old','_{}'.format(file_name)]))

# takes all the columns and calculates the mean for each row. which gives us a mean value for all stations at the given time.
num_columns_temp = temp_conc_data.shape[1]
num_columns_radi = radi_conc_data.shape[1]
temp_conc_data['mean'] = temp_conc_data.iloc[:,1:num_columns_temp].sum(axis=1) / (num_columns_temp-1)
radi_conc_data['mean'] = radi_conc_data.iloc[:,1:num_columns_radi].sum(axis=1) / (num_columns_radi-1)
dk2_mean = pd.DataFrame()
dk2_mean['time'] = temp_conc_data['time']
dk2_mean['temp_mean_past1h'] = temp_conc_data['mean']
dk2_mean['radia_glob_past1h'] = radi_conc_data['mean']
dk2_mean.head()

# Read Enernginet Pickle Data
# Change back path
# OBS THIS WILL HAVE TO BE CHANGED TO THE LOCATION YOU HAVE STORED THE WEATHER DATA IN ORDER FOR THE CODE TO RUN
home_path = r'PATH GOES HERE'

os.chdir(home_path)
df_DK2 = pd.read_parquet("Data/el_data_2010-2020_dk1")

#Merge data into one DF, on the hour of observations
dk2_mean['time'] = pd.to_datetime(dk2_mean['time'],format='%Y-%m-%dT%H:%M:%S', utc=True)
df_DK2['HourUTC'] = pd.to_datetime(df_DK2['HourUTC'],format='%Y-%m-%dT%H:%M:%S', utc=True)
df_DK2 = df_DK2.rename(columns={'HourUTC':'time', 'HourlySettledConsumption':'Con'})
conc_data = pd.merge(dk2_mean, df_DK2, on='time', how='outer')
conc_data.dropna(inplace=True)
conc_data = conc_data.iloc[::-1]
conc_data = conc_data.sort_values(['time'])

#Take data from the concatenated dataset and put it into label data and train data
Observed_data = pd.DataFrame(conc_data[['temp_mean_past1h','radia_glob_past1h','Con','time']])

conc_data = data_encoder(conc_data)
Observed_data['is_holiday'] = conc_data['is_holiday']
Observed_data['year'] = conc_data['time'].dt.year 
Observed_data['grad_dage'] = -(conc_data['temp_mean_past1h'])+17
Observed_data.loc[Observed_data['grad_dage'] <=0, 'grad_dage'] = 0
Observed_data['hour'] = conc_data['time'].dt.hour
