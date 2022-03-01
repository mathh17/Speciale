#%%
import os
import numpy as np
import pandas as pd
import Holidays_calc as hc
from matplotlib import pyplot as plt
import seaborn as sns

#%%
# read the files from the datafolder containing data fra DK2
# changing the path to the datafolder
path = r'C:\Users\oeste\OneDrive\Uni\Speciale\Scripts\Data\stations_data_dk2'

os.chdir(path)

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
temp_conc_data['mean'] = temp_conc_data.iloc[:,1:12].sum(axis=1) / 11
radi_conc_data['mean'] = radi_conc_data.iloc[:,1:7].sum(axis=1) / 6
dk2_mean = pd.DataFrame()
dk2_mean['time'] = temp_conc_data['time']
dk2_mean['temp_mean_past1h'] = temp_conc_data['mean']
dk2_mean['radia_glob_past1h'] = radi_conc_data['mean']
dk2_mean.head()

# Read Enernginet Pickle Data
# Change back path
old_path = r'C:\Users\oeste\OneDrive\Uni\Speciale\Scripts'
os.chdir(old_path)
df_DK1_2010_2015 = pd.read_pickle("data/dk1_data_2010_2015.pkl")
df_DK2_2010_2015 = pd.read_pickle("data/dk2_data_2010_2015.pkl")
df_DK1_2015_2020 = pd.read_pickle("data/dk1_data_2015_2020.pkl")
df_DK2_2015_2020 = pd.read_pickle("data/dk2_data_2015_2020.pkl")
df_DK1_2020_2022 = pd.read_pickle("data/dk1_data_2020_2022.pkl")
df_DK2_2020_2022 = pd.read_pickle("data/dk2_data_2020_2022.pkl")
df_DK1 = pd.concat([df_DK1_2010_2015,df_DK1_2015_2020,df_DK1_2020_2022], ignore_index=True)
df_DK2 = pd.concat([df_DK2_2010_2015,df_DK2_2015_2020,df_DK2_2020_2022], ignore_index=True)

#Merge data into one DF, on the hour of observations
dk2_mean['time'] = pd.to_datetime(dk2_mean['time'],format='%Y-%m-%dT%H:%M:%S', utc=True)
df_DK2['HourUTC'] = pd.to_datetime(df_DK2['HourUTC'],format='%Y-%m-%dT%H:%M:%S', utc=True)
df_DK2 = df_DK2.rename(columns={'HourUTC':'time', 'HourlySettledConsumption':'Con'})
conc_data = pd.merge(dk2_mean, df_DK2, on='time', how='outer')
conc_data.dropna(inplace=True)
conc_data = conc_data.iloc[::-1]
conc_data = conc_data.sort_values(['time'])

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

#Take data from the concatenated dataset and put it into label data and train data
pred_data = pd.DataFrame(conc_data[['temp_mean_past1h','radia_glob_past1h']])

conc_data = data_encoder(conc_data)
pred_data['is_holiday'] = conc_data['is_holiday']
conc_data['day'] = conc_data['time'].dt.dayofweek
conc_data['time'] = conc_data['time'].dt.hour
conc_data['grad_dage'] = -(conc_data['temp_mean_past1h'])+17
conc_data.loc[conc_data['grad_dage'] <=0, 'grad_dage'] = 0
cat_time = pd.get_dummies(conc_data['time'])
pred_data = pred_data.join(cat_time)
values = conc_data['Con']

hours_in_year = 8760

# %%
#del conc_data['is_holiday']
sns.pairplot(conc_data, diag_kind="hist")
# %%
"""
Plotting the temperature and the consumption in two graphs next to eachother.
"""
y1 = conc_data['temp_mean_past1h']
y2 = conc_data['Con']
x = range(0,len(y1))
fig, (ax1,ax2) = plt.subplots(2,1,figsize=(30,10))
ax1.plot(x, y1, label='temp')  # Plot some data on the axes.
ax1.set_xlabel('x label')  # Add an x-label to the axes.
ax1.set_ylabel('y label')  # Add a y-label to the axes.
ax1.legend();  # Add a legend.

ax2.plot(x, y2, label='consumption')  # Plot more data on the axes...
ax2.set_xlabel('x label')  # Add an x-label to the axes.
ax2.set_ylabel('y label')  # Add a y-label to the axes.
ax2.legend();  # Add a legend.
ax1.set_title("temperature vs consumption")  # Add a title to the axes.


# %%
"""
Plotting the Radiation and the consumption in two graphs next to eachother.
"""
y1 = conc_data['radia_glob_past1h']
y2 = conc_data['Con']
x = range(0,len(y1))
fig, (ax1,ax2) = plt.subplots(2,1,figsize=(100,10))
ax1.plot(x[0:10000], y1[0:10000], label='radiation')  # Plot some data on the axes.
ax1.set_xlabel('x label')  # Add an x-label to the axes.
ax1.set_ylabel('y label')  # Add a y-label to the axes.
ax1.legend();  # Add a legend.

ax2.plot(x[0:10000], y2[0:10000], label='consumption')  # Plot more data on the axes...
ax2.set_xlabel('x label')  # Add an x-label to the axes.
ax2.set_ylabel('y label')  # Add a y-label to the axes.
ax2.legend();  # Add a legend.
ax1.set_title("radiation vs consumption")  # Add a title to the axes.

#%%
"""
Plotting the Time of day and the consumption in two graphs next to eachother.
This is only for 1000 hours, else it is not possible to see anything
"""
y1 = conc_data['time']
y2 = conc_data['Con']
x = range(0,len(y1))
fig, (ax1,ax2) = plt.subplots(2,1,figsize=(30,10))
ax1.plot(x[0:1000], y1[0:1000], label='time')  # Plot some data on the axes.
ax1.set_xlabel('x label')  # Add an x-label to the axes.
ax1.set_ylabel('y label')  # Add a y-label to the axes.
ax1.legend();  # Add a legend.

ax2.plot(x[0:1000], y2[0:1000], label='consumption')  # Plot more data on the axes...
ax2.set_xlabel('x label')  # Add an x-label to the axes.
ax2.set_ylabel('y label')  # Add a y-label to the axes.
ax2.legend();  # Add a legend.
ax1.set_title("time of day vs consumption")  # Add a title to the axes.

#%%
"""
Plotting the Time of day and the consumption in two graphs next to eachother.
This is only for 1000 hours, else it is not possible to see anything
"""
y1 = conc_data['is_holiday']
y2 = conc_data['Con']
x = range(0,len(y1))
fig, (ax1,ax2) = plt.subplots(2,1,figsize=(30,10))
ax1.plot(x[0:1000], y1[1000:2000], label='holiday or not')  # Plot some data on the axes.
ax1.set_xlabel('x label')  # Add an x-label to the axes.
ax1.set_ylabel('y label')  # Add a y-label to the axes.
ax1.legend();  # Add a legend.

ax2.plot(x[0:1000], y2[1000:2000], label='consumption')  # Plot more data on the axes...
ax2.set_xlabel('x label')  # Add an x-label to the axes.
ax2.set_ylabel('y label')  # Add a y-label to the axes.
ax2.legend();  # Add a legend.
ax1.set_title("holiday or not vs consumption")  # Add a title to the axes.
# %%
"""
Plotting the temperature calculate as "graddage" and the consumption in two graphs next to eachother.
"""
y1 = conc_data['grad_dage']
y2 = conc_data['Con']
x = range(0,len(y1))
fig, (ax1,ax2) = plt.subplots(2,1,figsize=(30,10))
ax1.plot(x[hours_in_year:hours_in_year*2], y1[hours_in_year*3:hours_in_year*4], label='temp')  # Plot some data on the axes.
ax1.set_xlabel('x label')  # Add an x-label to the axes.
ax1.set_ylabel('y label')  # Add a y-label to the axes.
ax1.legend();  # Add a legend.

ax2.plot(x[hours_in_year:hours_in_year*2], y2[hours_in_year*3:hours_in_year*4], label='consumption')  # Plot more data on the axes...
ax2.set_xlabel('x label')  # Add an x-label to the axes.
ax2.set_ylabel('y label')  # Add a y-label to the axes.
ax2.legend();  # Add a legend.
ax1.set_title("Graddage vs consumption")  # Add a title to the axes.

#%%
"""
Plotting the day of the week and the consumption in two graphs next to eachother.
This is only for 1000 hours, else it is not possible to see anything
"""
y1 = conc_data['day']
y2 = conc_data['Con']
x = range(0,len(y1))
fig, (ax1,ax2) = plt.subplots(2,1,figsize=(30,10))
ax1.plot(x[0:1000], y1[1000:2000], label='Day of the week')  # Plot some data on the axes.
ax1.set_xlabel('x label')  # Add an x-label to the axes.
ax1.set_ylabel('y label')  # Add a y-label to the axes.
ax1.legend();  # Add a legend.

ax2.plot(x[0:1000], y2[1000:2000], label='consumption')  # Plot more data on the axes...
ax2.set_xlabel('x label')  # Add an x-label to the axes.
ax2.set_ylabel('y label')  # Add a y-label to the axes.
ax2.legend();  # Add a legend.
ax1.set_title("Day of the week vs consumption")  # Add a title to the axes.
#%%
