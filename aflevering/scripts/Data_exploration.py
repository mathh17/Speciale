#%%
from cProfile import label
import os
import numpy as np
import pandas as pd
import Holidays_calc as hc
from matplotlib import pyplot as plt
from matplotlib import dates
#import seaborn as sns
from sklearn import metrics


#%%
# read the files from the datafolder containing data fra DK2
# changing the path to the datafolder
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

# Read Enernginet Pickle Data
# Change back path
# OBS THIS WILL HAVE TO BE CHANGED TO THE LOCATION YOU HAVE STORED THE WEATHER DATA IN ORDER FOR THE CODE TO RUN
home_path = r'PATH GOES HERE'

os.chdir(home_path)
df_DK2 = pd.read_parquet("Data/el_data_2010-2020_dk2")

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
#%%
def get_forecast_merged(pred_ahead):
    forecast_df = pd.read_parquet("Data/dk2_forecast_data")
    forecast_df = forecast_df.rename(columns={'Date':'time'})
    pred_counter = 1
    forecast_dict = {}
    while pred_counter < 49:
        pred_df = forecast_df.groupby('predicted_ahead')
        
        forecast_dict[pred_counter] = pred_df.get_group(int(pred_counter))
        pred_counter +=1
    forecast_dict[pred_ahead]['mean_temp'] = forecast_dict[pred_ahead]['mean_temp'] - 272.15
    forecast_merge = pd.merge(dk2_mean,forecast_dict[pred_ahead],how='inner', on='time')
    forecast_merge = forecast_merge.sort_values(by='time')
    return forecast_merge

#%%
forecast_merge = get_forecast_merged(48)


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
ax1.plot(x[0:hours_in_year], y1[0:hours_in_year], label='Temperature')  # Plot some data on the axes.
ax1.set_xlabel('Hours')  # Add an x-label to the axes.
ax1.set_ylabel('Temperature')  # Add a y-label to the axes.
ax1.legend();  # Add a legend.

ax2.plot(x[0:hours_in_year], y2[0:hours_in_year], label='Consumption')  # Plot more data on the axes...
ax2.set_xlabel('Hours')  # Add an x-label to the axes.
ax2.set_ylabel('Consumption')  # Add a y-label to the axes.
ax2.legend();  # Add a legend.
ax1.set_title("Temperature & Consumption")  # Add a title to the axes.


# %%
"""
Plotting the Radiation and the consumption in two graphs next to eachother.
"""
y1 = conc_data['radia_glob_past1h'][79200:79500]
y2 = conc_data['Con'][79200:79500]
x = range(0,len(y1))
fig, (ax1,ax2) = plt.subplots(2,1,figsize=(10,6))
ax1.plot(x[0:len(y1)], y1[0:len(y1)], label='Radiation')  # Plot some data on the axes.
ax1.set_ylabel('radiation', fontsize=16)  # Add a y-label to the axes.


ax2.plot(x[0:len(y1)], y2[0:len(y1)], label='Consumption')  # Plot more data on the axes...
ax2.set_xlabel('Hours', fontsize=16)  # Add an x-label to the axes.
ax2.set_ylabel('Consumption', fontsize=16)  # Add a y-label to the axes.
ax1.set_title("Radiation & Consumption", fontsize=16)  # Add a title to the axes.

#%%
"""
Plotting the Time of day and the consumption in two graphs next to eachother.
This is only for 1000 hours, else it is not possible to see anything
"""
days = ['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17']
y1 = conc_data['time']
y2 = conc_data['Con']
x = range(0,len(y1))
fig = plt.figure(figsize=(15, 6))
ax = fig.add_subplot()
ax.plot(x[100:200], y2[100:200], label='Consumption')  # Plot more data on the axes...
ax.set_xlabel('Days', fontsize=20)  # Add an x-label to the axes.
ax.set_ylabel('Consumption', fontsize=20)  # Add a y-label to the axes.
ax.set_xticklabels(days)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
ax.set_xticks(range(100,200,24))
ax.set_title("Consumption with X-axis set to days", fontsize=20)  

#%%
"""
Plotting the Time of day and the consumption in two graphs next to eachother.
This is only for 1000 hours, else it is not possible to see anything
"""
y1 = conc_data['is_holiday']
y2 = conc_data['Con']
x = range(0,len(y1))
fig, (ax1,ax2) = plt.subplots(2,1,figsize=(10,6))
ax1.plot(x[2000:2500], y1[2000:2500], label='Holiday or not')  # Plot some data on the axes.
#ax1.set_xlabel('Hours', fontsize=16)  # Add an x-label to the axes.
ax1.set_ylabel('Holiday or not', fontsize=16)  # Add a y-label to the axes.
#ax1.set_xticks(range(2000,2500,24))

ax2.plot(x[2000:2500], y2[2000:2500], label='Consumption')  # Plot more data on the axes...
ax2.set_xlabel('Hours', fontsize=16)  # Add an x-label to the axes.
ax2.set_ylabel('Consumption', fontsize=16)  # Add a y-label to the axes.

ax1.set_title("Holiday or Not & Consumption", fontsize=16)  # Add a title to the axes.
# %%
"""
Plotting the temperature calculate as "graddage" and the consumption in two graphs next to eachother.
"""
y1 = conc_data['grad_dage']
y2 = conc_data['Con']
z = np.polyfit(y1, y2, 1)
p = np.poly1d(z)
x = range(0,len(y1))
fig, ax = plt.subplots(figsize=(6,6))
ax.plot(y1,p(y1),"b")
ax.scatter(y1[0:20000], y2[0:20000], 1)  # Plot some data on the axes.
ax.set_xlabel('Degree day')  # Add an x-label to the axes.
ax.set_ylabel('Consumption')  # Add a y-label to the axes.
ax.set_title("Degree-Days vs Consumption")  # Add a title to the axes.

plt.show()

"""
ax2.plot(x[0:len(y1)], y2[0:len(y1)], label='Consumption')  # Plot more data on the axes...
ax2.set_xlabel('Hours')  # Add an x-label to the axes.
ax2.set_ylabel('Consumption')  # Add a y-label to the axes.
ax2.legend();  # Add a legend.
"""
#%%
"""
Plotting the day of the week and the consumption in two graphs next to eachother.
This is only for 1000 hours, else it is not possible to see anything
"""
y1 = conc_data['day']
y2 = conc_data['Con']
x = range(0,len(y1))
fig, (ax1,ax2) = plt.subplots(2,1,figsize=(20,10))
ax1.plot(x[0:1000], y1[1000:2000], label='Day of the week')  # Plot some data on the axes.
ax1.set_xlabel('Hours')  # Add an x-label to the axes.
ax1.set_ylabel('Day of the week')  # Add a y-label to the axes.
ax1.legend();  # Add a legend.

ax2.plot(x[0:1000], y2[1000:2000], label='consumption')  # Plot more data on the axes...
ax2.set_xlabel('Hours')  # Add an x-label to the axes.
ax2.set_ylabel('Consumption')  # Add a y-label to the axes.
ax2.legend();  # Add a legend.
ax1.set_title("Day of the week vs consumption")  # Add a title to the axes.
#%%
"""

"""
months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

y2 = forecast_merge[:-1]['mean_temp']
y1 = forecast_merge[:-1]['temp_mean_past1h']
x = range(0,len(y1))
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot()
ax.plot(x, y1, color='tab:blue', label='Observed values')
ax.plot(x, y2, color='tab:orange', label='Forecast values')
ax.set_ylabel('Temperature C',fontsize=16)
ax.set_xticks(range(0,2914,250))
ax.set_xticklabels(months)
ax.set_xlabel('Months',fontsize=16)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
ax.set_title('Observed and forecast temperatures',fontsize=24)
ax.legend(fontsize=16)

#%%
"""

"""

y2 = forecast_merge[:-1]['mean_radi']
y1 = forecast_merge[:-1]['radia_glob_past1h']
x = range(0,len(y1))
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot()
ax.plot(x, y1, color='tab:blue', label='Observed values')
ax.plot(x, y2, color='tab:orange', label='Forecast values')
ax.set_ylabel('Radiation W/m^2',fontsize=16)
ax.set_xticks(range(0,2914,250))
ax.set_xticklabels(months)
ax.set_xlabel('Months',fontsize=16)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
ax.set_title('Observed and forecast radiation levels', fontsize=24)
ax.legend(fontsize=16)
# %%
print(metrics.r2_score(y1,y2))
#%%
