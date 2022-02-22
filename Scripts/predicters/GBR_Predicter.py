#%%
#Load all the packages
import os
import numpy as np
import pandas as pd
import Holidays_calc as hc
from joblib import dump, load
from matplotlib import pyplot as plt
from sklearn.metrics import mean_absolute_error

#%%
#Select the grid company you wis to predict on:

grid_company = "Fyn"

if grid_company == "Fyn":
    path = r'C:\Users\oeste\OneDrive\Uni\DS_3_semester\VI_Projekt\Scripts\data\stations_data_dk1'
    df_el_data = pd.read_pickle("data/Midtfyn_el_data.pkl")
    el_data_2021 = pd.read_pickle("data/Midtfyn_el_data_2021.pkl")
    forecast_data = pd.read_pickle("data/forecast_data_fyn.pkl")
    model_name = "fyn_gbr.joblib"
if grid_company == "Frederikshavn":
    path = r'C:\Users\oeste\OneDrive\Uni\DS_3_semester\VI_Projekt\Scripts\data\stations_data_dk1'
    df_el_data = pd.read_pickle("data/Frederikshavn_el_data.pkl")
    el_data_2021 = pd.read_pickle("data/Frederikshavn_el_data_2021.pkl")
    forecast_data = pd.read_pickle("data/forecast_data_Frederikshavn.pkl")
    model_name = "frederikshavn_gbr.joblib"
if grid_company == "Tarm":
    path = r'C:\Users\oeste\OneDrive\Uni\DS_3_semester\VI_Projekt\Scripts\data\stations_data_dk1'
    df_el_data = pd.read_pickle("data/Tarm_el_data.pkl")
    el_data_2021 = pd.read_pickle("data/Tarm_el_data_2021.pkl")
    forecast_data = pd.read_pickle("data/forecast_data_Tarm.pkl")
    model_name = "tarm_gbr.joblib" 
if grid_company == "Radius":
    path = r'C:\Users\oeste\OneDrive\Uni\DS_3_semester\VI_Projekt\Scripts\data\stations_data_dk2'
    df_el_data = pd.read_pickle("data/jaegerspris_el_data.pkl")
    el_data_2021 = pd.read_pickle("data/jaegerspris_el_data_2021.pkl")
    
    model_name = "Radius_gbr.joblib"   

#%%
# read the files from the datafolder containing data fra DK2
# changing the path to the datafolder
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
old_path = r'C:\Users\oeste\OneDrive\Uni\DS_3_semester\VI_Projekt\Scripts'
os.chdir(old_path)

#Merge data into one DF, on the hour of observations
dk2_mean['time'] = pd.to_datetime(dk2_mean['time'],format='%Y-%m-%dT%H:%M:%S', utc=True)
df_el_data['HourUTC'] = pd.to_datetime(df_el_data['HourUTC'],format='%Y-%m-%dT%H:%M:%S', utc=True)
df_el_data = df_el_data.rename(columns={'HourUTC':'time', 'HourlySettledConsumption':'Con'})
conc_data = pd.merge(dk2_mean, df_el_data, on='time', how='outer')
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
conc_data['time'] = conc_data['time'].dt.hour
cat_time = pd.get_dummies(conc_data['time'])
pred_data = pred_data.join(cat_time)
values = conc_data['Con']
#%%
# Dividing the complete set into train and test
train_size = 25000
val_size = train_size + 4000
test_size = val_size + 6063

X_train = pred_data[:train_size]
X_test = pred_data[val_size:test_size]
X_val = pred_data[train_size:val_size]
y_train = values[:train_size]
y_test = values[val_size:test_size]
y_val = values[train_size:val_size]

#%%
loaded_model = load(model_name)

#%%
y_test_hat = loaded_model.predict(X_test)
test_acc = loaded_model.score(X_test,y_test)
test_acc
# %%
#Calculates the MAE if you just use the data from 24hours before at prediction.
naive_y_val = np.roll(y_test.to_numpy(),24)
mean_absolute_error(y_test,naive_y_val)
#%%
#Calculates the MAE for the predicted values from the validation set.
mean_absolute_error(y_test,y_test_hat)
# %%
# Plots the predicted values with the exact values to compare how the model predicts.
y_plot = pd.DataFrame()
y_plot['exact_values'] = y_test
y_plot['predicted_values'] = y_test_hat
y_plot = y_plot.sort_index()
fig = plt.figure(figsize=(6, 6))
plt.subplot(1, 1, 1)
plt.title('Predicts vs Exact values')
plt.plot(range(len(y_plot)), y_plot['exact_values'], 'r-',
         label='Exact values')
plt.plot(range(len(y_plot)), y_plot['predicted_values'], 'b-',
         label='Precited Values')
plt.legend(loc='upper right')
#plt.xlabel('Boosting Iterations')
plt.ylabel('Consumption')
fig.tight_layout()
plt.show()

#%%

def get_station_temp_val(station):
    values = []
    time = []
    predicted = []
    counter = 0
    for index, row  in station.iterrows():
        if row['weather_type'] == 'temperatur_2m' and row['predicted_ahead']  == counter%49:
            values.append(row['value'])
            time.append(row['Date'])
            predicted.append(row['predicted_ahead'])
            counter+=1
        
    stations_df = pd.DataFrame(columns=['temp_mean_1hr','predicted_ahead','time'])
    stations_df['temp_mean_1hr'] = values
    stations_df['time'] = time
    stations_df['predicted_ahead'] = predicted
    return stations_df

def get_station_radi_val(station):
    values = []
    time = []
    predicted = []
    counter = 0
    for index, row  in station.iterrows():
        if row['weather_type'] == 'radiation_hour' and row['predicted_ahead'] == counter%49:
            values.append(row['value'])
            time.append(row['Date'])
            predicted.append(row['predicted_ahead'])
            counter+=1
    stations_df = pd.DataFrame(columns=['radiation_hour','predicted_ahead','time'])
    stations_df['radiation_hour'] = values
    stations_df['time'] = time
    stations_df['predicted_ahead'] = predicted
    return stations_df

"""
Henter forecast data fra stationen: Jægersborg.
Jægersborg tilhører grid companiet Radius Elnet. 
Fører det sammen i et datasæt og omregner temperaturen fra Kelvin Celsius
"""
forecast_data = pd.read_parquet("data/forecast_data")
data_temp_val = get_station_temp_val(forecast_data)
data_radi_val = get_station_radi_val(forecast_data)

data_temp_val = data_encoder(data_temp_val)
data_radi_val = data_encoder(data_radi_val)

pred_con = pd.DataFrame()
df_DK2_maj_con = pd.DataFrame()
df_DK2_maj_con['time'] = el_data_2021['HourUTC']
df_DK2_maj_con['Con'] = el_data_2021['HourlySettledConsumption']
df_DK2_maj_con['time'] = pd.to_datetime(df_DK2_maj_con['time'],format='%Y-%m-%dT%H:%M:%S', utc=True)
data_temp_val = pd.merge(df_DK2_maj_con,data_temp_val, on='time', how='outer')
data_temp_val

data_temp_val['time'] = data_temp_val['time'].dt.hour
data_radi_val['time'] = data_radi_val['time'].dt.hour

radi_val = data_radi_val['radiation_hour']
stations_concat_df = data_temp_val.join(radi_val)
stations_concat_df['temp_mean_1hr'] = stations_concat_df['temp_mean_1hr'].add(-273.15)

cat_time = pd.get_dummies(stations_concat_df['time'])
stations_concat_df = stations_concat_df.join(cat_time)
stations_concat_df = stations_concat_df.drop(columns=['predicted_ahead','time'])
stations_concat_df.dropna(inplace=True)
stations_concat_df = stations_concat_df.reindex(columns=['temp_mean_1hr',	'radiation_hour',	'is_holiday',	0,	1,	2,	3,	4,	5,	6,	7,	8,	9,	10,	11,	12,	13,	14,	15,	16,	17,	18,	19,	20,	21,	22,	23,	'Con'])
pred_con = stations_concat_df['Con']
stations_concat_df = stations_concat_df.drop(columns=['Con'])
# %%
preds = loaded_model.predict(stations_concat_df)
# %%
# Plots the predicted values with the exact values to compare how the model predicts.
y_plot = pd.DataFrame()
y_plot['exact_values'] = pred_con
y_plot['predicted_values'] = preds
y_plot = y_plot.sort_index()
fig = plt.figure(figsize=(6, 6))
plt.subplot(1, 1, 1)
plt.title('Predicts vs Exact values')
plt.plot(range(len(y_plot)), y_plot['exact_values'], 'r-',
         lw=True,label='Exact values')
plt.plot(range(len(y_plot)), y_plot['predicted_values'], 'b-',
         aa=True,label='Precited Values')
plt.legend(loc='upper right')
plt.ylabel('Consumption')
plt.show()
# %%
#Calculates the MAE if you just use the data from 24hours before at prediction.
naive_y_val = np.roll(y_plot['exact_values'].to_numpy(),48)
naive_mae_forecast = mean_absolute_error(pred_con,naive_y_val)
naive_mae_forecast
#%%
#Calculates the MAE for the predicted values from the validation set.
mae_forecast = mean_absolute_error(pred_con,preds)
mae_forecast
#%%