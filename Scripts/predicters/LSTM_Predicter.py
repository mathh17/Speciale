#%%
#Load all the packages
import os
import numpy as np
import pandas as pd
import Holidays_calc as hc
import tensorflow as tf
from matplotlib import pyplot as plt
from sklearn.metrics import mean_absolute_error
from tensorflow import keras
from tensorflow.keras import layers
from sklearn import preprocessing

#%%
#Select the grid company you wis to predict on:
grid_company = "Radius"

if grid_company == "Fyn":
    path = r'C:\Users\oeste\OneDrive\Uni\DS_3_semester\VI_Projekt\Scripts\data\stations_data_dk1'
    df_el_data = pd.read_pickle("data/Midtfyn_el_data.pkl")
    el_data_2021 = pd.read_pickle("data/Midtfyn_el_data_2021.pkl")
    loaded_model = keras.models.load_model('LSTM_250Epochs_fyn.h5')
if grid_company == "Frederikshavn":
    path = r'C:\Users\oeste\OneDrive\Uni\DS_3_semester\VI_Projekt\Scripts\data\stations_data_dk1'
    df_el_data = pd.read_pickle("data/Frederikshavn_el_data.pkl")
    el_data_2021 = pd.read_pickle("data/Frederikshavn_el_data_2021.pkl")
    loaded_model = keras.models.load_model('LSTM_250Epochs_frederikshavn.h5')
if grid_company == "Tarm":
    path = r'C:\Users\oeste\OneDrive\Uni\DS_3_semester\VI_Projekt\Scripts\data\stations_data_dk1'
    df_el_data = pd.read_pickle("data/Tarm_el_data.pkl")
    el_data_2021 = pd.read_pickle("data/Tarm_el_data_2021.pkl")
    loaded_model = keras.models.load_model('LSTM_250Epochs_tarm.h5')
if grid_company == "Radius":
    path = r'C:\Users\oeste\OneDrive\Uni\DS_3_semester\VI_Projekt\Scripts\data\stations_data_dk2'
    df_el_data = pd.read_pickle("data/jaegerspris_el_data.pkl")
    el_data_2021 = pd.read_pickle("data/jaegerspris_el_data_2021.pkl")
    loaded_model = keras.models.load_model('LSTM_250Epochs_Jaegerspris.h5')  

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

#Take data from the concatenated dataset and put it into label data and train data
pred_data = pd.DataFrame(conc_data[['temp_mean_past1h','radia_glob_past1h']])
conc_data = data_encoder(conc_data)
pred_data['is_holiday'] = conc_data['is_holiday']
conc_data['time'] = conc_data['time'].dt.hour
cat_time = pd.get_dummies(conc_data['time'])
pred_data = pred_data.join(cat_time)

#Normalize the data so its between 0-1, in this instance i just divided it by the max value of the columns
scaler = preprocessing.MinMaxScaler()
con_scaler = preprocessing.MinMaxScaler()
con_data = np.array(conc_data['Con'])
con_data = con_data.reshape(-1,1)
con_scaled = pd.DataFrame(con_scaler.fit_transform(con_data), columns=['Con'], index=conc_data.index)
pred_data['temp_mean_past1h'] = conc_data['temp_mean_past1h']
pred_data['radia_glob_past1h'] = conc_data['radia_glob_past1h']
scaler.fit(pred_data)
pred_data_scaled = pd.DataFrame(scaler.transform(pred_data),columns=pred_data.columns, index=pred_data.index)
pred_data_scaled['Con'] = con_scaled['Con']
#%%
def create_dataset(df, n_deterministic_features,
                   window_size, forecast_size,
                   batch_size):
    # Feel free to play with shuffle buffer size
    shuffle_buffer_size = len(df)
    # Total size of window is given by the number of steps to be considered
    # before prediction time + steps that we want to forecast
    total_size = window_size + forecast_size

    data = tf.data.Dataset.from_tensor_slices(df.values)

    # Selecting windows
    data = data.window(total_size, shift=1, drop_remainder=True)
    data = data.flat_map(lambda k: k.batch(total_size))

    # Shuffling data (seed=Answer to the Ultimate Question of Life, the Universe, and Everything)
    data = data.shuffle(shuffle_buffer_size, seed=42)

    # Extracting past features + deterministic future + labels
    data = data.map(lambda k: ((k[:-forecast_size],
                                k[-forecast_size:, 0:n_deterministic_features]),
                               k[-forecast_size:, -1]))

    return data.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)

# Dividing the complete set into train and test
train_size = 25000
val_size = train_size + 4000
test_size = val_size + 6063

X_train = pred_data_scaled[:train_size]
X_test = pred_data_scaled[val_size:test_size]
X_val = pred_data_scaled[train_size:val_size]

X_train_windowed = create_dataset(X_train,27,48,48,32)
X_val_windowed = create_dataset(X_val,27,48,48,32)
X_test_windowed = create_dataset(X_test,27,48,48,1)

# %%
#scores to evaluate how the model performs on the test data
score = loaded_model.evaluate(X_test_windowed,verbose=0)
print('loss value: '+str(score[0]))
print('MSE: '+ str(score[1]))

#%%
windows = 126
test_pred = []
for i, data in enumerate(X_test_windowed.take(windows)):
    (past, future),truth = data
    test_pred.append(loaded_model.predict((past,future)))
#%%
test_predicitions_unload = []
for i in range(0,windows):
  for l in range(0,48):
    test_predicitions_unload.append(test_pred[i][0][l][0])

range_len = windows * 48
test_plot = pd.DataFrame()
test_plot['exact_values'] = X_test['Con'][0:range_len]
test_plot['predicted_values'] = test_predicitions_unload
test_plot = test_plot.reset_index()
fig = plt.figure(figsize=(6, 6))
plt.subplot(1, 1, 1)
plt.title('Naive vs Exact values')
plt.plot(np.arange(0,range_len), test_plot['exact_values'], 'r-',
         label='Exact values')
plt.plot(np.arange(0,range_len), test_plot['predicted_values'], 'b-',
         label='Precited Values')
plt.legend(loc='upper right')
#plt.xlabel('Boosting Iterations')
plt.ylabel('Consumption')
fig.tight_layout()
plt.show()

#%%
test_pred_rescaled = con_scaler.inverse_transform(pd.DataFrame(test_predicitions_unload))
# %%
#Calculates the MAE if you just use the data from 24hours before at prediction.
naive_y_val = np.roll(con_data[29000:35048],48)
mean_absolute_error(con_data[29000:35048],naive_y_val)
#%%
#Calculates the MAE for the predicted values from the validation set.
mean_absolute_error(con_data[29000:35048],test_pred_rescaled)
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
Henter forecast data fra stationen
Jægersborg tilhører grid companiet Radius Elnet. 
Fører det sammen i et datasæt og omregner temperaturen fra Kelvin Celsius
"""
forecast_data = pd.read_parquet("data/forecast_data")
data_temp_val = get_station_temp_val(forecast_data)
data_radi_val = get_station_radi_val(forecast_data)

data_temp_val = data_encoder(data_temp_val)
data_radi_val = data_encoder(data_radi_val)

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
forecast_real_val = stations_concat_df['Con']
forecast_con = pd.DataFrame(stations_concat_df['Con'])
forecast_con = pd.DataFrame(con_scaler.transform(forecast_con))
stations_concat_df = stations_concat_df.reindex(columns=['temp_mean_1hr',	'radiation_hour',	'is_holiday',	0,	1,	2,	3,	4,	5,	6,	7,	8,	9,	10,	11,	12,	13,	14,	15,	16,	17,	18,	19,	20,	21,	22,	23,	'Con'])
stations_concat_df = stations_concat_df.drop(columns=['Con'])
stations_scaled = pd.DataFrame(scaler.transform(stations_concat_df),columns=stations_concat_df.columns, index=stations_concat_df.index)
stations_scaled['Con'] = forecast_con
#%%
forecast_windowed_in = create_dataset(stations_scaled,27,48,48,1)
forecast_windowed_in

#%%
forecast_window = 58
forecast_pred = []
for i, data in enumerate(forecast_windowed_in.take(forecast_window)):
    (past, future),truth = data
    forecast_pred.append(loaded_model.predict((past,future)))

# %%
predicitions_unload = []
for i in range(0,forecast_window):
  for l in range(0,48):
    predicitions_unload.append(forecast_pred[i][0][l][0])

range_len = forecast_window * 48
test_plot = pd.DataFrame()
test_plot['exact_values'] = stations_scaled['Con'][0:range_len]
test_plot['predicted_values'] = predicitions_unload
test_plot = test_plot.reset_index()
fig = plt.figure(figsize=(6, 6))
plt.subplot(1, 1, 1)
plt.title('Predicts vs Exact values')
plt.plot(np.arange(0,range_len), test_plot['exact_values'], 'r-',
         label='Exact values')
plt.plot(np.arange(0,range_len), test_plot['predicted_values'], 'b-',
         label='Precited Values')
plt.legend(loc='upper right')
plt.ylabel('Consumption')
fig.tight_layout()
plt.show()

#%%
forecast_pred_rescaled = con_scaler.inverse_transform(pd.DataFrame(predicitions_unload))

# %%
naive_y_val_forecast = np.roll(forecast_real_val[0:2784],48)
mae_forecast_naive = mean_absolute_error(forecast_real_val[0:2784],naive_y_val_forecast)
mae_forecast_naive
# %%
mae_forecast = mean_absolute_error(forecast_real_val[0:2784],forecast_pred_rescaled)
mae_forecast

# %%
