#%%
from gc import callbacks
import os
import numpy as np
import pandas as pd
import Holidays_calc as hc
from matplotlib import pyplot as plt
import tensorflow as tf
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from tensorflow import keras
from tensorflow.keras import layers
from sklearn import preprocessing
from tensorflow.keras.callbacks import EarlyStopping

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

def create_dataset(df, n_deterministic_features,
                   window_size, forecast_size,
                   batch_size):
    total_size = window_size + forecast_size

    data = tf.data.Dataset.from_tensor_slices(df.values)

    # Selecting windows
    data = data.window(total_size, shift=1, drop_remainder=True)
    data = data.flat_map(lambda k: k.batch(total_size))

    # Extracting past features + deterministic future + labels
    data = data.map(lambda k: ((k[:-forecast_size],
                                k[-forecast_size:, 0:n_deterministic_features]),
                               k[-forecast_size:, -1]))

    return data.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)



def create_forecast_dataset(df_forecast,
                   window_size, forecast_size,
                   batch_size):
    total_size = window_size + forecast_size

    data_forecast = tf.data.Dataset.from_tensor_slices(df_forecast.values)

    # Selecting windows
    data_forecast = data_forecast.window(total_size, shift=1, drop_remainder=True)
    data_forecast = data_forecast.flat_map(lambda k: k.batch(total_size))

    # Extracting past features + deterministic future + labels
    data_forecast = data_forecast.map(lambda k: ((k[:-forecast_size,0:28],
                                k[-forecast_size:, 28:55]),
                               k[-forecast_size:, -1]))

    return data_forecast.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)


def daily_meaner(df):
    df_len = len(df) / 365
    weekly_con = []
    iterations = 1
    while iterations <= 365:
        hours_in_a_week = int(df_len*iterations)
        #one_week_con = 0
        one_week_con = (df[int(hours_in_a_week-df_len):hours_in_a_week].sum())/df_len
        weekly_con.append(one_week_con)
        iterations += 1
    return weekly_con
#%%
# read the files from the datafolder containing data fra DK1
# changing the path to the datafolder
home_path = r'C:\Users\oeste\OneDrive\Uni\Speciale\Scripts\Data\dmi_data_dk1'
EN_path = r'C:\Users\MTG.ENERGINET\OneDrive - Energinet.dk\Dokumenter\Speciale\Scripts\Data\dmi_data_dk1'

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
home_path = r'C:\Users\oeste\OneDrive\Uni\Speciale\Scripts'
EN_path = r'C:\Users\MTG.ENERGINET\OneDrive - Energinet.dk\Dokumenter\Speciale\Scripts'

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
observed_data = pd.DataFrame(conc_data[['temp_mean_past1h','radia_glob_past1h','Con','time']])

conc_data = data_encoder(conc_data)
observed_data['is_holiday'] = conc_data['is_holiday']
observed_data['year'] = conc_data['time'].dt.year 
observed_data['grad_dage'] = -(conc_data['temp_mean_past1h'])+17
observed_data.loc[observed_data['grad_dage'] <=0, 'grad_dage'] = 0
observed_data['hour'] = conc_data['time'].dt.hour
cat_time = pd.get_dummies(observed_data['hour'])
observed_data = observed_data.join(cat_time)
observed_data = observed_data.loc[observed_data['Con'] < 3500]
observed_data = observed_data.loc[observed_data['Con'] > 1100]

con_scaler = preprocessing.MinMaxScaler()
scaler = preprocessing.MinMaxScaler()
con_data = np.array(observed_data['Con'])
con_data = con_data.reshape(-1,1)
con_scaled = pd.DataFrame(con_scaler.fit_transform(con_data), columns=['Con'], index=observed_data.index)
con_scaled['year'] = observed_data['year']
observed_scaled = observed_data
observed_scaled[['grad_dage','radia_glob_past1h','Con']] = scaler.fit_transform(observed_scaled[['grad_dage','radia_glob_past1h','Con']])

"""
Set up training and validation sets.
"""
train_set = observed_scaled.loc[observed_scaled['year'] <= 2017] 
val_set =  observed_scaled.loc[observed_scaled['year'] >= 2018]
y_train = con_scaled.loc[observed_scaled['year'] <= 2017]
y_val = con_scaled.loc[observed_scaled['year'] >= 2018]
y_train = y_train.drop(columns=['year'])
y_val = y_val.drop(columns=['year'])
forecast_val = val_set.rename(columns={'time':'time','grad_dage':'grad_dage_o','radia_glob_past1h':'radia_glob_past1h_o','is_holiday':'is_holiday_o',	0:'0_o',	1:'1_o',	2:'2_o',	3:'3_o',	4:'4_o',	5:'5_o',	6:'6_o',	7:'7_o',	8:'8_o',	9:'9_o',	10:'10_o',	11:'11_o',	12:'12_o',	13:'13_o',	14:'14_o',	15:'15_o',	16:'16_o',	17:'17_o',	18:'18_o',	19:'19_o',	20:'20_o',	21:'21_o',	22:'22_o',	23:'23_o',	'Con':'Con_o'})

forecast_val = forecast_val.reindex(columns=['time','grad_dage_o','radia_glob_past1h_o','is_holiday_o',	'0_o',	'1_o',	'2_o',	'3_o',	'4_o',	'5_o',	'6_o',	'7_o',	'8_o',	'9_o',	'10_o','11_o',	'12_o',	'13_o',	'14_o',	'15_o',	'16_o',	'17_o','18_o','19_o','20_o','21_o','22_o','23_o','Con_o'])


train_set = train_set.drop(columns=['year','temp_mean_past1h','time'])
val_set = val_set.drop(columns=['year','temp_mean_past1h','time'])
train_set = train_set.reindex(columns=['grad_dage',	'radia_glob_past1h',	'is_holiday',	0,	1,	2,	3,	4,	5,	6,	7,	8,	9,	10,	11,	12,	13,	14,	15,	16,	17,	18,	19,	20,	21,	22,	23,	'Con'])
val_set = val_set.reindex(columns=['grad_dage',	'radia_glob_past1h',	'is_holiday',	0,	1,	2,	3,	4,	5,	6,	7,	8,	9,	10,	11,	12,	13,	14,	15,	16,	17,	18,	19,	20,	21,	22,	23,	'Con'])
X_train_windowed = create_dataset(train_set,27,48,48,32)
X_val_windowed = create_dataset(val_set,27,48,48,32)

#%%
#earlystopping = EarlyStopping(patience=5)
# Setting up more layed LSTM which uses the encoding
Latent_dims = 16
past_inputs = tf.keras.Input(shape=(48,28), name='past_inputs')
encoder = layers.LSTM(Latent_dims, return_state=True, dropout=0.2)
encoder_outputs, state_h, state_c = encoder(past_inputs)

future_inputs = tf.keras.Input(shape=(48,27), name='future_inputs')
decoder_lstm = layers.LSTM(Latent_dims, return_sequences=True, dropout=0.2)
non_com_model = decoder_lstm(future_inputs, initial_state=[state_h,state_c])

non_com_model = layers.Dense(Latent_dims*2,activation='relu')(non_com_model)
non_com_model = layers.Dropout(0.2)(non_com_model)
non_com_model = layers.Dense(Latent_dims*4,activation='relu')(non_com_model)
non_com_model = layers.Dropout(0.2)(non_com_model)
output = layers.Dense(1,activation='relu')(non_com_model)

model = tf.keras.models.Model(inputs=[past_inputs,future_inputs], outputs=output)
optimizer = tf.keras.optimizers.SGD(momentum=0.1, lr=0.01)
loss = tf.keras.losses.MeanSquaredError()
model.compile(loss=loss,optimizer=optimizer,metrics=['mse'])
model.summary()
#%%
# Fit the model to our data
history = model.fit(X_train_windowed ,epochs=100, validation_data=(X_val_windowed), verbose=2)

#%%
model.save('LSTM_DK1.h5')

#%%
loaded_model = keras.models.load_model('LSTM_DK1.h5')  

#%%
history_dict = history.history
loss_vals = history_dict['loss']
val_loss = history_dict['val_loss']
epochs = range(1,len(loss_vals)+1)

plt.plot(epochs, loss_vals, 'bo')
plt.plot(epochs, val_loss, 'b')
plt.show
#%%
windows = 1
val_pred = pd.DataFrame()
for i, data in enumerate(X_val_windowed.take(windows)):
    (past, future),truth = data
    print(data[0][1])
    #print(future.shape)
    #print(truth.shape)
    model_pred = loaded_model.predict((past,future))
    window_df = pd.DataFrame(columns=['truth','pred'])
    window_df['truth'] = truth.numpy()[0]
    window_df['pred'] = pd.DataFrame(model_pred[0])
    val_pred = pd.concat([val_pred,window_df])

#%%
range_len = windows * 48
test_plot = pd.DataFrame()
test_plot['exact_values'] = val_pred['truth']
test_plot['predicted_values'] = val_pred['pred']
test_plot = test_plot.reset_index()
fig = plt.figure(figsize=(10, 6))
plt.subplot(1, 1, 1)
plt.title('Predicts vs Exact values for the LSTM model for DK1')
plt.plot(np.arange(0,range_len), test_plot['exact_values'], 'r-',
         label='Exact values')
plt.plot(np.arange(0,range_len), test_plot['predicted_values'], 'b-',
         label='Precited Values')
plt.legend(loc='upper right')
plt.ylabel('Consumption')
plt.xlabel('Time Steps')
fig.tight_layout()
plt.show()

#%%
val_pred_rescaled = con_scaler.inverse_transform(pd.DataFrame(val_pred['pred']))
val_truth_rescaled = con_scaler.inverse_transform(pd.DataFrame(val_pred['truth']))

forecast_mse = mean_squared_error(val_truth_rescaled,val_pred_rescaled)
forecast_r2 = r2_score(val_truth_rescaled,val_pred_rescaled)
print('forecast r2 score: '+ str(forecast_r2))
print('Forecast mse: '+ str(forecast_mse))
#%%
naive_y_val = np.roll(val_truth_rescaled,24)
naive_forecast_mse = mean_squared_error(val_truth_rescaled,naive_y_val)
naive_forecast_r2 = r2_score(val_truth_rescaled,naive_y_val)
print('baseline r2 score: '+ str(naive_forecast_r2))
print('Baseline mse: '+ str(naive_forecast_mse))

#%%
#------ IMPORT FORECAST DATA----------

forecast_df = pd.read_parquet('Data/dk1_forecast_sorted')
forecast_df = data_encoder(forecast_df)
forecast_df['hour'] = forecast_df['time'].dt.hour
forecast_df = forecast_df.drop(columns=['predicted_ahead'])
forecast_df.dropna(inplace=True)
forecast_df = pd.merge(forecast_df.reset_index(),forecast_val, on=['time'], how='inner').set_index('index')
forecast_df['mean_temp'] = forecast_df['mean_temp'] - 273.15
forecast_df['grad_dage'] = -(forecast_df['mean_temp'])+17
forecast_df.loc[forecast_df['grad_dage'] <=0, 'grad_dage'] = 0
forecast_df = forecast_df.rename(columns={'mean_radi':'radia_glob_past1h'})
forecast_df = forecast_df.drop(columns=['mean_temp'])
forecast_df = forecast_df.loc[forecast_df['Con'] < 3500]
forecast_df = forecast_df.loc[forecast_df['Con'] > 1100]
forecast_con = np.array(forecast_df['Con'])
forecast_con = forecast_con.reshape(-1,1)
cat_time = pd.get_dummies(forecast_df['hour'])
forecast_df = forecast_df.join(cat_time)
forecast_df = forecast_df.drop(columns=['hour'])
forecast_df = forecast_df.reindex(columns=['grad_dage_o','radia_glob_past1h_o','is_holiday_o',	'0_o',	'1_o',	'2_o',	'3_o',	'4_o',	'5_o',	'6_o',	'7_o',	'8_o',	'9_o',	'10_o','11_o',	'12_o',	'13_o',	'14_o',	'15_o',	'16_o',	'17_o','18_o','19_o','20_o','21_o','22_o','23_o','Con_o','grad_dage',	'radia_glob_past1h',	'is_holiday',	0,	1,	2,	3,	4,	5,	6,	7,	8,	9,	10,	11,	12,	13,	14,	15,	16,	17,	18,	19,	20,	21,	22,	23,	'Con'])
forecast_df[['grad_dage','radia_glob_past1h','Con']] = scaler.transform(forecast_df[['grad_dage','radia_glob_past1h','Con']])

#%%
forecast_windowed = create_forecast_dataset(forecast_df,48,48,48)

#%%
windows = 100
test_pred = pd.DataFrame()
for i, data in enumerate(forecast_windowed.take(windows)):
    (past, future),truth = data
    model_pred = loaded_model.predict((past,future))
    window_df = pd.DataFrame(columns=['truth','pred'])
    window_df['truth'] = truth.numpy()[0]
    window_df['pred'] = pd.DataFrame(model_pred[0])
    test_pred = pd.concat([test_pred,window_df])

#%%
forecast_pred_rescaled = con_scaler.inverse_transform(pd.DataFrame(test_pred['pred']))
forecast_truth_rescaled = con_scaler.inverse_transform(pd.DataFrame(test_pred['truth']))

forecast_mse = mean_squared_error(forecast_truth_rescaled,forecast_pred_rescaled)
forecast_r2 = r2_score(forecast_truth_rescaled,forecast_pred_rescaled)
print('forecast r2 score: '+ str(forecast_r2))
print('Forecast mse: '+ str(forecast_mse))
#%%
naive_y_val = np.roll(forecast_con,48)
naive_forecast_mse = mean_squared_error(forecast_con,naive_y_val)
naive_forecast_r2 = r2_score(forecast_con,naive_y_val)
print('baseline r2 score: '+ str(naive_forecast_r2))
print('Baseline mse: '+ str(naive_forecast_mse))

#%%


test_plot = pd.DataFrame()
test_plot['exact_values'] = pd.DataFrame(forecast_truth_rescaled)
test_plot['predicted_values'] = pd.DataFrame(forecast_pred_rescaled)
range_len = len(test_plot)
test_plot = test_plot.reset_index()
fig = plt.figure(figsize=(20, 6))
plt.subplot(1, 1, 1)
plt.title('Predicted vs Exact values for DK2 by the LSTM model')
plt.plot(np.arange(0,range_len), test_plot['exact_values'], 'r-',
         label='Exact values')
plt.plot(np.arange(0,range_len), test_plot['predicted_values'], 'b-',
         label='Precited Values')
plt.legend(loc='upper right')
plt.ylabel('Consumption')
plt.xlabel('Time steps')
fig.tight_layout()
plt.show()


#%%
forecast_truth_weekly = daily_meaner(forecast_truth_rescaled)
forecast_pred_weekly = daily_meaner(forecast_pred_rescaled)
#%%


test_plot = pd.DataFrame()
test_plot['exact_values'] = forecast_truth_weekly
test_plot['predicted_values'] = forecast_pred_weekly
range_len = len(test_plot)
test_plot = test_plot.reset_index()
fig = plt.figure(figsize=(15, 6))
plt.subplot(1, 1, 1)
plt.title('Predicted vs Exact values for DK1 by the LSTM model', fontsize=20)
plt.plot(np.arange(0,range_len), test_plot['exact_values'], 'r-',
         label='Exact values')
plt.plot(np.arange(0,range_len), test_plot['predicted_values'], 'b-',
         label='Precited Values')
plt.legend(loc='upper right', fontsize=16)
plt.ylabel('Consumption', fontsize=20)
plt.xlabel('Days', fontsize=20)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
fig.tight_layout()
plt.show()
# %%
