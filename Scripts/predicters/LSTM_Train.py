#%%
#Load all the packages
import os
import numpy as np
import pandas as pd
import Holidays_calc as hc
import tensorflow as tf
from tensorflow import keras
from matplotlib import pyplot as plt
from tensorflow.keras import layers
from sklearn import preprocessing

#%%
#Select the grid company you wis to predict on:
grid_company = "Frederikshavn"

if grid_company == "Fyn":
    path = r'C:\Users\oeste\OneDrive\Uni\DS_3_semester\VI_Projekt\Scripts\data\stations_data_dk1'
    df_el_data = pd.read_pickle("data/Midtfyn_el_data.pkl")
    el_data_2021 = pd.read_pickle("data/Midtfyn_el_data_2021.pkl")
if grid_company == "Frederikshavn":
    path = r'C:\Users\oeste\OneDrive\Uni\DS_3_semester\VI_Projekt\Scripts\data\stations_data_dk1'
    df_el_data = pd.read_pickle("data/Frederikshavn_el_data.pkl")
    el_data_2021 = pd.read_pickle("data/Frederikshavn_el_data_2021.pkl")
if grid_company == "Tarm":
    path = r'C:\Users\oeste\OneDrive\Uni\DS_3_semester\VI_Projekt\Scripts\data\stations_data_dk1'
    df_el_data = pd.read_pickle("data/Tarm_el_data.pkl")
    el_data_2021 = pd.read_pickle("data/Tarm_el_data_2021.pkl")
if grid_company == "Radius":
    path = r'C:\Users\oeste\OneDrive\Uni\DS_3_semester\VI_Projekt\Scripts\data\stations_data_dk2'
    df_el_data = pd.read_pickle("data/jaegerspris_el_data.pkl")
    el_data_2021 = pd.read_pickle("data/jaegerspris_el_data_2021.pkl")

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


#%%
# Setting up more layed LSTM which uses the encoding
Latent_dims = 16
past_inputs = tf.keras.Input(shape=(48,28), name='past_inputs')
encoder = layers.LSTM(Latent_dims, return_state=True, dropout=0.2)
encoder_outputs, state_h, state_c = encoder(past_inputs)

future_inputs = tf.keras.Input(shape=(48,27), name='future_inputs')
decoder_lstm = layers.LSTM(Latent_dims, return_sequences=True, dropout=0.2)
non_com_model = decoder_lstm(future_inputs, initial_state=[state_h,state_c])

non_com_model = layers.Dense(Latent_dims,activation='elu')(non_com_model)
non_com_model = layers.Dropout(0.2)(non_com_model)
non_com_model = layers.Dense(Latent_dims,activation='elu')(non_com_model)
non_com_model = layers.Dropout(0.2)(non_com_model)
output = layers.Dense(1,activation='elu')(non_com_model)

model = tf.keras.models.Model(inputs=[past_inputs,future_inputs], outputs=output)
optimizer = tf.keras.optimizers.SGD(momentum=0.9, lr=0.001)
loss = tf.keras.losses.Huber()
model.compile(loss=loss,optimizer=optimizer,metrics=['mse'])
model.summary()
#%%
# Fit the model to our data
history = model.fit(X_train_windowed ,epochs=250, validation_data=(X_val_windowed))


#%%
# Saving the model to be used later
model.save('LSTM_250Epochs_Jaegerspris.h5')
#%%
#%%
# Plot the test loss and validation loss to check for overfitting
history_dict = model.history
loss_vals = history_dict['loss']
val_loss = history_dict['val_loss']
epochs = range(1,len(loss_vals)+1)

plt.plot(epochs, loss_vals, 'bo')
plt.plot(epochs, val_loss, 'b')
plt.show