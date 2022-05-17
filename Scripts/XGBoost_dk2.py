#%%
import os
import numpy as np
import pandas as pd
import Holidays_calc as hc
from matplotlib import pyplot as plt
#import seaborn as sns
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
#import shap

#%%
# read the files from the datafolder containing data fra DK2
# changing the path to the datafolder
path = r'C:\Users\oeste\OneDrive\Uni\Speciale\Scripts\Data\dmi_data_dk2'
EN_path = r'C:\Users\MTG.ENERGINET\OneDrive - Energinet.dk\Dokumenter\Speciale\Scripts\Data\dmi_data_dk2'

os.chdir(EN_path)

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
old_path = r'C:\Users\oeste\OneDrive\Uni\Speciale\Scripts'
EN_path = r'C:\Users\MTG.ENERGINET\OneDrive - Energinet.dk\Dokumenter\Speciale\Scripts'

os.chdir(EN_path)
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
Observed_data = pd.DataFrame(conc_data[['temp_mean_past1h','radia_glob_past1h','Con','time']])

conc_data = data_encoder(conc_data)
Observed_data['is_holiday'] = conc_data['is_holiday']
Observed_data['year'] = conc_data['time'].dt.year 
Observed_data['grad_dage'] = -(conc_data['temp_mean_past1h'])+17
Observed_data.loc[Observed_data['grad_dage'] <=0, 'grad_dage'] = 0
Observed_data['hour'] = conc_data['time'].dt.hour
#%%
"""
Set up training, validation and test sets.
The XGBoost needs it to be in a special matrix.
"""

train_set = Observed_data.loc[Observed_data['year'] <= 2018] 
val_set =  Observed_data.loc[Observed_data['year'] == 2019]
y_train = train_set['Con']
y_val = val_set['Con']
train_set = train_set.drop(columns=['Con','year','temp_mean_past1h','time'])
val_set = val_set.drop(columns=['Con','year','temp_mean_past1h','time'])
train_set = train_set.reindex(columns=['grad_dage',	'radia_glob_past1h', 'is_holiday', 'hour'])
val_set = val_set.reindex(columns=['grad_dage',	'radia_glob_past1h', 'is_holiday','hour'])

dtrain = xgb.DMatrix(train_set,y_train)
dval = xgb.DMatrix(val_set,y_val)
#%%
depth_param =  [6,8,10]
eta_param =  [0.1,0.03,0.01]
gamma_param =  [4,6,8]
rounds_param =  [100,250,500,1000]
min_child_param = [3,5,8]
results = []
for depth in depth_param:
    for eta in eta_param:
        for gamma in gamma_param:
            for rounds in rounds_param:
                for children in min_child_param:
                    param = {'max_depth':depth, 
                            'eta':eta, 
                            'gamma': gamma,
                            'min_child_weight':children,
                            'objective':'reg:squarederror',
                            'seed':42}
                    num_round = rounds
                    bst = xgb.train(param, dtrain,num_round)
                    val_preds = bst.predict(dval)
                    val_mse = mean_squared_error(y_val,val_preds)
                    results.append([val_mse,depth,eta,gamma,children,rounds])
results = pd.DataFrame(results)
results.columns=['val_mse','Tree depth','Eta/learning rate','gamma','min_child_weight','rounds']

#%%
param = {'max_depth':10, 
                        'eta':0.03, 
                        'gamma':6,
                        'min_child_weight':8,
                        'objective':'reg:squarederror',
                        'seed':42}
num_round = 100
bst = xgb.train(param, dtrain,num_round)
val_preds = bst.predict(dval)

mse = mean_squared_error(y_val,val_preds)
val_r2= r2_score(y_val,val_preds)
print('forecast r2 score: '+ str(val_r2))
print('Forecast mse: '+ str(mse))
#%%
naive_y_val = np.roll(y_val,24)

naive_mse = mean_squared_error(y_val,naive_y_val)
naive_val_r2= r2_score(y_val,naive_y_val)
print('forecast r2 score: '+ str(naive_val_r2))
print('Forecast mse: '+ str(naive_mse))
#%%

test_plot = pd.DataFrame()
test_plot['exact_values'] = y_val[0:100]
test_plot['predicted_values'] = val_preds[0:100]
range_len = len(test_plot)
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
#------ IMPORT FORECAST DATA----------
#%%
forecast_df = pd.read_parquet('Data/dk2_forecast_sorted')
forecast_df = data_encoder(forecast_df)
forecast_df['hour'] = forecast_df['time'].dt.hour
forecast_df = forecast_df.drop(columns=['predicted_ahead'])
forecast_df.dropna(inplace=True)
forecast_df['mean_temp'] = forecast_df['mean_temp'] - 273.15
forecast_df['grad_dage'] = -(forecast_df['mean_temp'])+17
forecast_df.loc[forecast_df['grad_dage'] <=0, 'grad_dage'] = 0
forecast_df = forecast_df.rename(columns={'mean_radi':'radia_glob_past1h'})
forecast_df = forecast_df.drop(columns=['mean_temp'])
forecast_df = forecast_df.reindex(columns=['grad_dage',	'radia_glob_past1h', 'is_holiday', 'hour','Con'])
forecast_con = forecast_df['Con']
forecast_df = forecast_df.drop(columns=['Con'])
forecast_xgb = xgb.DMatrix(forecast_df,forecast_con)
# %%
forecast_preds = bst.predict(forecast_xgb)
# %%
forecast_mse = mean_squared_error(forecast_con,forecast_preds)
forecast_r2 = r2_score(forecast_con,forecast_preds)
print('forecast r2 score: '+ str(forecast_r2))
print('Forecast mse: '+ str(forecast_mse))
#%%
naive_y_val = np.roll(forecast_con,24)
naive_forecast_mse = mean_squared_error(forecast_con,naive_y_val)
naive_forecast_mse
#%%
test_plot = pd.DataFrame()
test_plot['exact_values'] = forecast_con
test_plot['predicted_values'] = forecast_preds
test_plot = test_plot.reset_index()
range_len = len(forecast_preds)
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
# %%
