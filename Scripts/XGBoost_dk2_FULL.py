#%%
import os
import numpy as np
import pandas as pd
import Holidays_calc as hc
from matplotlib import pyplot as plt
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

#import shap
#%%
#Calling the holiday function to build a column for if its a holiday or not
def holidays(df):
    holidays = []
    for i, row in df.iterrows():
        is_holiday = hc.get_date_type(row['time'])
        holidays.append(is_holiday)
    return holidays

#Encodes the holidays into the holidays
def data_encoder(df): 
    df['time'] = pd.to_datetime(df['time'],format='%Y-%m-%dT%H:%M:%S', utc=True)
    df['is_holiday'] = holidays(df)
    return df

# Takes the predictions done for a year and makes it into a mean pr day
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
Observed_data = pd.read_parquet('DK2_XGB_training_data')

"""
Set up training, validation and test sets.
The XGBoost needs it to be in a special matrix.
"""

train_set = Observed_data.loc[Observed_data['year'] <= 2017] 
val_set =  Observed_data.loc[Observed_data['year'] >= 2018]
y_train = train_set['Con']
y_val = val_set['Con']
train_set = train_set.drop(columns=['Con','year','temp_mean_past1h','time'])
val_set = val_set.drop(columns=['Con','year','temp_mean_past1h','time'])
train_set = train_set.reindex(columns=['grad_dage',	'radia_glob_past1h', 'is_holiday', 'hour'])
val_set = val_set.reindex(columns=['grad_dage',	'radia_glob_past1h', 'is_holiday','hour'])

#XGBoost takes a specific matrix, these two functions 
dtrain = xgb.DMatrix(train_set,y_train)
dval = xgb.DMatrix(val_set,y_val)
#%%
#set up lists containing settings for the parameters for the XGBoost to train on
#max depth
depth_param =  [6,8,10]
#learning rate
eta_param =  [0.1,0.03,0.01]
#gamma, also called pruning
gamma_param =  [4,6,8]
#Number of rounds
rounds_param =  [100,250,500,1000]
#minimum weight needed for making a new child
min_child_param = [3,5,8]

#Nested loops which bruteforces the optimal settings for the XGBoost model
#the results are appended to a list where the performance of the different settings can be inspected
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
#Optimal settings for the XGBoost
param = {'max_depth':10, 
                        'eta':0.03, 
                        'gamma':6,
                        'min_child_weight':8,
                        'three_method': 'exact',
                        'objective':'reg:squarederror',
                        'seed':42}
num_round = 100

#trains and predicts the optimal model
bst = xgb.train(param, dtrain,num_round)
val_preds = bst.predict(dval)

#Calculates MSE and R^2 for the predictions
mse = mean_squared_error(y_val,val_preds)
val_r2= r2_score(y_val,val_preds)
print('forecast r2 score: '+ str(val_r2))
print('Forecast mse: '+ str(mse))
#%%
#Calculates the MSE and R^2 scores for the naive model
naive_y_val = np.roll(y_val,24)
naive_mse = mean_squared_error(y_val,naive_y_val)
naive_val_r2= r2_score(y_val,naive_y_val)
print('forecast r2 score: '+ str(naive_val_r2))
print('Forecast mse: '+ str(naive_mse))
#%%
#Plots the true and predicted values in one graph.
test_plot = pd.DataFrame()
test_plot['exact_values'] = y_val
test_plot['predicted_values'] = val_preds
range_len = len(test_plot)
test_plot = test_plot.reset_index()
fig = plt.figure(figsize=(10, 6))
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
#this section imports the forecast data and processes the data so it is ready to be predicted on
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
#Makes predictions fore the forecast data and calculates the MSE and R^2
forecast_preds = bst.predict(forecast_xgb)
forecast_mse = mean_squared_error(forecast_con,forecast_preds)
forecast_r2 = r2_score(forecast_con,forecast_preds)
print('forecast r2 score: '+ str(forecast_r2))
print('Forecast mse: '+ str(forecast_mse))
#%%
#Calculates the MSE and R^2 scores for the naive model
naive_y_val = np.roll(forecast_con,24)
naive_forecast_mse = mean_squared_error(forecast_con,naive_y_val)
naive_forecast_r2 = r2_score(forecast_con,naive_y_val)
print('forecast r2 score: '+ str(naive_forecast_mse))
print('Forecast mse: '+ str(naive_forecast_r2))
#%%
#Plots the true and predicted values in one graph.
test_plot = pd.DataFrame()
test_plot['exact_values'] = forecast_con
test_plot['predicted_values'] = forecast_preds
test_plot = test_plot.reset_index()
range_len = len(test_plot)
fig = plt.figure(figsize=(10, 6))
plt.subplot(1, 1, 1)
plt.title('Predicts vs Exact values by XGBoost model for DK2', fontsize=16)
plt.plot(np.arange(0,range_len), test_plot['exact_values'], 'r-',
         label='Exact values')
plt.plot(np.arange(0,range_len), test_plot['predicted_values'], 'b-',
         label='Precited Values')

plt.legend(loc='upper right', fontsize=16)
plt.ylabel('Consumption', fontsize=16)
plt.xlabel('Time steps', fontsize=16)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
#fig.tight_layout()
plt.show()
# %%
#Calculates the daily mean for the predictions
forecast_truth_weekly = daily_meaner(forecast_con)
forecast_pred_weekly = daily_meaner(forecast_preds)

#Plots the daily means for the true and predicted values
test_plot = pd.DataFrame()
test_plot['exact_values'] = forecast_truth_weekly
test_plot['predicted_values'] = forecast_pred_weekly
range_len = len(test_plot)
test_plot = test_plot.reset_index()
fig = plt.figure(figsize=(15, 6))
plt.subplot(1, 1, 1)
plt.title('Predicted and Exact values for DK2 by the XGB model', fontsize=20)
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
