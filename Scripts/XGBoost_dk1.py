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
Observed_data = pd.read_parquet('DK1_XGB_training_data')



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

dtrain = xgb.DMatrix(train_set,y_train)
dval = xgb.DMatrix(val_set,y_val)
#%%
depth_param =  [8,10,12,14]
eta_param =  [0.03,0.01]
gamma_param =  [4]
rounds_param =  [100,500,1000]
min_child_param = [8,10,12]
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
                        'eta':0.01, 
                        'gamma':4,
                        'min_child_weight':8,
                        'objective':'reg:squarederror',
                        'seed':42}
num_round = 500
bst = xgb.train(param, dtrain,num_round)
val_preds = bst.predict(dval)

mse = mean_squared_error(y_val,val_preds)
val_r2= r2_score(y_val,val_preds)
print('forecast r2 score: '+ str(val_r2))
print('Forecast mse: '+ str(mse))
#%%
naive_y_val = np.roll(y_val,48)

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
forecast_df = pd.read_parquet('Data/dk1_forecast_sorted')
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
naive_forecast_r2 = r2_score(forecast_con,naive_y_val)
print('forecast r2 score: '+ str(naive_forecast_mse))
print('Forecast mse: '+ str(naive_forecast_r2))
#%%

test_plot = pd.DataFrame()
test_plot['exact_values'] = forecast_con
test_plot['predicted_values'] = forecast_preds
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
# %%
