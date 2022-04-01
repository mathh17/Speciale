#%%
import os
import numpy as np
import pandas as pd
import Holidays_calc as hc
from matplotlib import pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.metrics import mean_squared_error
import shap

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
df_DK1 = pd.concat([df_DK1_2010_2015,df_DK1_2015_2020], ignore_index=True)
df_DK2 = pd.concat([df_DK2_2010_2015,df_DK2_2015_2020], ignore_index=True)

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
pred_data = pd.DataFrame(conc_data[['temp_mean_past1h','radia_glob_past1h','Con']])

conc_data = data_encoder(conc_data)
pred_data['is_holiday'] = conc_data['is_holiday']
pred_data['year'] = conc_data['time'].dt.year 
pred_data['day'] = conc_data['time'].dt.dayofweek
pred_data['grad_dage'] = -(conc_data['temp_mean_past1h'])+17
pred_data.loc[pred_data['grad_dage'] <=0, 'grad_dage'] = 0
pred_data['time'] = conc_data['time'].dt.hour

hours_in_year = 8760

#%%
"""
Set up training, validation and test sets.
The XGBoost needs it to be in a special matrix.
"""
train_set = pred_data.loc[pred_data['year'] <= 2017] 
val_set =  pred_data.loc[pred_data['year'] == 2018]
test_set = pred_data.loc[pred_data['year'] == 2019]
y_train = train_set['Con']
y_val = val_set['Con']
y_test = test_set['Con']
train_set = train_set.drop(columns=['Con','year','day','temp_mean_past1h'])
val_set = val_set.drop(columns=['Con','year','day','temp_mean_past1h'])
test_set = test_set.drop(columns=['Con','year','day','temp_mean_past1h'])
dtrain = xgb.DMatrix(train_set,y_train)
dval = xgb.DMatrix(val_set,y_val)
dtest = xgb.DMatrix(test_set,y_test)
#%%
param = {'max_depth':2, 'eta':0.1, 'objective':'reg:squarederror'}
num_round = 500
bst = xgb.train(param, dtrain,num_round)
# make prediction
preds = bst.predict(dval)
preds


# %%
mse = mean_squared_error(y_val,preds)
mse
#%%
naive_y_val = np.roll(y_val,48)
#%%
mse = mean_squared_error(y_val,naive_y_val)
mse
#%%

test_plot = pd.DataFrame()
test_plot['exact_values'] = y_val[0:100]
test_plot['predicted_values'] = preds[0:100]
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

# Create object that can calculate shap values
explainer = shap.TreeExplainer(bst)

# Calculate Shap values
shap_values = explainer.shap_values(train_set)
#%%
shap.summary_plot(shap_values,train_set)

# %%
shap.plots.bar(shap_values)

# %%
