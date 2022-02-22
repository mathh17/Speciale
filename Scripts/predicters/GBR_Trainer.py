#%%
#Load all the packages
import os
import numpy as np
import pandas as pd
import Holidays_calc as hc
from joblib import dump, load
from matplotlib import pyplot as plt
from sklearn import ensemble
#%%
#Select the grid company you wis to predict on:
grid_company = "Radius"

if grid_company == "Fyn":
    path = r'C:\Users\oeste\OneDrive\Uni\DS_3_semester\VI_Projekt\Scripts\data\stations_data_dk1'
    df_el_data = pd.read_pickle("data/Midtfyn_el_data.pkl")
    el_data_2021 = pd.read_pickle("data/Midtfyn_el_data_2021.pkl")
    model_name = "fyn_gbr.joblib"
if grid_company == "Frederikshavn":
    path = r'C:\Users\oeste\OneDrive\Uni\DS_3_semester\VI_Projekt\Scripts\data\stations_data_dk1'
    df_el_data = pd.read_pickle("data/Frederikshavn_el_data.pkl")
    el_data_2021 = pd.read_pickle("data/Frederikshavn_el_data_2021.pkl")
    model_name = "frederikshavn_gbr.joblib"
if grid_company == "Tarm":
    path = r'C:\Users\oeste\OneDrive\Uni\DS_3_semester\VI_Projekt\Scripts\data\stations_data_dk1'
    df_el_data = pd.read_pickle("data/Tarm_el_data.pkl")
    el_data_2021 = pd.read_pickle("data/Tarm_el_data_2021.pkl")
    model_name = "tarm_gbr.joblib" 
if grid_company == "Radius":
    path = r'C:\Users\oeste\OneDrive\Uni\DS_3_semester\VI_Projekt\Scripts\data\stations_data_dk2'
    df_el_data = pd.read_pickle("data/jaegerspris_el_data.pkl")
    el_data_2021 = pd.read_pickle("data/jaegerspris_el_data_2021.pkl")
    model_name = "Radius_gbr.joblib"   

#%%
# read the files from the datafolder containing data fra DK2

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
# Declaring hyperparameters, 
# initializing the model, 
# fitting it to the data,
# predicting on the validation data.
losses = 'squared_error'
lrs = 0.01
estimators = 500
crits = 'friedman_mse'
depth = 10
s_samples = 1

results = []
# The for loops goes through the different hyperparameters
gbt = ensemble.GradientBoostingRegressor(#loss=losses,
                                        learning_rate=lrs,
                                        n_estimators=estimators,
                                        #criterion=crits,
                                        max_depth=depth,
                                        subsample=s_samples)
gbt.fit(X_train,y_train)
y_val_hat = gbt.predict(X_val)
acc = gbt.score(X_val,y_val)
# the results from the different hyperparameters are then stored so we can get the best one
results.append([acc,losses,lrs,estimators,crits, depth])
results_df = pd.DataFrame(results)
results_df.columns=['accuracy',"loss functions","learning_rate","n_estimators","criterions", "depth"]
#prints the results from best to worst in regards to accuracy, listting the hyperparameters for the result
results_df = results_df.sort_values('accuracy', ascending=False)
results_df

#%%
dump(gbt, model_name)
# %%
# Plots the deviance in the prediction and on the training data. To visualize how the model learns and behaves.
best_est = results_df.iloc[0,3]
test_score = np.zeros((best_est,), dtype=np.float64)
for i, y_pred in enumerate(gbt.staged_predict(X_test)):
    test_score[i] = gbt.loss_(y_test, y_pred)

fig = plt.figure(figsize=(6, 6))
plt.subplot(1, 1, 1)
plt.title('Deviance')
plt.plot(np.arange(best_est) + 1, gbt.train_score_, 'b-',
         label='Training Set Deviance')
plt.plot(np.arange(best_est) + 1, test_score, 'r-',
         label='Test Set Deviance')
plt.legend(loc='upper right')
plt.xlabel('Boosting Iterations')
plt.ylabel('Deviance')
fig.tight_layout()
plt.show()
#%%