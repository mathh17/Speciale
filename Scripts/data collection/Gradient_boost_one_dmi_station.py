"""
In this script the algorithm Adaboost will be implemented. 
TBC

"""
#%%
#Load all the packages
import numpy as np
import sklearn
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
import pandas as pd
import requests
from datetime import datetime, timedelta
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn import ensemble
from sklearn import tree

#%%
#Read DMI Pickle Data
dmi_data = pd.read_pickle('data/dmi_data.pkl')

#%%
#Read Enernginet Pickle Data
df_DK1_2010_2015 = pd.read_pickle("data/dk1_data_2010_2015.pkl")
df_DK2_2010_2015 = pd.read_pickle("data/dk2_data_2010_2015.pkl")
df_DK1_2015_2020 = pd.read_pickle("data/dk1_data_2015_2020.pkl")
df_DK2_2015_2020 = pd.read_pickle("data/dk2_data_2015_2020.pkl")
df_DK1 = pd.concat([df_DK1_2010_2015,df_DK1_2015_2020], ignore_index=True)
df_DK2 = pd.concat([df_DK2_2010_2015,df_DK2_2015_2020], ignore_index=True)

#%%
#Merge data into one DF, on the hour of observations
dmi_data['time'] = pd.to_datetime(dmi_data['time'],format='%Y-%m-%dT%H:%M:%S', utc=True)
df_DK1['HourUTC'] = pd.to_datetime(df_DK1['HourUTC'],format='%Y-%m-%dT%H:%M:%S', utc=True)
df_DK1 = df_DK1.rename(columns={'HourUTC':'time'})
conc_data = pd.merge(dmi_data, df_DK1, on='time', how='outer')
#%%
nan = float("NaN")
#conc_data.replace("",nan, inplace=True )
conc_data.dropna(inplace=True)

#%%
#Dividing train test and validation.
values = conc_data['Con']
pred_data = conc_data[['temp_mean_past1h','radia_glob_past1h']]
#%%
#Dividing dataset into test and train
X_train, X_test, y_train, y_test = train_test_split(pred_data, values, test_size=0.2, random_state=42)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

#%%
#Dividing the train set into train and validation
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)
print(X_train.shape, X_val.shape, y_train.shape, y_val.shape)
#%%
# Declaring hyperparameters, 
# initializing the model, 
# fitting it to the data,
# predicting on the validation data.
loss = ['squared_error']#,'absolute_error','huber']
learning_rate = [0.1]#,0.1,0.2,1]
n_estimators = [500]#,50,100,200]
criterion = ['mse']#,'mae']

results = []
# The for loops goes through the different hyperparameters
for losses in loss:
    for lrs in learning_rate:
        for estimators in n_estimators:
            for crits in criterion:
                gbt = ensemble.GradientBoostingRegressor(loss=losses,
                                                        learning_rate=lrs,
                                                        n_estimators=estimators,
                                                        criterion=crits)
                gbt.fit(X_train,y_train)
                y_val_hat = gbt.predict(X_val)
                acc = gbt.score(X_val,y_val)
# the results from the different hyperparameters are then stored so we can get the best one
                results.append([acc,losses,lrs,estimators,crits])
results_df = pd.DataFrame(results)
results_df.columns=['accuracy',"loss functions","learning_rate","n_estimators","criterions"]
#prints the results from best to worst in regards to accuracy, listting the hyperparameters for the result
results_df = results_df.sort_values('accuracy', ascending=False)
results_df
# %%
#Calculates the MSE if you just use the data from 24hours before at prediction.
naive_y_val = np.roll(y_val.to_numpy(),24)
mse_naive_y_val = np.sum((y_val-naive_y_val)**2)/len(y_val)
mse_naive_y_val
#%%
#Calculates the MSE for the predicted values from the validation set.
np.sum((y_val-y_val_hat)**2)/len(y_val)
# %%
#pair plot of DMI data and the consmption. 
pair_data = X_val
pair_data['con'] = y_val
pairplot = sns.pairplot(pair_data)
plt.show()
# %%
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
# %%
y_plot = pd.DataFrame()
y_plot['exact_values'] = y_val
y_plot['predicted_values'] = y_val_hat
y_plot = y_plot.sort_index()
fig = plt.figure(figsize=(6, 6))
plt.subplot(1, 1, 1)
plt.title('Predicts vs Exact values')
plt.plot(np.arange(0,100), y_plot['exact_values'], 'r-',
         label='Exact values')
plt.plot(np.arange(0,100), y_plot['predicted_values'], 'b-',
         label='Precited Values')
plt.legend(loc='upper right')
#plt.xlabel('Boosting Iterations')
plt.ylabel('Consumption')
fig.tight_layout()
plt.show()
# %%
