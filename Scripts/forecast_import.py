#%%
import pandas as pd
import os
#%%
"""
---------
This section imports the forecast data from DK2 for the year 2019. Merges it with the consumption for the given year and sorts it into the right time frames

--------
"""
df = pd.read_parquet("Data/dk2_forecast_data")
df = df.rename(columns={'Date':'time'})
df = df[18:]
#%%
# Read Enernginet Pickle Data
# Change back path
old_path = r'C:\Users\oeste\OneDrive\Uni\Speciale\Scripts'
os.chdir(old_path)
df_DK1_2015_2020 = pd.read_parquet("data/el_data_2010-2020_dk1")
df_DK2_2015_2020 = pd.read_parquet("data/el_data_2010-2020_dk1")
df_DK2_2015_2020['HourUTC'] = pd.to_datetime(df_DK2_2015_2020['HourUTC'],format='%Y-%m-%dT%H:%M:%S', utc=True)
df_DK2_2015_2020['HourUTC'] = pd.to_datetime(df_DK2_2015_2020['HourUTC'],format='%Y-%m-%dT%H:%M:%S', utc=True)
df_DK1_2015_2020 = df_DK1_2015_2020.rename(columns={'HourUTC':'time','HourlySettledConsumption':'Con'})
df_DK2_2015_2020 = df_DK2_2015_2020.rename(columns={'HourUTC':'time','HourlySettledConsumption':'Con'})
df_DK2_2015_2020 = df_DK2_2015_2020.drop(['PriceArea'],axis=1)
df_DK1_2015_2020 = df_DK1_2015_2020.drop(['PriceArea'],axis=1)

#%%
pred_counter = 1
dictio = {}
while pred_counter < 49:
    pred_df = df.groupby('predicted_ahead')
    #print(pred_df.get_group(int(pred_counter))[0:1])
    dictio[pred_counter] = pred_df.get_group(int(pred_counter))
    pred_counter +=1

counter = 1
while counter < 49:
    dictio[counter] = pd.merge(dictio[counter],df_DK2_2015_2020, on=['time'], how='inner')
    counter +=1

#%%
row_counter = 0
items = [item[1] for item in list(dictio.items())]
forecast_df = pd.DataFrame(columns = items[0].columns)

while row_counter < 2911:
    dict_counter = 0
    counter = 0
    while dict_counter < 48:
        drop_number = items[dict_counter].iloc[counter].name
        if dict_counter % 3 == 0 and dict_counter > 2:
            counter += 1
        forecast_df.loc[len(forecast_df)] = items[dict_counter].iloc[counter]
        items[dict_counter] = items[dict_counter].drop(index=drop_number)
        dict_counter += 1
    row_counter += 1

#%%
forecast_df.to_parquet('dk2_forecast_sorted')

#%%
"""
---------
This section imports the forecast data from DK1 for the year 2019. Merges it with the consumption for the given year and sorts it into the right time frames

--------
"""
df = pd.read_parquet("Data/dk1_forecast_data")
df = df.rename(columns={'Date':'time'})
df = df[18:]

# Read Enernginet Pickle Data
# Change back path
old_path = r'C:\Users\oeste\OneDrive\Uni\Speciale\Scripts'
os.chdir(old_path)
df_DK1_2015_2020 = pd.read_pickle("data/dk1_data_2015_2020.pkl")
df_DK1_2015_2020['HourUTC'] = pd.to_datetime(df_DK1_2015_2020['HourUTC'],format='%Y-%m-%dT%H:%M:%S', utc=True)
df_DK1_2015_2020 = df_DK1_2015_2020.rename(columns={'HourUTC':'time','HourlySettledConsumption':'Con'})
df_DK1_2015_2020 = df_DK1_2015_2020.drop(['PriceArea'],axis=1)



#%%
pred_counter = 1
dictio = {}
while pred_counter < 49:
    pred_df = df.groupby('predicted_ahead')
    #print(pred_df.get_group(int(pred_counter))[0:1])
    dictio[pred_counter] = pred_df.get_group(int(pred_counter))
    pred_counter +=1

counter = 1
while counter < 49:
    dictio[counter] = pd.merge(dictio[counter],df_DK1_2015_2020, on=['time'], how='inner')
    counter +=1



#%%
row_counter = 0
items = [item[1] for item in list(dictio.items())]
forecast_df = pd.DataFrame(columns = items[0].columns)

while row_counter < 2911:
    dict_counter = 0
    counter = 0
    while dict_counter < 48:
        drop_number = items[dict_counter].iloc[counter].name
        if dict_counter % 3 == 0 and dict_counter > 2:
            counter += 1
        forecast_df.loc[len(forecast_df)] = items[dict_counter].iloc[counter]
        items[dict_counter] = items[dict_counter].drop(index=drop_number)
        dict_counter += 1
    row_counter += 1

#%%
forecast_df.to_parquet('dk1_forecast_sorted')


# %%
forecast_df = pd.read_parquet('dk1_forecast_sorted')
#%%

