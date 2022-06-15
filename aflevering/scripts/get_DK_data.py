#%%
import numpy as np
import pandas as pd
import requests
from datetime import datetime, timedelta
from matplotlib import pyplot as plt
# %%
print('start')
date_start = datetime(2019,12,30)
date_end = datetime(2019,12,31)
url = """https://api.energidataservice.dk/datastore_search_sql?sql=SELECT * FROM "productionconsumptionsettlement" WHERE "HourUTC" >= '""" + date_start.strftime("%Y-%m-%dT%H:%M:%S") + """' AND "HourUTC" < '""" + date_end.strftime("%Y-%m-%dT%H:%M:%S") + """' """
api_call = requests.get(url)
df = pd.DataFrame(api_call.json()["result"]["records"]).drop(columns = ["_id", "_full_text"])
#extract relevant columns
df.fillna(0, inplace=True)
data_df = df[['HourUTC','PriceArea']].copy()
data_df['Con'] = df['GrossConsumptionMWh'] - df['PowerToHeatMWh'] - df['LocalPowerSelfConMWh'] - df['SolarPowerSelfConMWh'] - df['GridLossTransmissionMWh'] - df['GridLossInterconnectorsMWh'] - df['GridLossDistributionMWh']#subtract consumption from electric boilers
df_DK1 = data_df[data_df.PriceArea == 'DK1'] #split into DK1 and DK2
df_DK2 = data_df[data_df.PriceArea == 'DK2']
#%%
df_DK1.to_parquet("el_data_2010-2020_dk1")
df_DK2.to_parquet("el_data_2010-2020_dk2")
print('done')
#%%
#Converting the consumption to means pr week
con = np.array(df_DK1['Con'])
weekly_con = []
iterations = 0
while iterations <= 52:
    hours_in_a_week = 0
    one_week_con = 0
    while hours_in_a_week <= 168:
        hour = (iterations*168)+hours_in_a_week
        one_week_con += con[hour-1]
        hours_in_a_week += 1
    weekly_con.append(one_week_con/7)
    iterations += 1




# %%
plt.plot( df_DK1['HourUTC'],df_DK1['Con'])
plt.show()
# %%
plt.plot(weekly_con)
plt.show()
# %%
