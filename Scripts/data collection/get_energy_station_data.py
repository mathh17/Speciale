#%%
import numpy as np
import pandas as pd
import requests
from datetime import datetime, timedelta
from matplotlib import pyplot as plt
# %%
print('start')
date_start = datetime(2021,1,1)
date_end = datetime(2021,12,31)
url = """https://api.energidataservice.dk/datastore_search_sql?sql=SELECT "HourUTC", "GridCompany", "HourlySettledConsumption" FROM  "consumptionpergridarea" WHERE "GridCompany" = '051' AND "HourUTC" >= '""" + date_start.strftime("%Y-%m-%dT%H:%M:%S") + """' AND "HourUTC" < '""" + date_end.strftime("%Y-%m-%dT%H:%M:%S") + """' """

api_call = requests.get(url)
df = pd.DataFrame(api_call.json()["result"]["records"])

#extract relevant columns
df.fillna(0, inplace=True)
pkl_name1 = "data/Frederikshavn_el_data_2021.pkl"
df.to_pickle(pkl_name1, protocol=3)
print('done')
# %%
