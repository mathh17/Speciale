
#%%
from requests import get
from pandas import DataFrame, to_datetime, merge
from datetime import datetime, timedelta
from math import floor
from numpy import timedelta64
import matplotlib.pyplot as plt

#%%
api_key = "46481bd5-99c3-45df-b1b1-f2743b5277cf"
date_start = datetime(2019,12,1)
date_end = datetime(2019,12,5)
stationId = '06072'
params = ['temp_mean_past1h','precip_past1h', 'wind_speed_past1h']

#%%
data = DataFrame()
for param in params:
    url = 'https://dmigw.govcloud.dk/v2/metObs/collections/observation/items' # url for the current api version    
    url_params = {'api-key' : api_key,
            'datetime' : '2020-05-01T00:00:00Z/2020-05-02T00:00:00Z',
            'stationId' : stationId, #'06068', '06174'
            'parameterId' : param,
            "timeResolution" : "hour"
            'limit' : '300000',
            } 
    r = get(url, params=url_params) # Issues a HTTP GET request
    readings = r.json()
    data[param] = readings['features'] 



#%%
val = DataFrame()
for param in params:
    for line in range(len(data)):
        features = data[param][line]['properties']['value']
        val[param] += features


#%%


plt.plot(val)

# %%
