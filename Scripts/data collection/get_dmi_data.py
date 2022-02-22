# -*- coding: utf-8 -*-

#%%
"""
import packages
"""
import requests
import pandas as pd

#%% 
"""
List of parameters to include in request
"""
parameters_list = ['temp_mean_past1h','radia_glob_past1h']
url = "https://dmigw.govcloud.dk/v2/metObs/collections/observation/"
items = "items?"
resolution = "timeResolution=hour"
datetime = "datetime="
limit = "limit=300000"
start_date = "2010-01-01T00:00:00Z/"
end_date = "2021-12-31T23:00:00Z"
parameters = "parameterId="
key = "api-key=46481bd5-99c3-45df-b1b1-f2743b5277cf"
#%%
"""
min and max values for latitude and longitude for DK1 and DK2 stored in variables
"""
#DK1
lat_min_dk1 = 54.8
lat_max_dk1 = 57.6
lon_max_dk1 = 11
lon_min_dk1 = 8.1
 
# DK2
lon_min_dk2 = 10.9
lon_max_dk2 = 12.8
lat_max_dk2 = 56.1
lat_min_dk2 = 55

# %%
"""
API request for list of stations and their ID's
df stations: dataframe of stations, with id's and coordinates
"""
station_request = requests.get('https://dmigw.govcloud.dk/v2/metObs/collections/station/items?'+key)
station_request_list = station_request.json()['features']
stations_list = []
regions_list = []
coord_list = []
for data in station_request_list:
    if data['properties']['stationId'] not in stations_list:
            stations_list.append(data['properties']['stationId'])
            coord_list.append(data['geometry']['coordinates'])


stations = pd.DataFrame()
stations['stationID'] = stations_list
stations['coords'] = coord_list
# %%
"""
Divides the stations from the api request into dk1 and dk2 zones.
Uses the longitude and latitude to divide them.   

"""
dk1_stations = []
dk2_stations = []

for index, station in stations.iterrows():
    if station['coords'][1] is not None: 
        if station['coords'][1] < lat_max_dk1 and station['coords'][1] > lat_min_dk1: 
            if station['coords'][0] < lon_max_dk1 and station['coords'][0] > lon_min_dk1:
                dk1_stations.append(station['stationID'])
        if station['coords'][1] < lat_max_dk2 and station['coords'][1] > lat_min_dk2:
            if station['coords'][0] < lon_max_dk2 and station['coords'][0] > lon_min_dk2:
                dk2_stations.append(station['stationID'])


#%%
"""
Request for DMI data via API store in pkl file. 

Variables:
str url_call: URL string for api request
df df: temporary dataframe to store paramters before its stored in the dmi_data dataframe
df request: stores the data from api-call
list dates: list of dates
list values: list of observation values
list  stations_ids:list of station ID's
output:
pkl file: stored in data/dmi_data.pkl
df dmi_data: stores all data from one station
"""
for station_row in dk1_stations: 
    dmi_data = pd.DataFrame(columns=['time'])
    for param in parameters_list:
        df = pd.DataFrame()
        url_call = url + items + "stationId=" + station_row + '&' + datetime + start_date
        url_call += end_date + '&' + limit + '&' + parameters
        url_call += param + '&' + key

        print(url_call)
        response = requests.get(url_call)

        dates = []
        values = []
        station_ids = []
        data_list = response.json()['features']
        for data in data_list:
            date_to = pd.to_datetime(data['properties']['observed']).astimezone('UTC')
            dates.append(date_to)
            values.append(data['properties']['value'])
            station_ids.append(station_row)
        
        df["time"] = dates
        df[param] = values
        if len(df.index) is not 0:
            dmi_data = pd.merge(dmi_data, df, on='time', how='outer')

    pkl_name = ""+station_row+"dmi_data.pkl"
    dmi_data.to_pickle(pkl_name)

#%%
"""
Request for DMI data in the dk2 zones via API store in pkl file. 

Variables:
str url_call: URL string for api request
df df: temporary dataframe to store paramters before its stored in the dmi_data dataframe
df request: stores the data from api-call
list dates: list of dates
list values: list of observation values
list  stations_ids:list of station ID's
output:
pkl file: stored in data/dmi_data.pkl
df dmi_data: stores all data from one station
"""
for station_row in dk2_stations: 
    dmi_data = pd.DataFrame(columns=['time'])
    for param in parameters_list:
        df = pd.DataFrame()
        url_call = url + items + "stationId=" + station_row + '&' + datetime + start_date
        url_call += end_date + '&' + limit + '&' + parameters
        url_call += param + '&' + key

        print(url_call)
        response = requests.get(url_call)

        dates = []
        values = []
        station_ids = []
        data_list = response.json()['features']
        for data in data_list:
            date_to = pd.to_datetime(data['properties']['observed']).astimezone('UTC')
            dates.append(date_to)
            values.append(data['properties']['value'])
            station_ids.append(station_row)
        
        df["time"] = dates
        df[param] = values
        if len(df.index) is not 0:
            dmi_data = pd.merge(dmi_data, df, on='time', how='outer')

    pkl_name = ""+station_row+"dmi_data.pkl"
    dmi_data.to_pickle(pkl_name, protocol=3)
# %%
