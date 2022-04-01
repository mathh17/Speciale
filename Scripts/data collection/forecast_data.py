#%%
from os import access
from grpc import access_token_call_credentials
import pandas as pd
from io import BytesIO
from osiris.azure_client_authorization import ClientAuthorization
from osiris.egress import Egress
from configparser import ConfigParser
#%%
config = ConfigParser()
config.read('conf.ini')

token = 'eyJ0eXAiOiJKV1QiLCJhbGciOiJSUzI1NiIsIng1dCI6ImpTMVhvMU9XRGpfNTJ2YndHTmd2UU8yVnpNYyIsImtpZCI6ImpTMVhvMU9XRGpfNTJ2YndHTmd2UU8yVnpNYyJ9.eyJhdWQiOiJodHRwczovL3N0b3JhZ2UuYXp1cmUuY29tIiwiaXNzIjoiaHR0cHM6Ly9zdHMud2luZG93cy5uZXQvZjc2MTkzNTUtNmM2Ny00MTAwLTlhNzgtMTg0N2YzMDc0MmUyLyIsImlhdCI6MTY0ODc5ODA2MywibmJmIjoxNjQ4Nzk4MDYzLCJleHAiOjE2NDg4MDMxODksImFjciI6IjEiLCJhaW8iOiJBU1FBMi84VEFBQUFPUmIyOGVudjhQbWplQ2IwRGpPK2hsdElNU0djOU5wSkpJRnZsZlE3cFJVPSIsImFtciI6WyJwd2QiXSwiYXBwaWQiOiJkOWNkNTIwZS0yMzE3LTRkYjYtYTVhZS03N2YwOTQ5MDg1YWYiLCJhcHBpZGFjciI6IjAiLCJmYW1pbHlfbmFtZSI6IsOYc3RlcmdhYXJkIEhhbnNlbiIsImdpdmVuX25hbWUiOiJNYXRoaWFzIiwiZ3JvdXBzIjpbImY4NzdiYTAzLTU5NmYtNDZjOS05MTVhLTc4ZmI0MjlmYjQ2YyIsImYwM2MyYWQ5LTExYTYtNDAyMS04MTNhLWIxMjQ3NDdkMDAwYSIsIjE5ZTIyMDU4LTk5N2UtNGNhYy04NDI1LWFjNTEwNGMxMjYyMiIsIjY5OWFkNGFhLWU3ZmEtNDNjZS04MzNlLTc4NWYyODcwZmY5OSIsImRjNzM2OWU2LThhZjQtNGE2Mi05NzVlLTgwMzUxYzQ1MDFjMyIsIjYxYzc0ZDE0LTk1NTMtNDM1MS1hZjRhLWRlMDNiOTJjMmZkYiIsIjE0OTU5ZDUxLWIwMzMtNGUyNS1hZDg1LTIxYTNkM2Q2ZDMyMiIsIjU1OWNkMjVlLWRkMTAtNGI2NS1iZDViLTQwYTBjNTZlNmM5ZCIsIjY0OWVkY2RkLThlN2MtNGIwZC1iYmM4LTEyMWEzM2U5YWQzNCIsIjZkOWU5YWVmLWYwMzctNGQ1OS05ZmVmLWZlNDVmMjcwMDNjYyIsImRhNjljMjJiLTFhYjAtNDBkYS05NTc5LTAwYjdjMmFlNTI1YyIsIjAyYWQ4YmNiLTYwYzYtNDk2OC1iYzUxLTIwNjdmMmFlODgxNiIsImRmMTM0YzJkLWYyM2EtNGNmZC1hMWM4LWViNTI4MzI1MmJkMCIsImQ3ZjI2YTczLTlhOTgtNDY0OC05NWJhLTQwMjVmNTQzZjVkZCIsIjg4NDJjNzY2LTQ3OGUtNGZjZC04OTAyLTRhYWQwYzJmN2JhNyIsIjA4ZWZmMDJlLTQxMDItNDVmNy1iMGI5LWE1Mjc1OGFjM2E2YSIsIjBjMGMzY2RmLTBiYTgtNDYwOC1hYjE2LTI5NmFiNmFlY2Y2OSIsIjcwYjhjYjdjLWNmOGMtNDgwOC04MjBiLWIxNDhiZjhiYTUyZiIsImFhNDJmYTAyLWIyMDItNGRiNS05NjM5LTIzNTliMjMzYTgyMiIsImJkOTExMmE3LTEyMDctNGFlNC1iZGY2LWM4YjE3ZWY0NmJiNyIsIjJhNTNhZmJkLTAwZDgtNDQ5NC05NzYyLWE4N2NlZTg5MWE2YSIsImE0OTUxOTg3LTUyYTUtNGVlZi04Y2UzLTU4Y2M3ODY4MDY0MiIsImI5NTA1ODMxLWExZmMtNGIyNi1iNmJjLTFjMTEzYjlhMWNlNyIsImIyOTVhZTgyLWI4NTMtNGE1OS04ZTA2LTVmZTVjYzBkMzJhZiIsIjRiNDhmZDY4LTZkNDMtNDdkYy04NDE4LTFmMzBmZTQ4ZWE0MSIsImY3ZmUyMmJiLTllMWUtNGRkZC04YTcwLTFjMTQ0YzFlNWZkZSIsIjc1NWM2N2YzLTA1NGMtNDU5MC1iNmVhLWU5ZmNiNjg1YTIyZiIsImJiY2VkZWI2LWRhMTctNGY4ZC1hZDdhLTg0ZWI2ODcxZmUxYyIsIjg3YmZiOTI0LWVhZmEtNGY0Yy1iMzczLWJiNjE4YzgzOTNkNyIsImY4YjgyNDk4LTJkMTgtNGRkYy05OGFiLTEzZjViNTBjODBhOCIsIjkyZTQ5YTI5LTk0ZGUtNDBiYi1hMTc5LTQwZGNiZDZiOWJjOSIsIjc1OWRmMzU5LTdmMDYtNGIzOS1iZDkzLTZmN2UxODAyYTBiYyIsIjhiOWU4ODZlLTM0MmItNDE0Yi05MDY1LTg4MGE4MzQ3ZjliZiIsIjhhMjdjOGJhLWZjMDItNDA4Ny04NTczLTRlNDA1ZTQxYjZiYyIsImQ3YTk4MzQzLTc2ZDMtNDE5My1hODAyLTZmN2QxMmE2MmYwZiIsIjNkMzg1NTY3LTE3NGQtNDFjMS05MjE2LTU0YTM0YmIwNjY2ZSIsImVlZGQyOTQ2LTAwYjYtNDFlNi1hNjUwLTY0NWI1ZTU3NDVjMyIsIjljMTYxZGMyLTQ3NTEtNGY3Ni05MDI2LTEzOTAwYzcwYzBjMiIsIjc2NDg0YmU5LTNhZTYtNDQ3Ny1iZDZhLWFhYzdkMjQ2Zjg0YyIsIjNlNDYwOWZkLTQ2NjgtNDlmZC1hZDdlLThiMDkxMjRhODkzMyIsIjQ3NGE1OTQ1LTM3MTAtNDA2Ny05ZWQwLWVkZjFlZGNjNWRhMCIsIjc2OGI0MTAxLTE4NGMtNGNjNC1hNGUyLTgzM2JhNGUxZTc1OCIsIjE5YzY4YTEyLTJhN2MtNDIyNS04ZjViLWYwMWYxOTZlMWY5OCIsImM2ODkyZGY0LTVjNDUtNGM3YS04MTQ4LWUxYjkxMTZiNjIwMiIsImQyNTJlYzAxLTcyZTgtNDg0Yi05MGRmLWY5ZDA4Yjk4ZTFjOSIsIjgyNzcyZmI1LWVmYjktNDY2ZS04NTRiLTA3ZjZkMTJjNTRlZSIsIjIzOWJhZGZhLTk4YjEtNGUyMS1iYjQwLWU4MjQxYzQzMzJjYSIsImY2MWRjNzFmLTE1OWYtNDA2Yy04YzUyLTQ3YzNiNmNiNjhmZSIsImYxZTVlYjJiLTcwNWYtNGY3ZC05ODcyLWY2MjhmZTY5NjE1NSIsIjliZjg1YTg0LTMyMDMtNDhhYy05MDMzLWIzZTllOGJlMWI1OSIsImI1ZTk3ZmQwLTE4YzMtNGM5Yy1hNDk0LTU1YWFhMzY5NGVjOSIsIjMyZDVkNTBlLWJlNmMtNDBhOC1hMDg1LWMyYzg1NjQ0ZDRhMyIsIjUwNjY5OTIyLWIyMjItNDU4YS1iNWMyLTI2NDA2MjdjMzNhZSIsIjE3NGFhZmYzLTNiOTYtNDY1Mi05ZjU0LTJlNzQ1YzliYzQwMyIsIjgwY2M3NGYzLTAyYTMtNDZkMS05NGU2LTRhNWI4MWE4OTY5MCIsIjA4YTI2ZDRkLWI0NmEtNGE2Ny1iNTY0LTYzOGJjYmU2YmY0ZiIsImUwZjNkOGNjLTE5MGYtNDIyYy05MTQ5LTA4OWY1ZWUwYjYzOSIsIjFkY2UyZWEzLWJmZTItNDYyZi04ZGYzLWZiNGI3MWI5NDFmMSIsIjIzYjk3ZjliLWU5NWQtNDBjYS04NDBmLTRmMmE0ZWY4N2RmNyIsImVkZjk4M2U3LTg4YTMtNDEzNS1hM2YyLTVlZDg1NDk0MGQ1NyIsIjIzYzlmNTllLWFlZTAtNGEyMC05NzA2LTNhNzg3Yzk1MmI3MSIsIjE4NTJhYzNjLWM3NGQtNDU4Yi04MjE5LTc0NTg1OGMzN2U3NyIsIjJiNjAwN2JkLTNiNjUtNDVlNy1hMzE1LWE1ODg1NGQ0MzY1MCIsImNlOTdhMDJkLTJiYjgtNGE5Yy04NmNiLWI2MWNhMDI3MzlkNCIsIjk1OTdlMjE3LTdlOGUtNGVkYS05NWFhLTNmNGQ5NzRiNzAyZCIsIjNlZjNmMWY3LWI5YTMtNDgxZS1hZDc2LTJkZWRiZWM0MzM1YSIsIjZlOGYxM2JkLWYwMGMtNDEyMS1hNmUzLWZlYThmMmU2ZjM4ZiIsIjU3OTgzNjQ2LWMxYmYtNDFiYS1iZTc3LWI4MGI5MGIzNDViNyIsIjkxNWEyMjVhLWU2ZWQtNDJjMi05ODg0LTFmZmYyMGJkOGQ5YSIsIjE0ODQyYTg4LThjZDUtNGViNi1iNTI5LTJkMDMwN2I4YjdlZiIsImU4YjVmMTY0LTcwNGQtNGE0NC05NzlkLTNlNjAzMTRhNjUyOSIsIjIxYWRkMmY3LTQ0ODAtNDU4Zi1hZDZiLTBjYTE4OTU2Y2NmZSIsIjg2YjkyYzA3LTdkYWEtNDY0OS04MTEyLTlhNmVmNTY5NjJmYiIsImZiZjEwZDFlLTVmZDQtNDgzYy04NmM1LWNhMDRhN2Y3ZjZlOCIsImY2ZmJkNGQwLWZlOTItNDkxMi05OTllLTdjYmU3YTI3MTY1MyIsIjVlYjE0YmNhLTY4NjItNGNjYi1iOGFlLThiNThlYmU2ZDBmOCIsIjhkZTFkNWMzLThjZjItNDdhZC1hNTcwLTg5MzdmZDUzM2RhNCIsIjk4YmQ0ODJiLTczOTQtNGE4Zi04MDIyLWY5OGFjMDVjMjU3ZiIsIjQ5MWQ1MTIyLWFkNjMtNDA1Ny05MWQ0LTZkY2ViZGViZTMwZCIsIjczMjRjNWQwLWU2NzYtNDFlOC04Yjg1LWFjZDIzNTljOWFhNSIsIjlkYmIwNzliLWJjOGMtNGNjOS1iZjk1LWMxNzI5M2I5ZTVjZiIsImMwNTFjYzlhLTZmM2EtNDg2MC1hNzRmLTllOWZiYjdhN2ExOCIsImVmY2RmNTFhLWQ4YWYtNGE3MC04YmU4LWRiMDZjZTIwNDE0YyIsIjRkOWM5YjFkLWNlYzYtNGZhNy05ZWE1LWMxYTgwZGYyNWFmMSIsIjI1OTk3Y2NiLWE2ZWYtNGQ1Zi1iYjE3LTRiYzBmMzliYmE2ZiIsImEyNWUwN2Y2LTA5MDItNDUxMC05MjRlLTA0MjNjMzU0NTVlZSIsIjU3NDUxZWRhLThhOGYtNDQ1YS04YmNkLThmMzk1ZTVlMDc0OSIsIjg4YjQ5ZDg0LTc5YjktNGVlOS05ZDQ3LTk0ZmI5YTRiZmUzMyIsIjdmM2Y1NGYxLWNkNTYtNDZkMi1iMWI2LTFjNjZiMzNmM2VmMSIsIjNjNGJkOTg4LTU3NmYtNGNlOS1iNzhiLWVhZWU3N2RiNGNkYSIsIjM4NDFhOWFmLTczMjEtNDJjYi1iMzU4LWJhZTMxNTE5NDQzOCIsImY4NmU2OTExLWFmMDEtNGZiYS1hYWNjLWQ0MzNiZGU5ZGVjYiJdLCJpcGFkZHIiOiIyLjEwNC4yMjQuMTAwIiwibmFtZSI6Ik1hdGhpYXMgw5hzdGVyZ2FhcmQgSGFuc2VuIiwib2lkIjoiYjdmY2ZjZWQtYmViMi00NWFkLWE5ZmYtNjc4ZjZiOTRkZDIxIiwib25wcmVtX3NpZCI6IlMtMS01LTIxLTI5MDE0ODY1NzQtMjE5NDc1NDQ4Ni0xMDI1NTQyNDUwLTEwMTA1MyIsInB1aWQiOiIxMDAzMjAwMUQ5OTNEQTVFIiwicmgiOiIwLkFRc0FWWk5oOTJkc0FFR2FlQmhIOHdkQzRvR21CdVRVODZoQ2tMYkNzQ2xKZXZFTEFFMC4iLCJzY3AiOiJ1c2VyX2ltcGVyc29uYXRpb24iLCJzdWIiOiJqelkzQXVJQ3VRT2JfUV9HZ2xHMVBrVUhSVW5HZmMzbGNMdjY3cGRndlprIiwidGlkIjoiZjc2MTkzNTUtNmM2Ny00MTAwLTlhNzgtMTg0N2YzMDc0MmUyIiwidW5pcXVlX25hbWUiOiJNVEdAZW5lcmdpbmV0LmRrIiwidXBuIjoiTVRHQGVuZXJnaW5ldC5kayIsInV0aSI6IjFLVS02VTRpRUVPYTctODVEbG9HQUEiLCJ2ZXIiOiIxLjAifQ.Jb-SP0mBLWDT3i8BGgLBanu4RU8KZWhD5pcCUTCVlCdWjYFAiCn8M1lYTGkhxpG518zf8nZTyxLJC8zc2KXIN1guw5bTVo3wnyA2Q41kCp2z5pF8Hi10tyJVQwGeYw-bqQJHEbGKPJT-l6wJkFNQy_frxVPuzFoTC4GuXlbkKoacwqENIEofvkv5uA8xKaWQ9iDjjMWonoezvtjE_009nIjimuQdTq38OInqu7cuam-P540-TTO2JTVRbF6fg096SC8SobtHo6pEpXSyXy7loOfewMd_aaoc0yWvqou_2hXkLsjT-9xlTRpSsekz_ZV0_pKRxJkHUr9AuPC28a_fow'
#%%
client_auth = ClientAuthorization(access_token=token)

egress = Egress(client_auth=client_auth,
                egress_url=config['Egress']['url'])

#%%
coords = egress.download_dmi_list(from_date='2019-01')
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
Divides the stations from the api request into dk1 and dk2 zones.
Uses the longitude and latitude to divide them.   

"""
dk1_stations = []
dk2_stations = []

for station in coords:
    if station['lat'] is not None: 
        if station['lat'] < lat_max_dk1 and station['lat'] > lat_min_dk1: 
            if station['lon'] < lon_max_dk1 and station['lon'] > lon_min_dk1:
                dk1_stations.append(station)
        if station['lat'] < lat_max_dk2 and station['lat'] > lat_min_dk2:
            if station['lon'] < lon_max_dk2 and station['lon'] > lon_min_dk2:
                dk2_stations.append(station)

#%%
stations_concat_df = pd.DataFrame()
for station in dk2_stations:
    parquet_content = egress.download_dmi_file(lon=station['lon'], lat=station['lat'],
                                            from_date='2021-01', 
                                            to_date='2021-03')
    data = pd.read_parquet(BytesIO(parquet_content))
    data_temp_val = get_station_temp_val(data)
    data_radi_val = get_station_radi_val(data)
    stations_concat_df = pd.merge(data_temp_val,data_radi_val, on='time')


#%%
parquet_content = egress.download_dmi_file(lon=15.19, lat=55.00,
                                            from_date='2021-01',
                                            to_date='2021-03',)
data = pd.read_parquet(BytesIO(parquet_content))
data.head()
#%%
def get_station_temp_val(station):
    values = []
    time = []
    predicted = []
    for index, row  in station.iterrows():
        if row['weather_type'] == 'temperatur_2m' and row['predicted_ahead']  in [1,2,3]:
            values.append(row['value'])
            time.append(row['Date'])
            predicted.append(row['predicted_ahead'])
        
    stations_df = pd.DataFrame(columns=['temp_mean_1hr','predicted_ahead','time'])
    stations_df['temp_mean_1hr'] = values
    stations_df['time'] = time
    stations_df['predicted_ahead'] = predicted
    return stations_df
#%%
def get_station_radi_val(station):
    values = []
    time = []
    predicted = []
    for index, row  in station.iterrows():
        if row['weather_type'] == 'radiation_hour' and row['predicted_ahead'] in [1,2,3]:
            values.append(row['value'])
            time.append(row['Date'])
            predicted.append(row['predicted_ahead'])
    stations_df = pd.DataFrame(columns=['radiation_hour','predicted_ahead','time'])
    stations_df['radiation_hour'] = values
    stations_df['time'] = time
    stations_df['predicted_ahead'] = predicted
    return stations_df

# %%
"""
Henter forecast data fra stationen: Jægersborg.
Jægersborg tilhører grid companiet Radius Elnet. 
Fører det sammen i et datasæt og omregner temperaturen fra Kelvin Celsius
"""
parquet_content = egress.download_dmi_file(lon=12.55, lat=55.7,
                                            from_date='2020-01-01T00', 
                                            to_date='2020-05-01T00')
data = pd.read_parquet(BytesIO(parquet_content))
data_temp_val = get_station_temp_val(data)
data_radi_val = get_station_radi_val(data)

#%%
"""
Henter forecast data fra stationen: Fyn
Fyn tilhører grid companiet Vores Elnet. 
Fører det sammen i et datasæt og omregner temperaturen fra Kelvin Celsius
"""
parquet_content = egress.download_dmi_file(lon=10.40, lat=55.38,
                                            from_date='2020-01-01T00', 
                                            to_date='2020-05-01T00')
data = pd.read_parquet(BytesIO(parquet_content))
data_temp_val = get_station_temp_val(data)
data_radi_val = get_station_radi_val(data)

#%%
pkl_name = "data/forecast_data_dk2"+station_row+"dmi_data.pkl"
pd.to_pickle
# %%
