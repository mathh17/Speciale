#%%
from os import access
import pandas as pd
import holidays as hc
from io import BytesIO
from osiris.core.azure_client_authorization import ClientAuthorization
from osiris.apis.egress import Egress
from configparser import ConfigParser

#%%
config = ConfigParser()
config.read('conf.ini')

token = 'eyJ0eXAiOiJKV1QiLCJhbGciOiJSUzI1NiIsIng1dCI6ImpTMVhvMU9XRGpfNTJ2YndHTmd2UU8yVnpNYyIsImtpZCI6ImpTMVhvMU9XRGpfNTJ2YndHTmd2UU8yVnpNYyJ9.eyJhdWQiOiJodHRwczovL3N0b3JhZ2UuYXp1cmUuY29tIiwiaXNzIjoiaHR0cHM6Ly9zdHMud2luZG93cy5uZXQvZjc2MTkzNTUtNmM2Ny00MTAwLTlhNzgtMTg0N2YzMDc0MmUyLyIsImlhdCI6MTY0OTY3ODM4MSwibmJmIjoxNjQ5Njc4MzgxLCJleHAiOjE2NDk2ODM1MDMsImFjciI6IjEiLCJhaW8iOiJFMlpnWUFnS3FkWlJGMkprVmRoei9zQ3ZCSkZNL2I4WlBwTlZsWTVKR0Yvc3FyeTM3RFlBIiwiYW1yIjpbInB3ZCIsInJzYSJdLCJhcHBpZCI6ImQ5Y2Q1MjBlLTIzMTctNGRiNi1hNWFlLTc3ZjA5NDkwODVhZiIsImFwcGlkYWNyIjoiMCIsImRldmljZWlkIjoiMzExMjdiYjEtZjk4Yy00Y2I5LWIzN2EtYmVkNjc0YzFlMjUxIiwiZmFtaWx5X25hbWUiOiLDmHN0ZXJnYWFyZCBIYW5zZW4iLCJnaXZlbl9uYW1lIjoiTWF0aGlhcyIsImdyb3VwcyI6WyJmODc3YmEwMy01OTZmLTQ2YzktOTE1YS03OGZiNDI5ZmI0NmMiLCJmMDNjMmFkOS0xMWE2LTQwMjEtODEzYS1iMTI0NzQ3ZDAwMGEiLCIxOWUyMjA1OC05OTdlLTRjYWMtODQyNS1hYzUxMDRjMTI2MjIiLCI2OTlhZDRhYS1lN2ZhLTQzY2UtODMzZS03ODVmMjg3MGZmOTkiLCJkYzczNjllNi04YWY0LTRhNjItOTc1ZS04MDM1MWM0NTAxYzMiLCI2MWM3NGQxNC05NTUzLTQzNTEtYWY0YS1kZTAzYjkyYzJmZGIiLCIxNDk1OWQ1MS1iMDMzLTRlMjUtYWQ4NS0yMWEzZDNkNmQzMjIiLCI1NTljZDI1ZS1kZDEwLTRiNjUtYmQ1Yi00MGEwYzU2ZTZjOWQiLCI2NDllZGNkZC04ZTdjLTRiMGQtYmJjOC0xMjFhMzNlOWFkMzQiLCI2ZDllOWFlZi1mMDM3LTRkNTktOWZlZi1mZTQ1ZjI3MDAzY2MiLCJkYTY5YzIyYi0xYWIwLTQwZGEtOTU3OS0wMGI3YzJhZTUyNWMiLCIwMmFkOGJjYi02MGM2LTQ5NjgtYmM1MS0yMDY3ZjJhZTg4MTYiLCJkZjEzNGMyZC1mMjNhLTRjZmQtYTFjOC1lYjUyODMyNTJiZDAiLCJkN2YyNmE3My05YTk4LTQ2NDgtOTViYS00MDI1ZjU0M2Y1ZGQiLCI4ODQyYzc2Ni00NzhlLTRmY2QtODkwMi00YWFkMGMyZjdiYTciLCIwOGVmZjAyZS00MTAyLTQ1ZjctYjBiOS1hNTI3NThhYzNhNmEiLCIwYzBjM2NkZi0wYmE4LTQ2MDgtYWIxNi0yOTZhYjZhZWNmNjkiLCI3MGI4Y2I3Yy1jZjhjLTQ4MDgtODIwYi1iMTQ4YmY4YmE1MmYiLCJhYTQyZmEwMi1iMjAyLTRkYjUtOTYzOS0yMzU5YjIzM2E4MjIiLCJiZDkxMTJhNy0xMjA3LTRhZTQtYmRmNi1jOGIxN2VmNDZiYjciLCIyYTUzYWZiZC0wMGQ4LTQ0OTQtOTc2Mi1hODdjZWU4OTFhNmEiLCJhNDk1MTk4Ny01MmE1LTRlZWYtOGNlMy01OGNjNzg2ODA2NDIiLCJiOTUwNTgzMS1hMWZjLTRiMjYtYjZiYy0xYzExM2I5YTFjZTciLCJiMjk1YWU4Mi1iODUzLTRhNTktOGUwNi01ZmU1Y2MwZDMyYWYiLCI0YjQ4ZmQ2OC02ZDQzLTQ3ZGMtODQxOC0xZjMwZmU0OGVhNDEiLCJmN2ZlMjJiYi05ZTFlLTRkZGQtOGE3MC0xYzE0NGMxZTVmZGUiLCI3NTVjNjdmMy0wNTRjLTQ1OTAtYjZlYS1lOWZjYjY4NWEyMmYiLCJiYmNlZGViNi1kYTE3LTRmOGQtYWQ3YS04NGViNjg3MWZlMWMiLCI4N2JmYjkyNC1lYWZhLTRmNGMtYjM3My1iYjYxOGM4MzkzZDciLCJmOGI4MjQ5OC0yZDE4LTRkZGMtOThhYi0xM2Y1YjUwYzgwYTgiLCI5MmU0OWEyOS05NGRlLTQwYmItYTE3OS00MGRjYmQ2YjliYzkiLCI3NTlkZjM1OS03ZjA2LTRiMzktYmQ5My02ZjdlMTgwMmEwYmMiLCI4YjllODg2ZS0zNDJiLTQxNGItOTA2NS04ODBhODM0N2Y5YmYiLCI4YTI3YzhiYS1mYzAyLTQwODctODU3My00ZTQwNWU0MWI2YmMiLCJkN2E5ODM0My03NmQzLTQxOTMtYTgwMi02ZjdkMTJhNjJmMGYiLCIzZDM4NTU2Ny0xNzRkLTQxYzEtOTIxNi01NGEzNGJiMDY2NmUiLCJlZWRkMjk0Ni0wMGI2LTQxZTYtYTY1MC02NDViNWU1NzQ1YzMiLCI5YzE2MWRjMi00NzUxLTRmNzYtOTAyNi0xMzkwMGM3MGMwYzIiLCI3NjQ4NGJlOS0zYWU2LTQ0NzctYmQ2YS1hYWM3ZDI0NmY4NGMiLCIzZTQ2MDlmZC00NjY4LTQ5ZmQtYWQ3ZS04YjA5MTI0YTg5MzMiLCI0NzRhNTk0NS0zNzEwLTQwNjctOWVkMC1lZGYxZWRjYzVkYTAiLCI3NjhiNDEwMS0xODRjLTRjYzQtYTRlMi04MzNiYTRlMWU3NTgiLCIxOWM2OGExMi0yYTdjLTQyMjUtOGY1Yi1mMDFmMTk2ZTFmOTgiLCJjNjg5MmRmNC01YzQ1LTRjN2EtODE0OC1lMWI5MTE2YjYyMDIiLCJkMjUyZWMwMS03MmU4LTQ4NGItOTBkZi1mOWQwOGI5OGUxYzkiLCI4Mjc3MmZiNS1lZmI5LTQ2NmUtODU0Yi0wN2Y2ZDEyYzU0ZWUiLCIyMzliYWRmYS05OGIxLTRlMjEtYmI0MC1lODI0MWM0MzMyY2EiLCJmNjFkYzcxZi0xNTlmLTQwNmMtOGM1Mi00N2MzYjZjYjY4ZmUiLCJmMWU1ZWIyYi03MDVmLTRmN2QtOTg3Mi1mNjI4ZmU2OTYxNTUiLCI5YmY4NWE4NC0zMjAzLTQ4YWMtOTAzMy1iM2U5ZThiZTFiNTkiLCJiNWU5N2ZkMC0xOGMzLTRjOWMtYTQ5NC01NWFhYTM2OTRlYzkiLCIzMmQ1ZDUwZS1iZTZjLTQwYTgtYTA4NS1jMmM4NTY0NGQ0YTMiLCI1MDY2OTkyMi1iMjIyLTQ1OGEtYjVjMi0yNjQwNjI3YzMzYWUiLCIxNzRhYWZmMy0zYjk2LTQ2NTItOWY1NC0yZTc0NWM5YmM0MDMiLCI4MGNjNzRmMy0wMmEzLTQ2ZDEtOTRlNi00YTViODFhODk2OTAiLCIwOGEyNmQ0ZC1iNDZhLTRhNjctYjU2NC02MzhiY2JlNmJmNGYiLCJlMGYzZDhjYy0xOTBmLTQyMmMtOTE0OS0wODlmNWVlMGI2MzkiLCIxZGNlMmVhMy1iZmUyLTQ2MmYtOGRmMy1mYjRiNzFiOTQxZjEiLCIyM2I5N2Y5Yi1lOTVkLTQwY2EtODQwZi00ZjJhNGVmODdkZjciLCJlZGY5ODNlNy04OGEzLTQxMzUtYTNmMi01ZWQ4NTQ5NDBkNTciLCIyM2M5ZjU5ZS1hZWUwLTRhMjAtOTcwNi0zYTc4N2M5NTJiNzEiLCIxODUyYWMzYy1jNzRkLTQ1OGItODIxOS03NDU4NThjMzdlNzciLCIyYjYwMDdiZC0zYjY1LTQ1ZTctYTMxNS1hNTg4NTRkNDM2NTAiLCJjZTk3YTAyZC0yYmI4LTRhOWMtODZjYi1iNjFjYTAyNzM5ZDQiLCI5NTk3ZTIxNy03ZThlLTRlZGEtOTVhYS0zZjRkOTc0YjcwMmQiLCIzZWYzZjFmNy1iOWEzLTQ4MWUtYWQ3Ni0yZGVkYmVjNDMzNWEiLCI2ZThmMTNiZC1mMDBjLTQxMjEtYTZlMy1mZWE4ZjJlNmYzOGYiLCI1Nzk4MzY0Ni1jMWJmLTQxYmEtYmU3Ny1iODBiOTBiMzQ1YjciLCI5MTVhMjI1YS1lNmVkLTQyYzItOTg4NC0xZmZmMjBiZDhkOWEiLCIxNDg0MmE4OC04Y2Q1LTRlYjYtYjUyOS0yZDAzMDdiOGI3ZWYiLCJlOGI1ZjE2NC03MDRkLTRhNDQtOTc5ZC0zZTYwMzE0YTY1MjkiLCIyMWFkZDJmNy00NDgwLTQ1OGYtYWQ2Yi0wY2ExODk1NmNjZmUiLCI4NmI5MmMwNy03ZGFhLTQ2NDktODExMi05YTZlZjU2OTYyZmIiLCJmYmYxMGQxZS01ZmQ0LTQ4M2MtODZjNS1jYTA0YTdmN2Y2ZTgiLCJmNmZiZDRkMC1mZTkyLTQ5MTItOTk5ZS03Y2JlN2EyNzE2NTMiLCI1ZWIxNGJjYS02ODYyLTRjY2ItYjhhZS04YjU4ZWJlNmQwZjgiLCI4ZGUxZDVjMy04Y2YyLTQ3YWQtYTU3MC04OTM3ZmQ1MzNkYTQiLCI5OGJkNDgyYi03Mzk0LTRhOGYtODAyMi1mOThhYzA1YzI1N2YiLCI0OTFkNTEyMi1hZDYzLTQwNTctOTFkNC02ZGNlYmRlYmUzMGQiLCI3MzI0YzVkMC1lNjc2LTQxZTgtOGI4NS1hY2QyMzU5YzlhYTUiLCI5ZGJiMDc5Yi1iYzhjLTRjYzktYmY5NS1jMTcyOTNiOWU1Y2YiLCJjMDUxY2M5YS02ZjNhLTQ4NjAtYTc0Zi05ZTlmYmI3YTdhMTgiLCJlZmNkZjUxYS1kOGFmLTRhNzAtOGJlOC1kYjA2Y2UyMDQxNGMiLCI0ZDljOWIxZC1jZWM2LTRmYTctOWVhNS1jMWE4MGRmMjVhZjEiLCIyNTk5N2NjYi1hNmVmLTRkNWYtYmIxNy00YmMwZjM5YmJhNmYiLCJhMjVlMDdmNi0wOTAyLTQ1MTAtOTI0ZS0wNDIzYzM1NDU1ZWUiLCI1NzQ1MWVkYS04YThmLTQ0NWEtOGJjZC04ZjM5NWU1ZTA3NDkiLCI4OGI0OWQ4NC03OWI5LTRlZTktOWQ0Ny05NGZiOWE0YmZlMzMiLCI3ZjNmNTRmMS1jZDU2LTQ2ZDItYjFiNi0xYzY2YjMzZjNlZjEiLCIzYzRiZDk4OC01NzZmLTRjZTktYjc4Yi1lYWVlNzdkYjRjZGEiLCIzODQxYTlhZi03MzIxLTQyY2ItYjM1OC1iYWUzMTUxOTQ0MzgiLCJmODZlNjkxMS1hZjAxLTRmYmEtYWFjYy1kNDMzYmRlOWRlY2IiXSwiaXBhZGRyIjoiMi4xMDQuMjI0LjEwMCIsIm5hbWUiOiJNYXRoaWFzIMOYc3RlcmdhYXJkIEhhbnNlbiIsIm9pZCI6ImI3ZmNmY2VkLWJlYjItNDVhZC1hOWZmLTY3OGY2Yjk0ZGQyMSIsIm9ucHJlbV9zaWQiOiJTLTEtNS0yMS0yOTAxNDg2NTc0LTIxOTQ3NTQ0ODYtMTAyNTU0MjQ1MC0xMDEwNTMiLCJwdWlkIjoiMTAwMzIwMDFEOTkzREE1RSIsInJoIjoiMC5BUXNBVlpOaDkyZHNBRUdhZUJoSDh3ZEM0b0dtQnVUVTg2aENrTGJDc0NsSmV2RUxBRTAuIiwic2NwIjoidXNlcl9pbXBlcnNvbmF0aW9uIiwic3ViIjoianpZM0F1SUN1UU9iX1FfR2dsRzFQa1VIUlVuR2ZjM2xjTHY2N3BkZ3ZaayIsInRpZCI6ImY3NjE5MzU1LTZjNjctNDEwMC05YTc4LTE4NDdmMzA3NDJlMiIsInVuaXF1ZV9uYW1lIjoiTVRHQGVuZXJnaW5ldC5kayIsInVwbiI6Ik1UR0BlbmVyZ2luZXQuZGsiLCJ1dGkiOiJfdnVfYnpGYmdrcXU0eTV4X2UwaEFBIiwidmVyIjoiMS4wIn0.Wio_veedgz5E2Q-JjrWNV-ndeGoQixwnplahdmZ2jghsn10XBKJGWVWaFuoVn1BKeR4Gnr5seIR8GMlu8gEq9hq3IiWNmLue2Gg3MpvHOSegTxL0aR-AgIF_ES2d7ytL6jfk2Wn5NdXNE3sG64XSpjzXRJP2UWAgj0BCak7QploylJS_SEQqywYgj8yP-gdlY8z3aaAOmD9IAV6O_M2KJ99A-5ljkApeU0k02AhtuAiMeH8nu6BrPg7gYXtlTTO2VJ8pV6ldzfhxf0boC8VSmzbbel2q7MZIObwYXwPvWr3fGBSroUvKiATyldEOxny2w4O53hK3q_2daT9F7hbDPQ'

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
"""
Get temperature data from all stations in a list of stations
"""
def get_station_data(stations,feature):
    stations_df = pd.DataFrame()
    station_counter = 0
    for station in stations:
        parquet_content = egress.download_dmi_file(lon=station['lon'], lat=station['lat'],
                                            from_date='2019-01', 
                                            to_date='2020-01')
        data = pd.read_parquet(BytesIO(parquet_content))
        
        stations_group = data.groupby("weather_type")
        weather_df = stations_group.get_group(feature)
        print(station_counter)
        if station_counter == 0:
            stations_df = weather_df.filter(['Date','value','predicted_ahead'],axis=1)
        if station_counter > 0:
            stations_df = pd.merge(stations_df,weather_df[['Date','predicted_ahead','value']],on=['Date','predicted_ahead'],how='inner')
        #stations_df["temp_"+str(station_counter)] = weather_df['value']
        station_counter += 1
    #stations_df
    return stations_df

#%%
"""
Collects all data from dk2
Makes a mean of the data from all stations at the given time
Then merges its into one DF
"""
dk2_radiation = get_station_data(dk2_stations, 'radiation_hour')
dk2_radiation_mean = dk2_radiation.mean(axis=1)

dk2_temp = get_station_data(dk2_stations, 'temperatur_2m')
dk2_temp_mean = dk2_temp.mean(axis=1)


dk2_radiation = dk2_radiation[['Date','predicted_ahead']]
dk2_radiation['mean_radi'] = dk2_radiation_mean
dk2_radiation['predicted_ahead'] =dk2_radiation['predicted_ahead'] + 1

dk2_temp = dk2_temp[['Date','predicted_ahead']]
dk2_temp['mean_temp'] = dk2_temp_mean


dk2_forecast_data = pd.DataFrame()
dk2_forecast_data = pd.merge(dk2_radiation,dk2_temp,on=['Date','predicted_ahead'],how='inner')

#%%
#pkl_name = "dk2_forecast_data.pkl"
dk2_forecast_data.to_parquet('dk2_forecast_data')

#%%
#%%
"""
Collects all data from dk1
Makes a mean of the data from all stations at the given time
Then merges its into one DF
"""
dk1_radiation = get_station_data(dk1_stations, 'radiation_hour')
dk1_radiation_mean = dk1_radiation.mean(axis=1)

dk1_temp = get_station_data(dk1_stations, 'temperatur_2m')
dk1_temp_mean = dk1_temp.mean(axis=1)


dk1_radiation = dk1_radiation[['Date','predicted_ahead']]
dk1_radiation['mean_radi'] = dk1_radiation_mean
dk1_radiation['predicted_ahead'] =dk1_radiation['predicted_ahead'] + 1

dk1_temp = dk1_temp[['Date','predicted_ahead']]
dk2_temp['mean_temp'] = dk2_temp_mean


dk1_forecast_data = pd.DataFrame()
dk1_forecast_data = pd.merge(dk1_radiation,dk1_temp,on=['Date','predicted_ahead'],how='inner')

#%%
#pkl_name = "dk2_forecast_data.pkl"
dk1_forecast_data.to_parquet('dk1_forecast_data')


























