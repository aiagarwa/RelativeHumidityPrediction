import numpy as np
import pandas as pd
df = pd.read_csv("HumidityDataset.csv")
df.drop(["longitude","latitude", "WaveHeight", "WavePeriod", "MeanWaveDirection", "Hmax","QC_Flag"], inplace = True, axis = 1)
df = df.iloc[331371:]
buoy_ident = { 'M2':1 , 'M3': 2, 'M4':3, 'M5': 4, 'M6': 5}
df = df.loc[df.station_id.isin(buoy_ident.keys()) ]

df = df.drop(["time"], axis = 1)
df = df.replace({ 'station_id': buoy_ident})
df = df.dropna(how='all')
print(df.head())
print(df.corr())
