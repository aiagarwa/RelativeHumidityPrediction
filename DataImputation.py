import pandas as pd
import numpy as np
import missingno as msno
import matplotlib.pyplot as plt

# Read the data from file
df = pd.read_csv("WeatherData.csv")
print(df.shape[1])

# Filter the data from date 2010
df = df.iloc[331371:]

# Reset Index of the dataframe
df.reset_index(inplace=True)

# Visualize the missing data
msno.bar(df,figsize=(12, 6), fontsize=12, color='steelblue')
plt.show()

# Calculate and display missing data percentage
percent_missing = df.isnull().sum() * 100 / len(df)
missing_value_df = pd.DataFrame({'column_name': df.columns,
                                 'percent_missing': percent_missing})
print(missing_value_df)

# Drop the columns which are not required or which have missing data more than 50 per cent
df.drop(columns=["index","DewPoint","longitude","latitude","MeanWaveDirection","Hmax","QC_Flag"],inplace = True)

#  Filter the data only for 5 buouys
buoy_ident = { 'M2':1 , 'M3': 2, 'M4':3, 'M5': 4, 'M6': 5}
df = df.loc[df.station_id.isin(buoy_ident.keys()) ]

# Drop the rows which have missing Atmospheric Pressure value as missing data for Atmospheric Pressure is less than 5 per cent
df.dropna(subset=["AtmosphericPressure"],inplace = True) 

# Visualize the missing data again
msno.bar(df,figsize=(12, 6), fontsize=12, color='steelblue')
plt.show()
print(df)

# X=np.column_stack((df['AtmosphericPressure'],df['WindDirection'],df['WindSpeed'],df['Gust'],df['WaveHeight'],df['WavePeriod'],df['AirTemperature'],df['SeaTemperature'],df['RelativeHumidity']))
# X=np.column_stack((df['AirTemperature']))
X=df.copy()
X.drop(columns=["time","station_id"],inplace = True)
print(X.shape)

import time
start_time = time.time()

# Using Simple Imputer with strategy "Mean"
from sklearn.impute import SimpleImputer
imp_mean = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
imp_mean.fit(X)
X_trans=imp_mean.transform(X)

# # Using Iterative Imputer impute the missing data in all columns
# from sklearn.experimental import enable_iterative_imputer
# from sklearn.impute import IterativeImputer
# imp = IterativeImputer(imputation_order='ascending')
# imp.fit(X)
# X_trans=imp.transform(X)

print(X_trans)
end_time = time.time() - start_time
print(end_time)

# Copy the imputed data in data frames and save in csv file
df_new = pd.DataFrame(X_trans, columns = ['AtmosphericPressure','WindDirection','WindSpeed','Gust','WaveHeight','WavePeriod','AirTemperature','SeaTemperature','RelativeHumidity'])
df_final=pd.DataFrame()
df_final['station_id'] = df['station_id'].values
df_final['time'] = pd.Series(df['time'])
df_final = df_final.join(df_new)
print(df_final)
df_final.to_csv(r'ProcessedWeatherData_mostFrequent.csv', index = False, header=True)

# after Imputation again visualize the missing data
msno.bar(df_final,figsize=(12, 6), fontsize=12, color='steelblue')
plt.show()

