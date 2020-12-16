#!/usr/bin/env python
# coding: utf-8

# # Importing the Data and The Libraries

# In[1]:


import numpy as np
import pandas as pd 
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from tensorflow.keras import regularizers
from tensorflow.keras import initializers
from tensorflow.keras.callbacks import EarlyStopping


df = pd.read_csv("HumidityDataset.csv")


# ## Dropping all the libraires and selecting only rows that start after the 2010



df.drop(["longitude","latitude", "WaveHeight", "WavePeriod", "MeanWaveDirection", "Hmax","QC_Flag"],inplace = True, axis = 1)
df = df.iloc[331371:]
buoy_ident = { 'M2':1 , 'M3': 2, 'M4':3, 'M5': 4, 'M6': 5}
# df.station_id.nunique()




# Selecting only the values of rows that have buoys ident as M2 to M6
df = df.loc[df.station_id.isin(buoy_ident.keys()) ]
df = df.drop(["time"], axis = 1)
# df.station_id.unique()


# In[4]:


df = df.replace({ 'station_id': buoy_ident})
df = df.dropna(axis = 1, how='all')
df.reset_index(inplace = True)
df.dropna(inplace = True, how = 'all')
# df.head()


# In[5]:


df = df.dropna()
df.drop('index', axis = 1, inplace = True)
# df.reset_index(inplace= True)
# df.isna()
# drop('index',axis = 1, inplace = True)


# In[27]:


X_train1, X_test1, y_train1, y_test1 = train_test_split(df.iloc[:,:-1].values,df.iloc[:,-1].values, test_size=0.2)

class CustomScaler():
    
        def __init__(self, X_Scale, y_Scale ) :
        
            # self.X_Scale =   X_Scale
            # self.y_Scale =   y_Scale

            self.x_maxs =    [np.max(X_Scale[:,i]) for i in range(X_Scale.shape[1])]
            self.x_mins =    [np.min(X_Scale[:,i]) for i in range(X_Scale.shape[1])]
#             self.y_mean =    np.mean(self.y_train)
            self.y_min =    np.min(y_Scale)
            self.y_max_min = np.max(y_Scale) - np.min(y_Scale)

        def scaleX(self, x_value):
            x = x_value
            for i in range(x.shape[1]):
                
                x[:,i]= (x[:,i] - self.x_mins[i])/(self.x_maxs[i]-self.x_mins[i])
            return x

        def inverseScaleX(self, x_value):
            x = x_value.copy()
            
            for i in range(x.shape[1]):
                x[:,i]= (x[:,i] * (self.x_maxs[i]-self.x_mins[i])) + self.x_means[i]
            return x



        

        def scaleY(self, y_value):
            y = y_value
            ys = (y - self.y_min)/(self.y_max_min)
            return ys

        def inverseScaleY(self, y_value):
            y = y_value
            ys = (y * self.y_max_min) + self.y_min
            return ys


        



custom_scaler = CustomScaler(X_train1,y_train1)
    
X_train = custom_scaler.scaleX(X_train1.copy())
y_train = custom_scaler.scaleY(y_train1.copy())

print(X_train[-70000: -1])

print(y_train1.shape)


# In[ ]:


(X_train1 - np.min(X_train1))/(np.max(X_train1) - np.min(X_train1))


# In[ ]:


X_train.shape


# In[ ]:





# In[ ]:



physical_devices = tf.config.list_physical_devices('GPU') 
tf.config.experimental.set_memory_growth(physical_devices[0], True)


# In[ ]:


def model1():    
    model = keras.Sequential()
    model.add(Dense(10, activation = "relu",input_dim = 8, name = "layer1"))
    model.add(Dense(20, activation = "relu", name = "layer2"))
    model.add(Dense(20, activation = "relu", name = "layer5"))
    model.add(Dense(20, activation = "relu", name = "layer6"))

    model.add(Dense(1, kernel_initializer='normal', name = "layer7"))
    model.compile(loss="mean_squared_error", optimizer='SGD')
    return model

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1)

estimator = KerasRegressor(build_fn=model1, epochs=1000, batch_size=4096, verbose=1,callbacks=[es])
history=estimator.fit(np.asarray(X_train).astype('float32'),np.asarray(y_train).astype('float32'))



#history = model.fit(x_pca, df.iloc[:, -1], batch_size=batch_size, epochs=20, validation_split=0.1)
# estimator = KerasRegressor(build_fn=model, epochs=100, batch_size=batch_size, verbose=0)
# history = model.fit(np.asarray(x).astype('float32'),np.asarray(y).astype('float32') , epochs=150, batch_size=50,  verbose=1, validation_split=0.3)
# kfold = KFold(n_splits=5)
# results = cross_val_score(estimator,np.asarray(x).astype('float32'), np.asarray(y).astype('float32'), cv=kfold)
# print("Results: %.2f (%.2f) MSE" % (results.mean(), results.std()))


# In[ ]:


import matplotlib.pyplot as plt
plt.plot(history.history['loss'])

plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()


# In[ ]:


from sklearn.metrics import mean_squared_error

y_pred = estimator.predict(X_test)
# y_pred = max_min*y_pred + y_min
print(mean_squared_error(y_test, y_pred))


# In[ ]:


np.column_stack((y_test,y_pred))


# In[ ]:


predTest = scaler.inverse_transform(np.column_stack((X_test, y_pred)))
ogTest = scaler.inverse_transform(np.column_stack((X_test, y_test)))
np.column_stack((predTest[:, -1], ogTest[:, -1]))


# In[ ]:


x


# In[ ]:




