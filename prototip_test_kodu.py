# -*- coding: utf-8 -*-
"""
Created on Sat Jan 13 21:30:42 2024

@author: sakin
"""
import pandas as pd
from datetime import timedelta
import seaborn as sns
from scipy.stats import randint
from sklearn.model_selection import RandomizedSearchCV
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import time
from datetime import datetime, timedelta
import requests
import json
import pandas as pd
import warnings
import certifi
import urllib3
import os
import serial
commPort = 'COM3'
ser = serial.Serial(commPort, baudrate=9600, timeout=1)
# Get the desktop directory
desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")

# Change the current working directory to the desktop
os.chdir(desktop_path)
df = pd.read_csv('data.csv')
df2 = pd.read_csv('data (2).csv')
df3 = pd.read_csv('data (3).csv')
df4 = pd.read_csv('data (4).csv')
df5 = pd.read_csv('data (5).csv')
df6 = pd.read_csv('data (6).csv')
df7 = pd.read_csv('data (7).csv')
df8 = pd.read_csv('data (8).csv')
df9 = pd.read_csv('data (9).csv')
df1 = pd.read_csv('data (1).csv')

df = df.rename(columns={"2017-01-01T00:00:00Z": 'time'})
df2 = df2.rename(columns={"2015-01-01T00:00:00Z": 'time'})
df3 = df3.rename(columns={"2014-01-01T00:00:00Z": 'time'})
df4 = df4.rename(columns={"2013-01-01T00:00:00Z": 'time'})
df5 = df5.rename(columns={"2018-01-01T00:00:00Z": 'time'})
df6 = df6.rename(columns={"2020-01-01T00:00:00Z": 'time'})
df7 = df7.rename(columns={"2019-01-01T00:00:00Z": 'time'})
df8 = df8.rename(columns={"2012-01-01T00:00:00Z": 'time'})
df9 = df9.rename(columns={"2011-01-10T00:00:00Z": 'time'})
df1 = df1.rename(columns={"2016-01-01T00:00:00Z": 'time'})

df = df.rename(columns={"65.0": 'Bx'})
df = df.rename(columns={"32.0": 'By'})
df = df.rename(columns={"0.196": 'Bz'})
df = df.rename(columns={"72.9": 'Bt'})

df1 = df1.rename(columns={"37.0": 'Bx'})
df1 = df1.rename(columns={"85.6": 'By'})
df1 = df1.rename(columns={"-12.1": 'Bz'})
df1 = df1.rename(columns={"94.2": 'Bt'})

df2 = df2.rename(columns={"84.0": 'Bx'})
df2 = df2.rename(columns={"37.9": 'By'})
df2 = df2.rename(columns={"0.0653": 'Bz'})
df2 = df2.rename(columns={"92.4": 'Bt'})

df3 = df3.rename(columns={"85.0": 'Bx'})
df3 = df3.rename(columns={"31.0": 'By'})
df3 = df3.rename(columns={"-0.123": 'Bz'})
df3 = df3.rename(columns={"90.2": 'Bt'})

df4 = df4.rename(columns={"85.0": 'Bx'})
df4 = df4.rename(columns={"33.9": 'By'})
df4 = df4.rename(columns={"-0.883": 'Bz'})
df4 = df4.rename(columns={"91.6": 'Bt'})

df5 = df5.rename(columns={"119.0": 'Bx'})
df5 = df5.rename(columns={"13.0": 'By'})
df5 = df5.rename(columns={"20.7": 'Bz'})
df5 = df5.rename(columns={"122.0": 'Bt'})

df6 = df6.rename(columns={"100.0": 'Bx'})
df6 = df6.rename(columns={"25.3": 'By'})
df6 = df6.rename(columns={"9.35": 'Bz'})
df6 = df6.rename(columns={"103.0": 'Bt'})

df7 = df7.rename(columns={"104.0": 'Bx'})
df7 = df7.rename(columns={"20.3": 'By'})
df7 = df7.rename(columns={"16.9": 'Bz'})
df7 = df7.rename(columns={"107.0": 'Bt'})

df8 = df8.rename(columns={"68.0": 'Bx'})
df8 = df8.rename(columns={"29.6": 'By'})
df8 = df8.rename(columns={"-2.0": 'Bz'})
df8 = df8.rename(columns={"74.0": 'Bt'})

df9 = df9.rename(columns={"60.0": 'Bx'})
df9 = df9.rename(columns={"24.6": 'By'})
df9 = df9.rename(columns={"1.75": 'Bz'})
df9 = df9.rename(columns={"64.9": 'Bt'})

a = pd.concat([df1,df,df2,df3,df4,df5,df6,df7,df8,df9],join="inner", ignore_index=False)
a = a.sort_values(by='time')
a=a[a['Bt']>-9998]
a=a[a['Bx']>-9998]
a=a[a['By']>-9998]
a=a[a['Bz']>-9998]

data = pd.read_csv('speed.csv')
data1 = pd.read_csv('speed1.csv')
data2 = pd.read_csv('speed2.csv')
data3 = pd.read_csv('speed3.csv')
data4 = pd.read_csv('speed4.csv')

data = data.rename(columns={"2019-01-01T00:00:00Z": 'time'})
data1 = data1.rename(columns={"2011-01-01T00:00:00Z": 'time'})
data2 = data2.rename(columns={"2013-01-01T00:00:00Z": 'time'})
data3 = data3.rename(columns={"2015-01-01T00:00:00Z": 'time'})
data4 = data4.rename(columns={"2017-01-01T00:00:00Z": 'time'})

data = data.rename(columns={"3.2": 'density'})
data = data.rename(columns={"466.8": 'speed'})
data = data.rename(columns={"46600.0": 'temp'})

data1 = data1.rename(columns={"3.5": 'density'})
data1 = data1.rename(columns={"342.7": 'speed'})
data1 = data1.rename(columns={"29900.0": 'temp'})

data2 = data2.rename(columns={"1.7": 'density'})
data2 = data2.rename(columns={"353.6": 'speed'})
data2 = data2.rename(columns={"26000.0": 'temp'})

data3 = data3.rename(columns={"2.6": 'density'})
data3 = data3.rename(columns={"557.2": 'speed'})
data3 = data3.rename(columns={"310000.0": 'temp'})

data4 = data4.rename(columns={"-9999.9": 'density'})
data4 = data4.rename(columns={"-9999.9.1": 'speed'})
data4 = data4.rename(columns={"-100000.0": 'temp'})

b = pd.concat([data,data1,data2,data3,data4],join='inner',ignore_index=False)
b = b.sort_values(by='time')
b=b[b['density']>-9998]
b=b[b['speed']>-9998]
b=b[b['temp']>-9998]

dst = pd.read_csv('dst.csv')
dst1 = pd.read_csv('dst1.csv')

dst = dst.rename(columns={"-11.0": 'dst'})
dst = dst.rename(columns={"2011-01-01T00:00:00Z": 'time'})

dst1 = dst1.rename(columns={"-13.0": 'dst'})
dst1 = dst1.rename(columns={"2017-01-01T00:00:00Z": 'time'})

c = pd.concat([dst,dst1],join='inner',ignore_index=False)
c = c.sort_values(by='time')

d = pd.read_csv('kp1.csv')
d = d.rename(columns={"0.333": 'kp'})
d = d.rename(columns={"2011-01-01T00:00:00Z": 'time'})
d = d.drop(columns=['2.0'], inplace=False)

#print(d.head(5))

data = pd.merge(a,b,on='time')
data = pd.merge(data,c,on='time')
data = pd.merge(data,d,on='time')


data['time']= pd.to_datetime(data['time'],format='%Y-%m-%dT%H:%M:%SZ',errors='coerce')
ilk=data.head(1)
#print(data.info())
data['kp']=data['kp'].shift(-1)
#print(data['kp'].head(5))
data=data.dropna()
datat=data['time']
X = data.drop(['kp','time','Bz','Bt'],axis=1)
y = data['kp']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
train_data = X_train.join(y_train)
#plt.figure(figsize=(15,8))
#sns.heatmap(train_data.corr(), annot=True)
forest= RandomForestRegressor(n_estimators=120,bootstrap=False,max_features='log2',max_depth=65)
forest.fit(X_train, y_train)

for i in range(0,100):
    def make_predictions(input_data):
        predictions = forest.predict(input_data)
        return predictions
    
    Bx = float(input("Enter Bx value: "))
    By = float(input("Enter By value: "))
    density = float(input("Enter density value: "))
    speed = float(input("Enter speed value: "))
    temp = float(input("Enter temp value: "))
    dst = float(input("Enter dst value: "))
    new_input_data = pd.DataFrame({
        'Bx': [Bx],
        'By': [By],
        'density': [density],
        'speed': [speed],
        'temp': [temp],
        'dst': [dst]
    })
    
    # Make predictions using the trained model
    predicted_kp = make_predictions(new_input_data)
    #print(predicted_kp)
    predicted_kp_r = ((predicted_kp*3).round())/3
    print(predicted_kp_r)
    if  predicted_kp_r > 6:
        ser.write(b'x')
    else:
        ser.write(b'o')


