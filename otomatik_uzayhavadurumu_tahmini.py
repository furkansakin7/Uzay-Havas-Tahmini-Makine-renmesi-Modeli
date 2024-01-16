# -*- coding: utf-8 -*-
"""
Created on Tue Dec 26 15:39:49 2023

@author: sakin
"""
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 08:50:43 2023

@author: sakin
"""
import pandas as pd
from datetime import timedelta
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import time
import requests
import serial
from datetime import datetime, timedelta
commPort = 'COM3'
ser = serial.Serial(commPort, baudrate=9600, timeout=1)
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
for i in range (0,5):
    max_retries = 10
    retry_delay = 5  # saniye cinsinden
    retry_count = 0
    bugunun_tarihi = datetime.utcnow() - timedelta(hours=2)
    max_tarih = datetime(2099, 1, 1)
    max_tarih_str = max_tarih.strftime('%Y-%m-%dT%H:%M:%S.0Z')
    print(bugunun_tarihi)
    while retry_count < max_retries:
        try:
            url = f"https://iswa.gsfc.nasa.gov/IswaSystemWebApp/hapi/data?id=goesp_mag_p1m&time.min={bugunun_tarihi}&time.max=2030-04-26T00:00:00.0Z&format=json"
            response = requests.get(url,verify=False,timeout=15)
            response.raise_for_status()
        #print(response.status_code)
            if response.status_code==200:
                dataa= response.json()
                columns = ['time', 'Bx', 'By', 'Bz', 'Bt']
                df = pd.DataFrame(dataa['data'], columns=columns)
                df['time'] = pd.to_datetime(df['time'],format="%Y-%m-%dT%H:%M:%S%z")
                df.set_index('time', inplace=True)
                mag = df[df.index.minute == 0]
                #print(mag)
            url2 = f"https://iswa.gsfc.nasa.gov/IswaSystemWebApp/hapi/data?id=dscovr_plasma_1m&time.min={bugunun_tarihi}&time.max=2030-04-26T00:00:00.0Z&format=json"
            response2 = requests.get(url2,verify=False,timeout=15)
            response.raise_for_status()
            if response2.status_code==200:
                data2 =response2.json()
                columns = ['time','density','speed','temp']
                df2 = pd.DataFrame(data2['data'],columns=columns)
                df2['time'] = pd.to_datetime(df2['time'],format="%Y-%m-%dT%H:%M:%S%z")
                df2.set_index('time',inplace=True)
                speed = df2[df2.index.minute == 0]
                #print(speed)
            url3 = f"https://iswa.gsfc.nasa.gov/IswaSystemWebApp/hapi/data?id=dst_quicklook&time.min={bugunun_tarihi}&time.max=2030-04-26T00:00:00.0Z&format=json"
            response3 = requests.get(url3,verify=False,timeout=15)
            response.raise_for_status()
            if response3.status_code==200:
                data3 =response3.json()
                columns = ['time','dst']
                df3 = pd.DataFrame(data3['data'],columns=columns)
                df3['time'] = pd.to_datetime(df3['time'],format="%Y-%m-%dT%H:%M:%S%z")
                df3.set_index('time',inplace=True)
                #print(df3)
            tahmin = pd.merge(mag,speed,on='time')
            tahmin = pd.merge(tahmin,df3,on='time')
            tahmin = tahmin.head(1)
            #tahmin =tahmin.dropna()
            #print(tahmin)
            tahmin_degeri = forest.predict(tahmin.drop(['Bz','Bt'],axis=1))
            saat = bugunun_tarihi + timedelta(hours=4)
            saat = saat.replace(minute=0,second=0,microsecond=0)
            tahmin = ((tahmin_degeri * 3).round() / 3)
            print(tahmin,saat)
            if tahmin >4.9:
                ser.write(b'x')
            else:
                ser.write(b'o')
            retry_count=11
        except requests.exceptions.RequestException as e:
            print(f"Hata oluştu: {e}")
            retry_count += 1
            print(f"Tekrar deneme ({retry_count}/{max_retries})...")
            time.sleep(retry_delay)
    time.sleep(1800)















