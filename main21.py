import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
import tensorflow as tf
from keras.layers import Bidirectional
from keras import backend as K
from keras.layers import *
from keras.models import *
from sklearn.metrics import mean_squared_error
import math
from numpy import array
from statistics import stdev, mean

#Loading the data in our code and scaling it for LSTM
df=pd.read_csv('AAPL1.csv')
df1=df.reset_index()['close']
plt.plot(df1)
print(df1)
scaler=MinMaxScaler(feature_range=(0,1))#We use Minmaxscaler to produce values between 0 and 1
df1=scaler.fit_transform(np.array(df1).reshape(-1,1))

#Preprocessing the data by splitting into training and testing sets
training_size=int(len(df1)*0.65)
test_size=len(df1)-training_size
train_data,test_data=df1[0:training_size,:],df1[training_size:len(df1),:1]

class attention(Layer):
    def __init__(self, return_sequences=True):
        self.return_sequences = return_sequences

        super(attention,self).__init__()

    def build(self, input_shape):
        self.W=self.add_weight(name="att_weight", shape=(input_shape[-1],1),
                               initializer="normal")
        self.b=self.add_weight(name="att_bias", shape=(input_shape[1],1),
                               initializer="zeros")


    def call(self, x):
        e = K.tanh(K.dot(x,self.W)+self.b)
        a = K.softmax(e, axis=1)
        output = x*a
        if self.return_sequences:

            return output
        return K.sum(output, axis=1)

def rsi(a):
    gain=0
    loss=0
    gc=0
    lc=0
    for j in range(1,len(a)):
        if a[j]>a[j-1]:
            gain=gain+a[j]-a[j-1]
            gc+=1
        else:
            loss=loss+a[j-1]-a[j]
            lc+=1
    avg_gain=gain/gc
    avg_loss=loss/lc
    RSI=100-100/(1+(avg_gain/avg_loss))
    print(RSI)
    return RSI

'''def obv(a, vol_data):
    OBV=vol_data[0]
    for j in range(1,len(a)-1):
        if a[j]>a[j-1]:
            OBV+=vol_data[j]
        else:
            OBV-=vol_data[j]
    if OBV>=0:
        return 1
    else:
        return 0'''

def create_dataset(dataset, time_step=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-time_step-1):
        a = dataset[i:(i+time_step), 0]   ###i=0, 0,1,2,3-----99   100 
        a = a.tolist()
        a.append(rsi(a))
        a.append(mean(a))
        a.append(stdev(a))
        a = np.array(a).reshape(-1, 1)
        dataX.append(a)
        dataY.append(dataset[i + time_step, 0])
    return np.array(dataX), np.array(dataY)

time_step = 100
X_train, y_train = create_dataset(train_data, time_step)
X_test, ytest = create_dataset(test_data, time_step)

#Create and run a stacked LSTM model
model=Sequential()
model.add(Bidirectional(LSTM(50,return_sequences=True,input_shape=(103,1))))
model.add(attention(return_sequences=True))
model.add(Bidirectional(LSTM(50,return_sequences=True)))
model.add(attention(return_sequences=True))
model.add(Bidirectional(LSTM(50)))
model.add(Dense(1))
model.compile(loss='mean_squared_error',optimizer='adam')
model.fit(X_train,y_train,validation_data=(X_test,ytest),epochs=10,batch_size=64,verbose=1)

#Predicting the data from the model
train_predict=model.predict(X_train)
test_predict=model.predict(X_test)
#rescaling the data back
train_predict=scaler.inverse_transform(train_predict)
test_predict=scaler.inverse_transform(test_predict)
print(math.sqrt(mean_squared_error(y_train,train_predict)))
print("error=",math.sqrt(mean_squared_error(ytest,test_predict)))
look_back=100
trainPredictPlot = np.empty_like(df1)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(train_predict)+look_back, :] = train_predict
# shift test predictions for plotting
testPredictPlot = np.empty_like(df1)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(train_predict)+(look_back*2)+1:len(df1)-1, :] = test_predict
# plot baseline and predictions
plt.plot(scaler.inverse_transform(df1))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.show()



#Predicting the future 30 days
x_input=test_data[341:].reshape(1,-1)
temp_input=list(x_input)
temp_input=temp_input[0].tolist()
lst_output=[]
n_steps=100
i=0
while(i<30):
    
    if(len(temp_input)>100):
        #print(temp_input)
        x_input=np.array(temp_input[1:])
        print("{} day input {}".format(i,x_input))
        x_input=x_input.reshape(1,-1)
        x_input = x_input.reshape((1, n_steps, 1))
        #print(x_input)
        yhat = model.predict(x_input, verbose=0)
        print("{} day output {}".format(i,yhat))
        temp_input.extend(yhat[0].tolist())
        temp_input=temp_input[1:]
        #print(temp_input)
        lst_output.extend(yhat.tolist())
        i=i+1
    else:
        x_input = x_input.reshape((1, n_steps,1))
        yhat = model.predict(x_input, verbose=0)
        print(yhat[0])
        temp_input.extend(yhat[0].tolist())
        print(len(temp_input))
        lst_output.extend(yhat.tolist())
        i=i+1
    

#print(lst_output)
day_new=np.arange(1,101)
day_pred=np.arange(101,131)
plt.plot(day_new,scaler.inverse_transform(df1[1158:]))
plt.plot(day_pred,scaler.inverse_transform(lst_output))
df3=df1.tolist()
df3.extend(lst_output)
plt.plot(df3[1200:])
plt.show()
df3=scaler.inverse_transform(df3).tolist()
plt.plot(df3)
