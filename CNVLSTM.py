import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential, Model
from keras.layers import LSTM, Dense, Dropout, Input, Conv1D, MaxPooling1D, Flatten, Activation, concatenate, Permute, multiply, Lambda
from keras import backend as K
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam

df = pd.read_csv('AAPL1.csv')

# Preprocess the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df['close'].values.reshape(-1, 1))

# Split the data into training and testing sets
train_size = int(len(scaled_data) * 0.8)
test_size = len(scaled_data) - train_size
train_data = scaled_data[0:train_size, :]
test_data = scaled_data[train_size:len(scaled_data), :]

# Create sequences of data for the model
def create_sequences(data, seq_length):
    x = []
    y = []
    for i in range(len(data)-seq_length-1):
        x.append(data[i:(i+seq_length), 0])
        y.append(data[(i+seq_length), 0])
    return np.array(x), np.array(y)

seq_length = 30
x_train, y_train = create_sequences(train_data, seq_length)
x_test, y_test = create_sequences(test_data, seq_length)

# Reshape the data for the model
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))


# Define the model architecture
input_shape = (seq_length, 1)

inputs = Input(shape=input_shape)

# Convolutional layer
conv1 = Conv1D(filters=32, kernel_size=5, padding='same', activation='relu')(inputs)

# Max pooling layer
pool1 = MaxPooling1D(pool_size=2)(conv1)

# Flatten layer
flatten1 = Flatten()(pool1)

# LSTM layer
lstm1 = LSTM(64, return_sequences=True)(inputs)

# Attention mechanism
attention = Dense(1, activation='tanh')(lstm1)
attention = Flatten()(attention)
attention = Activation('softmax')(attention)
attention = multiply([lstm1, attention])

# LSTM layer
lstm2 = LSTM(64, return_sequences=False)(attention)

# Concatenate the output of the convolutional layer and the LSTM layer
merge = concatenate([flatten1, lstm2])

# Dense layer
dense1 = Dense(64, activation='relu')(merge)

# Dropout layer
dropout1 = Dropout(0.2)(dense1)

# Output layer
outputs = Dense(1)(dropout1)

model = Model(inputs=inputs, outputs=outputs)

# Compile the model
optimizer = Adam(lr=0.001)
model.compile(loss='mean_squared_error', optimizer=optimizer)

# Train the model
early_stopping = EarlyStopping(monitor='val_loss', patience=10)
history = model.fit(x_train, y_train, batch_size=64, epochs=100, validation_split=0.2, callbacks=[early_stopping])

# Make predictions on the test data
y_pred = model.predict(x_test)

# Convert the predictions back to the original scale
y_pred = scaler.inverse_transform(y_pred)
y_test = scaler.inverse_transform(y_test.reshape(-1, 1))

# Calculate the mean absolute error and mean squared error
mae = np.mean(np.abs(y_test - y_pred))
mse = np.mean(np.square(y_test - y_pred))
print("mse-> ", mse)
print("mae-> ", mae)
