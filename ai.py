#ai.py

import numpy as np
import pandas as pd
import pywt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Bidirectional, Conv1D

# Convert the string representation of arrays in 'data' column to actual arrays
def convert_to_array(data_str):
    return np.fromstring(data_str.strip('[]').replace('\n', '').replace('  ', ' '), sep=' ')

df = pd.read_csv('train.csv')
df['data'] = df['data'].apply(convert_to_array)

# Prepare data for network
X = np.array(df['data'].to_list())
y = df['label'].values

# Reshape input to be [samples, time steps, features]
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

# Building the model
model = Sequential()

# Simplified WaveNet component
for _ in range(2):  # Two blocks
    dilation_rate = 1
    for _ in range(4):  # Four convolutional layers per block
        model.add(Conv1D(32, 3, padding='causal', dilation_rate=dilation_rate, activation='relu'))
        dilation_rate *= 2

# Stacked LSTM layers component
model.add(Bidirectional(LSTM(128, return_sequences=True)))  # First LSTM layer is bidirectional
model.add(LSTM(128, return_sequences=True))
model.add(LSTM(128))
model.add(Dense(1))

model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(X, y, epochs=10, batch_size=64, validation_split=0.2)

# Save the model
model.save('wavenet_lstm_model.keras')
model.save_weights('wavenet_lstm_weights.keras')