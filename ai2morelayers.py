import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# 1. Preprocess the data

# Load the data
df = pd.read_csv('train.csv')
df['data'] = df['data'].str.strip('[]').str.split().apply(lambda x: [float(i) for i in x])


# Convert the string representation of the list to an actual list
def convert_to_array(data_entry):
    if isinstance(data_entry, str):
        return np.array(eval(data_entry))
    else:
        return data_entry


df['data'] = df['data'].apply(convert_to_array)

# Prepare the data for LSTM
X = np.array(df['data'].to_list())
y = df['label'].values

# Reshape input to be [samples, time steps, features]
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

# 2. Define the LSTM model
model = Sequential()
model.add(LSTM(100, input_shape=(X.shape[1], X.shape[2])))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# 3. Train the LSTM model on the data
model.fit(X, y, epochs=15, batch_size=64, validation_split=0.3)

# Save the model
model.save('enchanced_lstm_model.keras')
