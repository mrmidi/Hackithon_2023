import numpy as np
import pandas as pd
from numpy.fft import fft
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense

# Convert the string representation of arrays in 'data' column to actual arrays
def convert_to_array(data_str):
    return np.fromstring(data_str.strip('[]').replace('\n', '').replace('  ', ' '), sep=' ')

# Load the data
df = pd.read_csv('train.csv')
df['data'] = df['data'].apply(convert_to_array)
X = np.array(df['data'].to_list())

# Convert each block to frequency domain (keep only magnitude)
X_freq = np.abs(fft(X, axis=1))

# Define autoencoder model
input_layer = Input(shape=(X_freq.shape[1],))
encoded = Dense(512, activation='relu')(input_layer)
encoded = Dense(128, activation='relu')(encoded)
encoded = Dense(64, activation='relu')(encoded)

decoded = Dense(128, activation='relu')(encoded)
decoded = Dense(512, activation='relu')(decoded)
decoded = Dense(X_freq.shape[1], activation='sigmoid')(decoded)

autoencoder = Model(input_layer, decoded)
autoencoder.compile(optimizer='adam', loss='mean_squared_error')

# Train the autoencoder
autoencoder.fit(X_freq, X_freq, epochs=50, batch_size=256, validation_split=0.2)

# Save the model
autoencoder.save
