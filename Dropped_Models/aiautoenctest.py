import numpy as np
import pandas as pd
from numpy.fft import fft
from tensorflow.keras.models import load_model

# Convert the string representation of arrays in 'data' column to actual arrays
def convert_to_array(data_str):
    return np.fromstring(data_str.strip('[]').replace('\n', '').replace('  ', ' '), sep=' ')

# Load the test data
df_test = pd.read_csv('test.csv')
df_test['data'] = df_test['data'].apply(convert_to_array)
X_test = np.array(df_test['data'].to_list())

# Convert each block to frequency domain (keep only magnitude)
X_test_freq = np.abs(fft(X_test, axis=1))

# Load the saved autoencoder model
autoencoder = load_model('autoencoder_model.keras')

# Predict with the autoencoder
X_test_freq_reconstructed = autoencoder.predict(X_test_freq)

# Compute the reconstruction error for each sample
reconstruction_error = np.mean(np.square(X_test_freq - X_test_freq_reconstructed), axis=1)

print(reconstruction_error)
