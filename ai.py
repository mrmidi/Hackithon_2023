import numpy as np
import pandas as pd
import pywt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Convert the string representation of arrays in 'data' column to actual arrays
def convert_to_array(data_str):
    return np.fromstring(data_str.strip('[]').replace('\n', '').replace('  ', ' '), sep=' ')

df = pd.read_csv('train.csv')
df['data'] = df['data'].apply(convert_to_array)

# Apply wavelet decomposition
def wavelet_transform(row):
    coeffs = pywt.wavedec(row, 'db1', level=4)
    cA, _, _, _, _ = coeffs
    return cA

df['wavelet_coeffs'] = df['data'].apply(wavelet_transform)

# Prepare data for LSTM training
X = np.array(df['wavelet_coeffs'].to_list())
y = df['label'].values

# Reshape input to be [samples, time steps, features]
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

# Build and train the LSTM model
model = Sequential()
model.add(LSTM(100, input_shape=(X.shape[1], 1)))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X, y, epochs=10, batch_size=64, validation_split=0.2)

# Save the model (optional)
model.save('wavelet_lstm_model.keras')

print("Model training completed.")

# ... [Your previous code here]

# 1. Load the test data
test_df = pd.read_csv('test.csv')
test_df['data'] = test_df['data'].apply(convert_to_array)

# 2. Pre-process the test data
test_df['wavelet_coeffs'] = test_df['data'].apply(wavelet_transform)

X_test = np.array(test_df['wavelet_coeffs'].to_list())
y_test = test_df['label'].values

# Reshape input to be [samples, time steps, features]
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# 3. Evaluate the model on the test data
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Accuracy: {accuracy*100:.2f}%")
