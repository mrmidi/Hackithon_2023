import numpy as np
import pandas as pd
import pywt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Convert the string representation of arrays in 'data' column to actual arrays
def convert_to_array(data_str):
    return np.fromstring(data_str.strip('[]').replace('\n', '').replace('  ', ' '), sep=' ')

# Apply wavelet decomposition
def wavelet_transform(row):
    coeffs = pywt.wavedec(row, 'db1', level=4)
    cA, _, _, _, _ = coeffs
    return cA

# Load the test data
df_test = pd.read_csv('test.csv')
df_test['data'] = df_test['data'].apply(convert_to_array)

# Apply wavelet transform to the test data
df_test['wavelet_coeffs'] = df_test['data'].apply(wavelet_transform)

# Prepare data for prediction
X_test = np.array(df_test['wavelet_coeffs'].to_list())
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
y_test = df_test['label'].values

# Define model architecture
model = Sequential()
model.add(LSTM(100, input_shape=(X_test.shape[1], X_test.shape[2])))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Load the saved weights
model.load_model('wavelet_lstm_model.keras')

# Predict on the test data
y_prob = model.predict(X_test).flatten()
y_pred = (y_prob > 0.5).astype(int)  # Threshold the probabilities to get class labels

# Print results for each block
for i in range(len(y_test)):
    block_number = df_test['block_number'].iloc[i]
    true_label = y_test[i]
    pred_label = y_pred[i]

    if true_label == 1 and pred_label == 1:
        result = "True Positive"
    elif true_label == 0 and pred_label == 1:
        result = "False Positive"
    elif true_label == 1 and pred_label == 0:
        result = "False Negative"
    else:
        result = "True Negative"

    print(f"Block Number: {block_number}, Result: {result}")
