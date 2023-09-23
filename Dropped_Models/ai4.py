import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Convert the string representation of the list to an actual list
def convert_to_array(data_entry):
    if isinstance(data_entry, str):
        return np.array(eval(data_entry))
    else:
        return data_entry

# Load the data
df = pd.read_csv('train.csv')
df['data'] = df['data'].str.strip('[]').str.split().apply(lambda x: [float(i) for i in x])
df['data'] = df['data'].apply(convert_to_array)

# Prepare the data for LSTM
X = np.array(df['data'].to_list())
y = df['label'].values

WINDOW_SIZE = 5  # Adjust based on how many previous and next blocks you want to consider

def create_windowed_data(data, window_size=WINDOW_SIZE):
    windowed_data = []
    half_window = window_size // 2
    for i in range(len(data)):
        start_idx = max(0, i - half_window)
        end_idx = min(len(data), i + half_window + 1)
        window = data[start_idx:end_idx]
        
        # If near the beginning or end, pad the sequence to the desired window size
        while len(window) < window_size:
            if len(window) < half_window + 1:  # pad at the beginning
                window.insert(0, np.zeros_like(data[0]))
            else:  # pad at the end
                window.append(np.zeros_like(data[0]))
        
        windowed_data.append(window)
    
    return np.array(windowed_data)

X_windowed = create_windowed_data(X.tolist())

print("Shape of X_windowed:", X_windowed.shape)


# Define the LSTM model
model = Sequential()
input_shape = (X_windowed.shape[1], X_windowed.shape[2])
model.add(LSTM(100, input_shape=input_shape))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the LSTM model on the data
model.fit(X_windowed, y, epochs=100, batch_size=64, validation_split=0.3)

# Save the model
model.save('lstm_windowed_model.keras')
