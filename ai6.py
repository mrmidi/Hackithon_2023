# CNN

import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Dense, MaxPooling1D, Flatten
from tensorflow.keras.optimizers import Adam

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

# Prepare the data for RNN
X = np.array(df['data'].to_list())
y = df['label'].values

# Reshape input to be [samples, time steps, features]
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

# Define the extended CNN model
model = Sequential()

# 1D Convolutional layers
model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(X.shape[1], X.shape[2])))
model.add(MaxPooling1D(pool_size=2))

model.add(Conv1D(filters=128, kernel_size=3, activation='relu'))
model.add(MaxPooling1D(pool_size=2))

model.add(Conv1D(filters=256, kernel_size=3, activation='relu'))
model.add(MaxPooling1D(pool_size=2))

model.add(Conv1D(filters=512, kernel_size=3, activation='relu'))
model.add(MaxPooling1D(pool_size=2))

model.add(Flatten())

# Fully connected layers
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X, y, epochs=15, batch_size=64, validation_split=0.3)

# Save the model
model.save('extended_cnn_model.keras')
