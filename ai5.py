import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense
from tensorflow.keras.optimizers import Adam

# Load the data
df = pd.read_csv('blocks_with_labels.csv')
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

# Define the RNN model
model = Sequential()
model.add(SimpleRNN(100, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
model.add(SimpleRNN(100))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X, y, epochs=15, batch_size=64, validation_split=0.3)

# Save the model
model.save('rnn_model.keras')
