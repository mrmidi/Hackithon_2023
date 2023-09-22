import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Input, Conv1D, Add, Activation, Flatten, Reshape
from tensorflow.keras.optimizers import Adam

# Convert the string representation of arrays in 'data' column to actual arrays
def convert_to_array(data_str):
    return np.fromstring(data_str.strip('[]').replace('\n', '').replace('  ', ' '), sep=' ')

# ResNet block
def resnet_block(x, filters, kernel_size=3):
    # Shortcut
    s = Conv1D(filters, 1, padding='same')(x)
    
    # Convolutional layer
    x = Conv1D(filters, kernel_size, padding='same')(x)
    x = Activation('relu')(x)

    # Residual connection
    x = Add()([x, s])
    return x

# Load data
df = pd.read_csv('train.csv')
df['data'] = df['data'].apply(convert_to_array)

# Prepare data for ResNet + LSTM
X = np.array(df['data'].to_list())
y = df['label'].values

# Reshape input to be [samples, time steps, features]
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

# Define the model
input_layer = Input(shape=(X.shape[1], 1))

# ResNet blocks
x = resnet_block(input_layer, 64)
x = resnet_block(x, 64)
x = resnet_block(x, 64)

# Instead of Flattening, reshape to (sequence_length, features)
sequence_length = 50  # You can adjust this value
features = int(X.shape[1] * 64 / sequence_length)  # Derived based on the chosen sequence_length
x = Reshape((sequence_length, features))(x)

# Pass to LSTM
x = LSTM(100, return_sequences=True)(x)
x = LSTM(100)(x)

# Output layer
output_layer = Dense(1, activation='sigmoid')(x)

model = Model(inputs=input_layer, outputs=output_layer)

model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

model.summary()

# Train the model
model.fit(X, y, epochs=10, batch_size=64, validation_split=0.2)

# Save the model
model.save('resnet_lstm_model.keras')
