# RESNET + LSTM model
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, LSTM, Dense, Conv1D, Add, Activation, BatchNormalization, Bidirectional

# Convert the string representation of arrays in 'data' column to actual arrays
def convert_to_array(data_str):
    return np.fromstring(data_str.strip('[]').replace('\n', '').replace('  ', ' '), sep=' ')

# ResNet block with causal padding
def resnet_block(x, filters, kernel_size=3):
    # Shortcut
    s = Conv1D(filters, 1, padding='causal')(x)
    
    # Convolutional layer
    x = Conv1D(filters, kernel_size, padding='causal')(x)
    x = BatchNormalization()(x)  # Batch normalization after convolution
    x = Activation('relu')(x)

    # Residual connection
    x = Add()([x, s])
    return x

# Load the data
df = pd.read_csv('train.csv')
df['data'] = df['data'].apply(convert_to_array)
X = np.array(df['data'].to_list())
y = df['label'].values

# Reshape input
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

# Define the model
input_layer = Input(shape=(X.shape[1], 1))

# ResNet blocks
filters = 64
x = input_layer
for _ in range(4):  # Four ResNet blocks
    x = resnet_block(x, filters)
    filters *= 2  # Double the number of filters

# LSTM layers
x = Bidirectional(LSTM(128, return_sequences=True))(x)
x = LSTM(128, return_sequences=True)(x)
x = LSTM(128)(x)

# Output layer
output_layer = Dense(1, activation='sigmoid')(x)

model = Model(inputs=input_layer, outputs=output_layer)

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.summary()

# Train the model
model.fit(X, y, epochs=15, batch_size=64, validation_split=0.2)

# Save the model
model.save('resnet_lstm_model.keras')
