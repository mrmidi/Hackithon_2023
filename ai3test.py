import pandas as pd
import numpy as np
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Conv1D, Add, Activation, LSTM, Dense, Reshape

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

# Load test data
df_test = pd.read_csv('test.csv')
df_test['data'] = df_test['data'].apply(convert_to_array)
X_test = np.array(df_test['data'].to_list())
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
y_test = df_test['label'].values

# Define model architecture

# Input
input_layer = Input(shape=(X_test.shape[1], 1))

# ResNet blocks
x = resnet_block(input_layer, 64)
x = resnet_block(x, 64)
x = resnet_block(x, 64)

# Reshape for LSTM
sequence_length = 50
features = int(X_test.shape[1] * 64 / sequence_length)
x = Reshape((sequence_length, features))(x)

# LSTM layers
x = LSTM(100, return_sequences=True)(x)
x = LSTM(100)(x)

# Output layer
output_layer = Dense(1, activation='sigmoid')(x)

model = Model(inputs=input_layer, outputs=output_layer)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Load the trained model weights
model.load_weights('resnet_lstm_model.keras')

# Predict
y_prob = model.predict(X_test).flatten()
y_pred = (y_prob > 0.5).astype(int)  # Threshold the probabilities to get class labels

# Print results for each block
for i in range(len(y_test)):
    block_number = df_test['block_number'].iloc[i]
    true_label = y_test[i]
    pred_label = y_pred[i]

    # if true_label == 1 and pred_label == 1:
    #     result = "True Positive"
    # elif true_label == 0 and pred_label == 1:
    #     result = "False Positive"
    # elif true_label == 1 and pred_label == 0:
    #     result = "False Negative"
    # else:
    #     result = "True Negative"

    if true_label == 1 and pred_label == 1:
        result = "True Positive"
    elif true_label == 0 and pred_label == 1:
        result = "False Positive"
    elif true_label == 1 and pred_label == 0:
        result = "False Negative"
    else:
        result = "True Negative"

    print(f"Block Number: {block_number}, Result: {result}")


# calculate total accuracy
total = len(y_test)
correct = 0
for i in range(len(y_test)):
    if y_test[i] == y_pred[i]:
        correct += 1
print("Total Accuracy: ", correct/total)