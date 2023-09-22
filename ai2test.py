import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Load test data
df_test = pd.read_csv('test.csv')
df_test['data'] = df_test['data'].str.strip('[]').str.split().apply(lambda x: [float(i) for i in x])
X_test = np.array(df_test['data'].tolist())
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
y_test = df_test['label'].values

# Define model architecture (as per your training script)
model = Sequential()
model.add(LSTM(100, input_shape=(X_test.shape[1], X_test.shape[2])))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Load model weights
model.load_weights('lstm_model.keras')

# Predict
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
