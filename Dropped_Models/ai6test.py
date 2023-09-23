# ai_test_cnn.py

import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

from tensorflow.keras.optimizers.legacy import Adam


# Convert the string representation of arrays in 'data' column to actual arrays
def convert_to_array(data_str):
    return np.fromstring(data_str.strip('[]').replace('\n', '').replace('  ', ' '), sep=' ')

# 1. Load and preprocess the test data

# Load the test data
df_test = pd.read_csv('test.csv')
df_test['data'] = df_test['data'].apply(convert_to_array)

# Prepare the data for testing
X_test = np.array(df_test['data'].to_list())
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
y_test = df_test['label'].values

# 2. Load the model and evaluate on test data
model = load_model('extended_cnn_model.keras')

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Accuracy: {accuracy*100:.2f}%")

# Predict on the test data
y_prob = model.predict(X_test).flatten()
y_pred = (y_prob > 0.5).astype(int)  # Threshold the probabilities to get class labels

# 3. Print results for each block
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
