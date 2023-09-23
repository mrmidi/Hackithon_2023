# ai2test.py
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import json

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# Load test data
df_test = pd.read_csv('blocks.csv')
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


results = []

# Create results for each block
for i in range(len(y_test)):
    block_number = df_test['block_number'].iloc[i]
    true_label = y_test[i]
    pred_label = y_pred[i]
    
    result_dict = {
        "block_number": int(block_number),
        "true_label": int(true_label),
        "pred_label": int(pred_label)
    }
    
    results.append(result_dict)

# Serialize the results to JSON format
json_output = json.dumps(results, indent=4)

# Optionally, you can write the JSON output to a file
with open('results.json', 'w') as outfile:
    outfile.write(json_output)

# Print the JSON output
print(json_output)

# Compute and print overall metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"\nOverall Metrics:")
print(f"Accuracy: {accuracy*100:.2f}%")
print(f"Precision: {precision*100:.2f}%")
print(f"Recall: {recall*100:.2f}%")
print(f"F1 Score: {f1*100:.2f}%")
