# import numpy as np
# import pandas as pd
# import pywt
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import LSTM, Dense

# # Convert the string representation of arrays in 'data' column to actual arrays
# def convert_to_array(data_str):
#     return np.fromstring(data_str.strip('[]').replace('\n', '').replace('  ', ' '), sep=' ')

# # Apply wavelet decomposition
# def wavelet_transform(row):
#     coeffs = pywt.wavedec(row, 'db1', level=4)
#     cA, _, _, _, _ = coeffs
#     return cA

# # Load test data
# df_test = pd.read_csv('test.csv')
# df_test['data'] = df_test['data'].apply(convert_to_array)

# # Apply wavelet transform to the test data
# df_test['wavelet_coeffs'] = df_test['data'].apply(wavelet_transform)

# # Prepare data for prediction
# X_test = np.array(df_test['wavelet_coeffs'].to_list())
# X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
# y_test = df_test['label'].values

# # Define the model architecture (as in the training script)
# model = Sequential()
# model.add(LSTM(128, input_shape=(X_test.shape[1], 1)))
# model.add(Dense(1, activation='sigmoid'))
# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# # Load weights
# model.load_weights('wavelet_lstm_model_weights.keras')

# # Predict on the test data
# y_prob = model.predict(X_test).flatten()
# y_pred = (y_prob > 0.5).astype(int)  # Threshold the probabilities to get class labels

# # Print results for each block
# for i in range(len(y_test)):
#     block_number = df_test['block_number'].iloc[i]
#     true_label = y_test[i]
#     pred_label = y_pred[i]

#     if true_label == 1 and pred_label == 1:
#         result = "True Positive"
#     elif true_label == 0 and pred_label == 1:
#         result = "False Positive"
#     elif true_label == 1 and pred_label == 0:
#         result = "False Negative"
#     else:
#         result = "True Negative"

#     print(f"Block Number: {block_number}, Result: {result}")

# # Optionally, you can compute and print overall metrics like accuracy, precision, recall, etc.

# # Compute overall metrics
# accuracy = np.mean(y_test == y_pred)
# precision = np.sum((y_test == 1) & (y_pred == 1)) / np.sum(y_pred)
# recall = np.sum((y_test == 1) & (y_pred == 1)) / np.sum(y_test)
# f1 = 2 * precision * recall / (precision + recall)

# # Print overall metrics
# print(f"Accuracy: {accuracy*100:.2f}%")
# print(f"Precision: {precision:.4f}")
# print(f"Recall: {recall:.4f}")
# print(f"F1-score: {f1:.4f}")




import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tensorflow.keras.models import load_model

# Load the model
model_path = 'wavenet_lstm_model.keras'
model = load_model(model_path)

# Load the weights
model.save_weights("wavenet_lstm_model_weights.keras")

# Re-compile the model
#model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.load_weights('wavenet_lstm_model_weights.keras')


# Convert the string representation of arrays in 'data' column to actual arrays
def convert_to_array(data_str):
    return np.fromstring(data_str.strip('[]').replace('\n', '').replace('  ', ' '), sep=' ')

# Load test data
df_test = pd.read_csv('train.csv')
df_test['data'] = df_test['data'].apply(convert_to_array)

# Prepare data for prediction
X_test = np.array(df_test['data'].to_list())
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
y_test = df_test['label'].values

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
