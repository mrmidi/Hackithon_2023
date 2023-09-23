import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import SimpleRNN, Dense
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def convert_to_array(data_str):
    return np.fromstring(data_str.strip('[]').replace('\n', '').replace('  ', ' '), sep=' ')

# Load test data
df_test = pd.read_csv('test.csv')
df_test['data'] = df_test['data'].apply(convert_to_array)

# Prepare data for prediction
X_test = np.array(df_test['data'].to_list())
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
y_test = df_test['label'].values

# Reconstruct the model architecture
model = Sequential()
model.add(SimpleRNN(100, input_shape=(X_test.shape[1], X_test.shape[2]), return_sequences=True))
model.add(SimpleRNN(100))
model.add(Dense(1, activation='sigmoid'))

# Load the saved weights into the reconstructed model
model.load_weights('rnn_model_weights.h5')

# Compile the model
model.compile(loss='binary_crossentropy', optimizer=Adam(), metrics=['accuracy'])


# Predict on the test data
y_prob = model.predict(X_test).flatten()
y_pred = (y_prob > 0.5).astype(int)  # Threshold the probabilities to get class labels

# Print results for each block
results = []
for i in range(len(y_test)):
    block_number = df_test['block_number'].iloc[i]
    true_label = y_test[i]
    pred_label = y_pred[i]

    result_dict = {
        "block_number": block_number,
        "true_label": true_label,
        "pred_label": pred_label
    }
    results.append(result_dict)

# Print the results
for res in results:
    print(f"Block Number: {res['block_number']}, True Label: {res['true_label']}, Predicted Label: {res['pred_label']}")

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
