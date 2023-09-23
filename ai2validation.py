import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load data
df = pd.read_csv('valid.csv')
df['data'] = df['data'].str.strip('[]').str.split().apply(lambda x: [float(i) for i in x])
X = np.array(df['data'].tolist())
X = np.reshape(X, (X.shape[0], X.shape[1], 1))
y = df['label'].values

# Split data into training and validation sets (80% train, 20% validation)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Define model architecture
model = Sequential()
model.add(LSTM(100, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dense(1, activation='sigmoid'))

# Load pre-trained weights
model.load_weights('lstm_model.keras')

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Early stopping based on validation loss
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Train the model using training data and validate using validation data
model.fit(X_train, y_train, epochs=50, batch_size=64, validation_data=(X_val, y_val), callbacks=[early_stopping])

# Save the fine-tuned model weights for future use
model.save_weights('lstm_model_tuned.keras')

# Predict on the validation set
y_prob = model.predict(X_val).flatten()
y_pred = (y_prob > 0.5).astype(int)  # Threshold the probabilities to get class labels

# Compute and print overall metrics
accuracy = accuracy_score(y_val, y_pred)
precision = precision_score(y_val, y_pred)
recall = recall_score(y_val, y_pred)
f1 = f1_score(y_val, y_pred)

print(f"\nValidation Metrics After Fine-Tuning:")
print(f"Accuracy: {accuracy*100:.2f}%")
print(f"Precision: {precision*100:.2f}%")
print(f"Recall: {recall*100:.2f}%")
print(f"F1 Score: {f1*100:.2f}%")
