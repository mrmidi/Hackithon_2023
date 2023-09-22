import pandas as pd
from sklearn.model_selection import train_test_split

# 1. Load the data from the CSV file
data = pd.read_csv('blocks.csv')

# 2. Shuffle the data
data = data.sample(frac=1).reset_index(drop=True)

# 3. Split the data into training, validation, and test sets
train_data, temp_data = train_test_split(data, test_size=0.2, random_state=42)
valid_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)

print(f"Training Data Shape: {train_data.shape}")
print(f"Validation Data Shape: {valid_data.shape}")
print(f"Test Data Shape: {test_data.shape}")

# 4. Save the data to CSV files
train_data.to_csv('train.csv', index=False)
valid_data.to_csv('valid.csv', index=False)
test_data.to_csv('test.csv', index=False)
