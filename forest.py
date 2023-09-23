import pandas as pd

# Read the data from train.csv
train_data = pd.read_csv('train.csv')

# Display the first few rows of the dataset to get an overview
print(train_data.head())


all_features = []


from scipy.stats import skew, kurtosis
import numpy as np


def string_to_array(s):
    """Convert a string representation of a list into a numpy array."""
    return np.array([float(item) for item in s[1:-1].split()])



def compute_features(block_data):
    """
    Compute features for a given block of data.
    
    Parameters:
    - block_data: List of data samples in the block.

    Returns:
    - Dictionary containing computed features.
    """

    # Time-domain features
    mean = np.mean(block_data)
    std_dev = np.std(block_data)
    skewness = skew(block_data)
    kurt = kurtosis(block_data)
    rms = np.sqrt(np.mean(np.square(block_data)))
    peak_to_peak = np.max(block_data) - np.min(block_data)
    zero_crossings = np.where(np.diff(np.sign(block_data)))[0].shape[0]

    features = {
        'mean': mean,
        'std_dev': std_dev,
        'skewness': skewness,
        'kurtosis': kurt,
        'rms': rms,
        'peak_to_peak': peak_to_peak,
        'zero_crossings': zero_crossings
    }

    return features



# 1. Feature Extraction

# Loop through each row in the train_data DataFrame
for index, row in train_data.iterrows():
    block_data_str = row['data']
    block_data = string_to_array(block_data_str)
    features = compute_features(block_data)
    print(f"Procesing block {index}")
    features['label'] = row['label']  # Add the label to the feature dictionary
    all_features.append(features)

# Convert the list of feature dictionaries to a DataFrame
feature_df = pd.DataFrame(all_features)

# Display the first few rows of the feature DataFrame
feature_df.head()
