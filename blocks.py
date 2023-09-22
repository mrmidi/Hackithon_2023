import datetime
import h5py as h5
import parse_xml as px
import pandas as pd

def create_blocks_with_labels(data, base_timestamp, artefacts, sampling_rate=100):
    """
    Create blocks of data and assign labels based on anomalies.
    
    Parameters:
    - data: The ABP signal data.
    - base_timestamp: The starting timestamp of the entire dataset.
    - artefacts: List of anomaly dictionaries with start and end times.
    - sampling_rate: Sampling frequency (default is 100Hz).

    Returns:
    - List of dictionaries. Each dictionary contains:
        - 'block_number': The index of the block.
        - 'data': The block's data (1000 samples).
        - 'label': 1 if the block contains an anomaly; 0 otherwise.
    """

    blocks = []
    for i in range(0, len(data), 1000):
        if (i + 1000) > len(data):  # This condition will exclude the last incomplete block
            break
        
        print(f"Processing block {i//1000}...")
        
        block_data = data[i:i+1000]
        
        block_start_time = base_timestamp + datetime.timedelta(seconds=i/sampling_rate)
        block_end_time = block_start_time + datetime.timedelta(seconds=10)  # Each block is 10 seconds long

        label = 0
        for artefact in artefacts:
            artefact_start_time = datetime.datetime.strptime(artefact['StartTime'], "%d/%m/%Y %H:%M:%S.%f")
            artefact_end_time = datetime.datetime.strptime(artefact['EndTime'], "%d/%m/%Y %H:%M:%S.%f")

            # Check if the block overlaps with the artefact
            if artefact_start_time < block_end_time and artefact_end_time > block_start_time:
                label = 1
                break

        block_info = {
            'block_number': i//1000,
            'data': block_data,
            'label': label
        }
        blocks.append(block_info)

    return blocks

# Sample usage:

with h5.File('TBI_003.hdf5', 'r') as f:
    base_timestamp = datetime.datetime.utcfromtimestamp(f['waves']['art'].attrs['index'][0][1] / 1e6)  # Convert from microseconds
    abp_data = f['waves']['art'][:]

# Open and parse the XML
with open('TBI_003.artf', 'r') as xml_file:
    xml_content = xml_file.read()

parsed_data = px.parse_icm_artefacts(xml_content)

blocks = create_blocks_with_labels(abp_data, base_timestamp, parsed_data['art'])

print(blocks[0])

df = pd.DataFrame(blocks)

# Save the DataFrame to a CSV file
df.to_csv('blocks_with_labels.csv', index=False)

print("Saved blocks_with_labels.csv")
