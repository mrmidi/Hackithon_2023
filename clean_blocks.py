import pandas as pd

def remove_blocks_from_csv(filename, start_block):
    # Load the CSV file into a DataFrame
    df = pd.read_csv(filename)

    # Drop blocks from start_block till the end
    df = df[df['block_number'] < start_block]

    # Save the modified DataFrame back to the CSV file
    df.to_csv("blocks.csv", index=False)
    print(f"Removed blocks from {start_block} till the end from {filename}")

# Sample usage
filename = 'blocks_with_labels.csv'
start_block = 13106
remove_blocks_from_csv(filename, start_block)
