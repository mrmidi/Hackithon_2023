import h5py as h5

# Open the HDF5 file
with h5.File('TBI_003.hdf5', 'r') as f:
    # Get the length of the 'art' dataset
    art_length = len(f['waves']['art'][:])

print(f"Length of 'art' dataset: {art_length}")