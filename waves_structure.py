import h5py
def get_index_field_names(filename, wave_type):
    with h5py.File(filename, 'r') as f:
        index_attr = f['waves'][wave_type].attrs['index']
        dtype = index_attr.dtype
        
        # Extracting field names and their datatypes
        field_names = [(name, dtype[name]) for name in dtype.names]
    
    return field_names

filename = 'TBI_003.hdf5'
index_fields = get_index_field_names(filename, 'art')
for name, dtype in index_fields:
    print(f"{name}: {dtype}")
