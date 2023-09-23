# import h5py
# def get_index_field_names(filename, wave_type):
#     with h5py.File(filename, 'r') as f:
#         index_attr = f['waves'][wave_type].attrs['index']
#         dtype = index_attr.dtype
        
#         # Extracting field names and their datatypes
#         field_names = [(name, dtype[name]) for name in dtype.names]
    
#     return field_names

# filename = 'TBI_003.hdf5'
# index_fields = get_index_field_names(filename, 'art')
# for name, dtype in index_fields:
#     print(f"{name}: {dtype}")



import h5py as h5

def explore_hdf5_group(group, indent=0):
    """Explore the contents of an HDF5 group."""
    items = sorted(group.items())
    for name, item in items:
        if isinstance(item, h5.Group):
            print("  " * indent + f"Group: {name}")
            explore_hdf5_group(item, indent + 1)
        elif isinstance(item, h5.Dataset):
            print("  " * indent + f"Dataset: {name}")
            print("  " * (indent + 1) + f"Shape: {item.shape}")
            print("  " * (indent + 1) + f"Datatype: {item.dtype}")

            # Print attributes if they exist
            if len(item.attrs) > 0:
                print("  " * (indent + 1) + "Attributes:")
                for attr_name, attr_value in item.attrs.items():
                    print("  " * (indent + 2) + f"{attr_name}: {attr_value}")

        else:
            print("  " * indent + f"Unknown type: {name}")


# Open the HDF5 file and explore its structure
with h5.File('TBI_003.hdf5', 'r') as f:
    explore_hdf5_group(f)
