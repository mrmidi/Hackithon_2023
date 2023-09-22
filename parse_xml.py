import xml.etree.ElementTree as ET

def parse_icm_artefacts(xml_string):
    root = ET.fromstring(xml_string)
    
    artefacts_data = {}
    
    # Iterate through each SignalGroup in the XML
    for signal_group in root.findall('SignalGroup'):
        group_name = signal_group.get('Name')
        artefacts_data[group_name] = []
        
        # Extract artefact data for each SignalGroup
        for artefact in signal_group.findall('Artefact'):
            artefact_data = {
                'ModifiedBy': artefact.get('ModifiedBy'),
                'ModifiedDate': artefact.get('ModifiedDate'),
                'StartTime': artefact.get('StartTime'),
                'EndTime': artefact.get('EndTime')
            }
            artefacts_data[group_name].append(artefact_data)
    
    # Extract additional info if needed
    info = root.find('Info')
    if info is not None:
        hdf5_filename = info.get('HDF5Filename')
        artefacts_data['info'] = {'HDF5Filename': hdf5_filename}
    
    return artefacts_data

# Sample usage:
# load content from file TBI_003.artf

with open('TBI_003.artf', 'r') as f:
    xml_content = f.read()

f.close()

parsed_data = parse_icm_artefacts(xml_content)
# print(parsed_data)