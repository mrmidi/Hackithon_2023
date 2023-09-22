from flask import Flask, render_template
import h5py as h5
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import parse_xml as px

import datetime

app = Flask(__name__)


SAMPLES_PER_SEGMENT = 1200  # 120 Hz * 10 seconds

@app.route('/')
def index():
    return render_template('index.html') # take template from templates folder



@app.route('/draw/<int:idx>')
def draw(idx):
    # Define start and end points based on index and segment size
    start = idx * SAMPLES_PER_SEGMENT
    end = (idx + 1) * SAMPLES_PER_SEGMENT
    segment_start_time = idx * 10
    segment_end_time = (idx + 1) * 10
    time_array = np.linspace(segment_start_time, segment_end_time, SAMPLES_PER_SEGMENT)

    # Open the HDF5 file and extract the subset of data
    with h5.File('TBI_003.hdf5', 'r') as f:
        abp_data = f['waves']['art'][start:end]

    # Open and parse the XML
    with open('TBI_003.artf', 'r') as xml_file:
        xml_content = xml_file.read()
    artefacts_data = px.parse_icm_artefacts(xml_content)
    art_artefacts = artefacts_data['art']

    # Plot the data against the time array
    plt.figure()
    plt.plot(time_array, abp_data)

    # Shade regions with artefacts
    for artefact in art_artefacts:
        artefact_start_time = (datetime.datetime.strptime(artefact['StartTime'], "%d/%m/%Y %H:%M:%S.%f") - datetime.datetime(2020, 3, 1)).total_seconds()
        artefact_end_time = (datetime.datetime.strptime(artefact['EndTime'], "%d/%m/%Y %H:%M:%S.%f") - datetime.datetime(2020, 3, 1)).total_seconds()

        # Check overlap
        if artefact_start_time < segment_end_time and artefact_end_time > segment_start_time:
            plt.axvspan(max(artefact_start_time, segment_start_time), min(artefact_end_time, segment_end_time), color='red', alpha=0.5)

    plt.title(f"ABP Waveform: Segment {idx} (10 seconds)")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude")
    
    # Save the plot
    image_path = f'plot_{idx}.png'
    plt.savefig(f'static/{image_path}')
    plt.close()

    # Return the plot
    return render_template('plot.html', image_path=image_path)




@app.route("/inspect")
def inspect_dataset():
    with h5.File('TBI_003.hdf5', 'r') as f:
        art_dataset = f['waves']['art']
        shape = art_dataset.shape
        dtype = art_dataset.dtype
        print(shape, dtype)
    return "See console for output"


@app.route('/get_keys')
def list_keys_in_waves():
    with h5.File('TBI_003.hdf5', 'r') as f:
        waves_keys = list(f['waves'].keys())
    return waves_keys

if __name__ == '__main__':
    app.run(debug=True)
