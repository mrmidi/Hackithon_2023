from flask import Flask, render_template
import h5py as h5
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import parse_xml as px

import json

import datetime

app = Flask(__name__)

SAMPLES_PER_SEGMENT = 1000  # 100 Hz * 10 seconds

@app.route('/')
def index():
    return render_template('index.html')  # take template from templates folder

@app.route('/draw/<int:idx>')
def draw(idx):
    with h5.File('TBI_003.hdf5', 'r') as f:
        # Get the start time of the entire dataset from the HDF5 file
        base_timestamp = datetime.datetime.utcfromtimestamp(f['waves']['art'].attrs['index'][0][1] / 1e6)  # Convert from microseconds

        # Define start and end points based on index and segment size
        start = idx * SAMPLES_PER_SEGMENT
        end = (idx + 1) * SAMPLES_PER_SEGMENT

        segment_start_time = base_timestamp + datetime.timedelta(seconds=idx*10)
        segment_end_time = segment_start_time + datetime.timedelta(seconds=10)
        time_array = np.arange(segment_start_time, segment_end_time, datetime.timedelta(seconds=10/SAMPLES_PER_SEGMENT))

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
        artefact_start_time = datetime.datetime.strptime(artefact['StartTime'], "%d/%m/%Y %H:%M:%S.%f")
        artefact_end_time = datetime.datetime.strptime(artefact['EndTime'], "%d/%m/%Y %H:%M:%S.%f")

        # Check overlap
        if artefact_start_time < segment_end_time and artefact_end_time > segment_start_time:
            plt.axvspan(max(artefact_start_time, segment_start_time), min(artefact_end_time, segment_end_time), color='red', alpha=0.5)

    plt.title(f"ABP Waveform: Segment {idx} (10 seconds)")
    plt.xlabel("Time (DD/MM/YYYY HH:MM:SS)")
    plt.ylabel("Amplitude")
    plt.xticks(rotation=45)
    plt.gca().xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%d/%m/%Y %H:%M:%S'))

    # Save the plot
    image_path = f'plot_{idx}.png'
    plt.tight_layout()
    plt.savefig(f'static/{image_path}')
    plt.close()

       # Load the results from the JSON file
    with open('results.json', 'r') as f:
        results_json = json.load(f)

    true_positives = [r['block_number'] for r in results_json if r['true_label'] == 1 and r['pred_label'] == 1]
    true_negatives = [r['block_number'] for r in results_json if r['true_label'] == 0 and r['pred_label'] == 0]
    false_positives = [r['block_number'] for r in results_json if r['true_label'] == 0 and r['pred_label'] == 1]
    false_negatives = [r['block_number'] for r in results_json if r['true_label'] == 1 and r['pred_label'] == 0]

    # print(f"True Positives: {true_positives}")
    print("True Negatives:", true_negatives)
    print("False Positives:", false_positives)
    print("False Negatives:", false_negatives)

    # Return the plot with the processed JSON data
    return render_template('plot.html', image_path=image_path, true_positives=true_positives, true_negatives=true_negatives, false_positives=false_positives, false_negatives=false_negatives)


    # Return the plot
    # return render_template('plot.html', image_path=image_path, anomalies=art_artefacts)




# @app.route("/inspect")
# def inspect_dataset():
#     with h5.File('TBI_003.hdf5', 'r') as f:
#         for group_name in f['waves']['art']
#             group = f[group_name]
#             print(f"Inspecting {group_name}:")
#             for key in group.keys():
#                 print(f"  {key}")
#                 item = group[key]
#                 if isinstance(item, h5.Group):
#                     for subkey in item.keys():
#                         print(f"    {subkey}")
#                 else:
#                     print(f"    {key} is a dataset, not a group")
    
#     return "See console for deep inspection output"

if __name__ == '__main__':
    app.run(debug=True)


@app.route('/get_keys')
def list_keys_in_waves():
    with h5.File('TBI_003.hdf5', 'r') as f:
        waves_keys = list(f['waves'].keys())
    return waves_keys

@app.route('/get_all_keys')
def list_all_keys():
    with h5.File('TBI_003.hdf5', 'r') as f:
        all_keys = list(f.keys())
    return all_keys

@app.route('/explore_keys')
def explore_hdf5():
    with h5.File('TBI_003.hdf5', 'r') as f:
        for key in f.keys():
            print(f"Contents of '{key}':")
            print(list(f[key].keys()))
            print("\n")


if __name__ == '__main__':
    app.run(debug=True)
