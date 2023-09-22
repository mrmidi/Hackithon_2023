from flask import Flask, render_template
import h5py as h5
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html') # take template from templates folder

SAMPLES_PER_SEGMENT = 1200  # 120 Hz * 10 seconds

import numpy as np

SAMPLES_PER_SEGMENT = 1200  # 120 Hz * 10 seconds

@app.route('/draw/<int:idx>')
def draw(idx):
    # Define start and end points based on index and segment size
    start = idx * SAMPLES_PER_SEGMENT
    end = (idx + 1) * SAMPLES_PER_SEGMENT

    # Generate a time array for the segment
    # Start time is idx * 10 seconds and end time is (idx + 1) * 10 seconds
    time_array = np.linspace(idx * 10, (idx + 1) * 10, SAMPLES_PER_SEGMENT)

    # Open the HDF5 file and extract the subset of data
    with h5.File('TBI_003.hdf5', 'r') as f:
        abp_data = f['waves']['art'][start:end]

    # Plot the data against the time array
    plt.figure()
    plt.plot(time_array, abp_data)
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
