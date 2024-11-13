from datetime import datetime, timedelta
import yaml
import os
from communication_models.channel_model import CSIDataGenerator

def load_parameters(yaml_file):
    with open(yaml_file, 'r') as file:
        params = yaml.safe_load(file)
    return params

# Load configuration file
current_dir = os.path.dirname(os.path.abspath(__file__))
config_file = os.path.join(current_dir, 'config.yaml')
params = load_parameters(config_file)

# Extract time configuration
start_time = datetime.fromisoformat(params['start_time'])
end_time = datetime.fromisoformat(params['end_time'])
interval = timedelta(seconds=params['interval_seconds'])

timestamps = []
current_time = start_time

# Generate timestamps based on the interval
while current_time <= end_time:
    timestamps.append(current_time)
    current_time += interval


# Initialize CSI Data Generator with all required parameters
csi_generator = CSIDataGenerator(
    Nt=params['Nt'], 
    Nr=params['Nr'], 
    f_c=params['f_c'], 
    N_clusters=params['N_clusters'], 
    L_path=params['L_path'], 
    DL_TDD=params['DL_TDD'],
    UL_TDD=params['UL_TDD'],
    slice_bandwidths=params['slice_bandwidths'],
    slice_carrier_frequencies=params['slice_carrier_frequencies'],
    shadowing_std_dev=params['shadowing_std_dev'],
    channel_bandwidth=params['channel_bandwidth']
)

# Generate CSI Data
# This function will dump the created CSI data to the h5 file
csi_generator.generate_CSI_data(timestamps, params['subcarriers'], scenario='C2_NLOS', output_file='csi_data.h5')
