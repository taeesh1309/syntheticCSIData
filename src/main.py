# main.py
from datetime import datetime, timedelta
import numpy as np
from communication_models.channel_model import CSIDataGenerator
from communication_models.traffic_model import TrafficGenerator 

# Parameters. 
# TODO: These parameters we can take from a YAML file as per Dr. Nareen's requirements.
Nt = 64                           # Number of transmit antennas
Nr = 4                            # Number of receive antennas
f_c = 6e9                         # Carrier frequency in Hz (6 GHz)
N_clusters = 5                    # Number of clusters
L_path = 3                        # Number of paths per cluster
t_instances = 100                 # Number of time instances
subcarriers = 1                   # Number of subcarrier frequencies
shadowing_std_dev = 8             # dB shadowing standard deviation

# Time configuration
start_time = datetime(2024, 11, 3, 8, 0, 0)  # Example start time
end_time = datetime(2024, 11, 3, 9, 0, 0)    # Example end time
timestamps = [start_time + i * (end_time - start_time) / (t_instances - 1) for i in range(t_instances)]

# Initialize CSI Data Generator
csi_generator = CSIDataGenerator(Nt, Nr, f_c, N_clusters, L_path, shadowing_std_dev)

# Generate CSI Data
CSI_df = csi_generator.generate_CSI_data(timestamps, subcarriers)

# Output sample of CSI dataset
print("Sample CSI dataset with timestamps:")
print(CSI_df["CSI_matrix"][0])
