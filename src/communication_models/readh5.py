import numpy as np
import h5py
import csv

def extract_and_save_csi_to_csv(file_path, output_csv):
    # Open the HDF5 file
    with h5py.File(file_path, 'r') as h5_file:
        for group_name in ['ue_uplink']:
            print(f"\nProcessing group: {group_name}")
            try:
                # Access datasets
                hdl = h5_file[f"{group_name}/hdl"][:]
                ch = h5_file[f"{group_name}/ch"][:]
                frame = h5_file[f"{group_name}/frame"][:]
                bin_data = h5_file[f"{group_name}/bin"][:]
                data = h5_file[f"{group_name}/data"][:]
                
                # Prepare the CSV file
                with open(output_csv, mode='w', newline='') as file:
                    writer = csv.writer(file)
                    
                    # Write the header
                    writer.writerow(['hdl', 'ch', 'frame', 'bin_data', 'data real', 'data unreal'])
                    
                    # Number of rows per group
                    rows_per_group = 256
                    
                    # Reshape data to extract real and imaginary parts
                    reshaped_data = data.reshape(-1, 2)
                    real_parts = reshaped_data[:, 0]
                    imaginary_parts = reshaped_data[:, 1]
                    
                    # Iterate through the groups
                    for i in range(len(hdl)):
                        # Slice the data corresponding to the current group
                        start_idx = i * rows_per_group
                        end_idx = (i + 1) * rows_per_group

                        for j in range(start_idx, end_idx):
                            if real_parts[j] != 0 and imaginary_parts[j] != 0:
                                # Write the data to the CSV file
                                writer.writerow([
                                    hdl[i], ch[i], frame[i], bin_data[i],
                                    real_parts[j], imaginary_parts[j]
                                ])
                
                print(f"Data successfully written to {output_csv}")

            except KeyError as e:
                print(f"Dataset not found in {group_name}: {e}")

# Define file paths
input_file_path = 'csi_563000000.0_6000000.0_06_26_2023_15_18_40.h5'
output_csv_path = 'csi_output1.csv'

# Process and save data to CSV
extract_and_save_csi_to_csv(input_file_path, output_csv_path)
