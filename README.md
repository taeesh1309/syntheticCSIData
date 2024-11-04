# Wireless Network Simulation: CSI and Traffic Data Generation

This repository provides tools for simulating wireless communication channel data and generating synthetic traffic data, tailored for research and development in wireless networks, including massive MIMO and OFDMA systems. The repository consists of:
- **Channel State Information (CSI) Data Generator**: Implements small-scale and large-scale fading models, incorporating the Jakes model for temporal correlation to produce realistic CSI matrices.
- **Traffic Data Generator**: TODO

## Overview

This repository provides a flexible framework for wireless network simulations, enabling researchers and engineers to generate Channel State Information (CSI) and traffic data with high customizability. The CSI generator simulates realistic wireless channel conditions by considering:
1. **Small-scale fading**: Simulated using Rayleigh fading for magnitude and a uniform distribution for phase.
2. **Large-scale fading**: Includes path loss and shadowing models.
3. **Temporal correlation**: Implemented using the Jakes model to apply time-dependent correlation based on the Doppler effect, influenced by transmitter/receiver velocity.

The **Traffic Data Generator** TODO

## Features

- **Channel Modeling**: Supports realistic channel fading models suitable for urban macro NLOS environments, with adjustable parameters for antennas, frequency, clusters, and paths.
- **Jakes Model for Temporal Correlation**: Incorporates Doppler shift to simulate temporal correlation due to receiver mobility.
- **Large-Scale Fading**: Path loss and shadowing models for distance-based attenuation.
- **Variable Velocity**: Allows for randomized receiver velocities to simulate realistic movement patterns in wireless networks.
- **Traffic Data Simulation**: TODO

## Installation

### Prerequisites

- Python 3.7 or higher
- Libraries: `numpy`, `pandas`, `scipy`

Install the required dependencies via pip:

```bash
pip install numpy pandas scipy
```

# Repository Setup

Clone this repository:

```bash
git clone https://github.com/AhmedAredah/syntheticCSIData.git
cd syntheticCSIData
```

## Usage

### Generating CSI Data

- Configuration: Adjust parameters for transmit/receive antennas, clusters, paths, frequency, and velocity in main.py.
- Running CSI Generation: Use main.py to execute the data generation script, specifying time instances and subcarriers.
- Configuring Distance and Velocity: Adjust the distance parameter and average velocity to simulate various Doppler shifts and signal attenuations based on user mobility.

### Generating Traffic Data

TODO 

## Configuration

### Channel State Information (CSI) Generator

The CSI generator can be customized using the following parameters:

- Nt (int): Number of transmit antennas.
- Nr (int): Number of receive antennas.
- f_c (float): Carrier frequency in Hz.
- N_clusters (int): Number of multipath clusters.
- L_path (int): Number of paths per cluster.
- shadowing_std_dev (float): Standard deviation for log-normal shadowing.
- avg_velocity (float): Average receiver/transmitter velocity in m/s for Doppler shift calculation.

### Traffic Generator

TODO

## Examples

```python
csi_generator = CSIDataGenerator(Nt=64, Nr=4, f_c=5e9, N_clusters=5, L_path=3, avg_velocity=10)
timestamps = [start_time + timedelta(seconds=i) for i in range(100)]
CSI_data = csi_generator.generate_CSI_data(timestamps, subcarriers=50, distance=None)
```

## Contributing

We welcome contributions! To contribute:

1. Fork the repository.
2. Create a new branch (git checkout -b feature/YourFeature).
3. Commit your changes (git commit -m 'Add YourFeature').
4. Push to the branch (git push origin feature/YourFeature).
5. Open a Pull Request.

## License

This project is licensed under the GNU License - see the LICENSE file for details.
