"""
CSI Data Generator for OFDMA System with Massive MIMO
======================================================

This file defines a `CSIDataGenerator` class, which models the Channel State Information (CSI) for an
Orthogonal Frequency Division Multiple Access (OFDMA) system with a Massive Multiple-Input Multiple-Output
(MIMO) setup. This class simulates channel gain matrices under an urban macro scenario with non-line-of-sight
(NLOS) conditions, using parameters such as transmit and receive antennas, frequency, clusters, and paths.

The channel gain magnitude follows a Rayleigh distribution, and the phase is modeled using a uniform
distribution. This setup reflects multipath scattering in dense urban environments without a direct line of sight.

Classes:
--------
	- CSIDataGenerator: Generates the CSI matrix for a given number of time instances and subcarrier frequencies,
	  accounting for small-scale fading (Rayleigh and uniform distributions) and large-scale fading (path loss and shadowing).

Usage:
------
	csi_generator = CSIDataGenerator(Nt=64, Nr=4, f_c=5e9, N_clusters=5, L_path=3)
	csi_data = csi_generator.generate_CSI_data(timestamps, subcarriers)

Dependencies:
-------------
	- numpy: For mathematical operations and generating random variables.
	- pandas: For structuring the generated CSI data in a DataFrame.

"""

import numpy as np
import pandas as pd

class CSIDataGenerator:
	"""
	CSIDataGenerator Class
	=======================

	This class generates Channel State Information (CSI) data matrices for a massive MIMO setup
	within an OFDMA system. The class supports a user-specified number of transmit and receive antennas,
	frequency, clusters, and paths. The CSI matrices are created for each time instance and subcarrier,
	modeling both small-scale and large-scale fading.

	Attributes:
	-----------
		- Nt (int): Number of transmit antennas.
		- Nr (int): Number of receive antennas.
		- f_c (float): Carrier frequency in Hz.
		- N_clusters (int): Number of multipath clusters.
		- L_path (int): Number of paths per cluster.
		- shadowing_std_dev (float): Standard deviation of shadowing for large-scale fading.

	Methods:
	--------
		- generate_channel_gain(): Generates a complex channel gain matrix for the small-scale fading model.
		- path_loss(d): Computes the path loss in dB based on an empirical formula for an NLOS urban macro environment.
		- large_scale_fading(d): Computes the large-scale fading (path loss + shadowing) for given distances.
		- generate_CSI_data(timestamps, subcarriers): Generates the CSI data matrices over time instances and subcarriers.

	"""
	def __init__(self, Nt, Nr, f_c, N_clusters, L_path, shadowing_std_dev=8):
		"""
		Initializes the CSIDataGenerator class with system parameters.

		Parameters:
		-----------
			- Nt (int): Number of transmit antennas.
			- Nr (int): Number of receive antennas.
			- f_c (float): Carrier frequency in Hz.
			- N_clusters (int): Number of clusters in the channel.
			- L_path (int): Number of paths within each cluster.
			- shadowing_std_dev (float): Standard deviation of shadowing for the large-scale fading (default is 8 dB).
		"""
		self.Nt = Nt				  # Number of transmit antennas
		self.Nr = Nr				  # Number of receive antennas
		self.f_c = f_c				# Carrier frequency in Hz
		self.N_clusters = N_clusters  # Number of clusters
		self.L_path = L_path		  # Number of paths per cluster
		self.shadowing_std_dev = shadowing_std_dev  # Shadowing standard deviation

	def generate_channel_gain(self):
		"""
		Generates a complex channel gain matrix for the small-scale fading model.

		This method simulates small-scale fading by generating random magnitudes and phases for each
		transmit-receive antenna pair. The magnitude follows a Rayleigh distribution, modeling
		multipath scattering with no direct line of sight, while the phase is uniformly distributed
		between 0 and 2π, reflecting random phase shifts.

		Returns:
		--------
			- numpy.ndarray: A complex matrix of size (Nt, Nr) representing the channel gain between each
			  transmit-receive antenna pair. Each element is complex, with Rayleigh-distributed magnitude
			  and uniformly distributed phase.
		"""
		magnitude = np.random.rayleigh(scale=1.0, size=(self.Nt, self.Nr))  # Rayleigh RV
		phase = np.random.uniform(0, 2 * np.pi, size=(self.Nt, self.Nr))	# Uniform phase
		return magnitude * np.exp(1j * phase)

	def path_loss(self, d):
		"""
		Computes the path loss in dB for an NLOS urban macro environment based on an empirical formula.

		This formula models the average path loss for urban environments with non-line-of-sight conditions,
		and it accounts for factors such as frequency, distance, and other urban parameters.

		Parameters:
		-----------
			- d (numpy.ndarray): An array of distances in meters between the transmitter and receiver.

		Returns:
		--------
			- numpy.ndarray: An array of path loss values in dB for each distance value in `d`.
		"""
		return 44.9 - 6.55 * np.log10(25) * np.log10(d) + 34.46 + 5.83 * np.log10(25) + 23 * np.log10(self.f_c / 5) + 8 # 8 is the effect of shadowing

	def large_scale_fading(self, d):
		"""
		Computes large-scale fading, including both path loss and shadowing effects.

		This method first calculates the path loss using the `path_loss` function. It then adds a
		shadowing component, modeled as a log-normal distribution with zero mean and standard deviation
		defined by `shadowing_std_dev`.

		Parameters:
		-----------
			- d (numpy.ndarray): An array of distances in meters between the transmitter and receiver.

		Returns:
		--------
			- numpy.ndarray: An array of large-scale fading values in dB, combining path loss and shadowing,
			  for each distance value in `d`.
		"""
		PL_d = self.path_loss(d)
		shadowing = np.random.normal(0, self.shadowing_std_dev, size=d.shape)  # Shadowing component
		return PL_d + shadowing

	def generate_CSI_data(self, timestamps, subcarriers, distance=None):
		"""
		Generates the CSI data matrices over specified time instances and subcarriers, 
		including large-scale fading effects if `distance` is provided.

		Parameters:
		-----------
			- timestamps (list): List of datetime objects representing each time instance.
			- subcarriers (int): Number of subcarrier frequencies.
			- distance (float, list, or None): Distance(s) between transmitter and receiver.
			  If it's a list, it should match the length of `timestamps` to represent varying distances over time.
			  If None, large-scale fading is ignored.

		Returns:
		--------
			- pandas.DataFrame: A DataFrame with columns `timestamp`, `subcarrier`, and `CSI_matrix`. Each row
			  represents a time instance and subcarrier frequency with the corresponding CSI matrix of size
			  (Nt, Nr), adjusted for large-scale fading if `distance` is provided.
		"""
		# Determine if large-scale fading should be applied
		apply_large_scale_fading = distance is not None
		if apply_large_scale_fading:
			if isinstance(distance, list):
				if len(distance) != len(timestamps):
					raise ValueError("If distance is a list, it must be the same length as timestamps.")
			else:
				distance = [distance] * len(timestamps)  # Uniform distance for all timestamps

		CSI_data = []
		for t, timestamp in enumerate(timestamps):
			if apply_large_scale_fading:
				d = distance[t]
				large_scale_fade = 10 ** (-self.large_scale_fading(np.array([d]))[0] / 20)  # Convert dB to linear scale
			else:
				large_scale_fade = 1  # No large-scale fading applied

			for f in range(subcarriers):
				h_total = np.zeros((self.Nt, self.Nr), dtype=complex)
				for n in range(self.N_clusters):
					for l in range(self.L_path):
						h_cluster_path = self.generate_channel_gain()
						h_total += h_cluster_path  # Sum contributions from clusters and paths
				h_total *= large_scale_fade  # Apply large-scale fading if applicable

				CSI_data.append({
					"timestamp": timestamp,
					"subcarrier": f,
					"CSI_matrix": h_total
				})
		return pd.DataFrame(CSI_data)