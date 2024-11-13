import numpy as np
import h5py
from scipy.interpolate import interp2d

class CSIDataGenerator:
	"""
	CSIDataGenerator Class for Massive MIMO OFDMA 5G RAN Slicing
	============================================================

	This class generates synthetic Channel State Information (CSI) matrices for massive MIMO setups
	in 5G OFDMA systems, modeling RAN slicing, numerology, and propagation scenarios based on user-defined parameters.

	Attributes:
	-----------
		- Nt (int): Number of transmit antennas.
		- Nr (int): Number of receive antennas.
		- f_c (float): Main carrier frequency in Hz.
		- N_clusters (int): Number of multipath clusters.
		- L_path (int): Number of rays (paths) within each cluster.
		- shadowing_std_dev (float): Standard deviation for shadow fading in dB.
		- DL_TDD (float): Downlink Time Division Duplexing configuration in ms.
		- UL_TDD (float): Uplink Time Division Duplexing configuration in ms.
		- slice_bandwidths (list): Bandwidths allocated to slices.
		- slice_carrier_frequencies (list): Carrier frequencies for slices.

	Methods:
	--------
		- generate_channel_gain(): Generates complex channel gains with small-scale fading.
		- path_loss(d, scenario): Computes path loss based on propagation scenarios.
		- large_scale_fading(d, scenario): Calculates large-scale fading including path loss and shadowing.
		- generate_CSI_data(): Simulates the CSI matrices over given time instances and subcarriers.
	"""
 
	__SPEED_OF_LIGHT = 3e8 # Speed of light in m/s

	def __init__(self, Nt, Nr, f_c, N_clusters, L_path, DL_TDD, UL_TDD, slice_bandwidths, slice_carrier_frequencies, shadowing_std_dev=8, channel_bandwidth=None):
		self.Nt = Nt
		self.Nr = Nr
		self.f_c = f_c
		self.N_clusters = N_clusters
		self.L_path = L_path
		self.DL_TDD = DL_TDD
		self.UL_TDD = UL_TDD
		self.slice_bandwidths = slice_bandwidths
		self.slice_carrier_frequencies = slice_carrier_frequencies
		self.shadowing_std_dev = shadowing_std_dev
		self.channel_bandwidth = channel_bandwidth

		if sum(self.slice_bandwidths) > self.channel_bandwidth:
			raise ValueError("Total slice bandwidth exceeds main channel bandwidth.")

	def generate_channel_gain(self):
		"""
		Generates complex channel gains with small-scale fading effects.
		
		Returns:
		--------
			- ndarray: Complex matrix representing channel gains for each path between transmit and receive antennas.
		"""
		magnitude = np.random.rayleigh(scale=1.0, size=(self.Nt, self.Nr))
		phase = np.random.uniform(0, 2 * np.pi, size=(self.Nt, self.Nr))
		return magnitude * np.exp(1j * phase)

	def path_loss(self, d, scenario):
		"""
		Computes path loss based on the propagation scenario and distance.
		
		Parameters:
		-----------
			- d (float): Distance between transmitter and receiver.
			- scenario (str): Propagation scenario ('C2_LOS', 'C2_NLOS', 'D1_LOS').

		Returns:
		--------
			- float: Path loss in dB.
		"""
		if scenario == 'C2_LOS':
			return 40 * np.log10(d) + 13.47 - 14.0 * np.log10(25) + 6.0
		elif scenario == 'C2_NLOS':
			return (44.9 - 6.55 * np.log10(25)) * np.log10(d) + 34.46 + 8
		elif scenario == 'D1_LOS':
			return 40 * np.log10(d) + 10.5 - 18.5 * np.log10(32) + 6.0
		return 25.1 * np.log10(d) + 55.4

	def is_downlink_slot(self, time_delta):
		"""
		Determines if a given timestamp falls within a downlink or uplink slot
		based on TDD configuration.

		Parameters:
		- time_delta (timedelta): Time since the simulation started in milliseconds.

		Returns:
		- bool: True if within downlink (DL) slot, False if within uplink (UL) slot.
		"""
		total_tdd_cycle = self.DL_TDD + self.UL_TDD
		cycle_position = (time_delta.total_seconds() * 1000) % total_tdd_cycle  # Convert to ms and modulo cycle length
		return cycle_position < self.DL_TDD  # True for DL slot, False for UL slot

	def large_scale_fading(self, d, scenario):
		"""
		Calculates large-scale fading including path loss and shadowing effects.
		
		Parameters:
		-----------
			- d (ndarray): Distance array between transmitters and receivers.
			- scenario (str): Propagation scenario ('C2_LOS', 'C2_NLOS', 'D1_LOS').

		Returns:
		--------
			- ndarray: Large-scale fading in dB, including path loss and shadowing.
		"""
		PL_d = self.path_loss(d, scenario)
		shadowing = np.random.normal(0, self.shadowing_std_dev, size=d.shape)
		return PL_d + shadowing

	def generate_user_mobility(self, lambda_val, time_steps, area_size, avg_speed, speed_std):
		latitude, longitude = area_size
		user_locations = []
		num_users = np.random.poisson(lambda_val * latitude * longitude)
		x_coords = np.random.uniform(0, latitude, num_users)
		y_coords = np.random.uniform(0, longitude, num_users)
		directions = np.random.uniform(0, 2 * np.pi, num_users)  # TODO: simplified than what omda is doing. but i think what Omda did is more accurate

		for _ in range(time_steps):
			speeds = np.random.normal(avg_speed, speed_std, num_users)
			dx = speeds * np.cos(directions)
			dy = speeds * np.sin(directions)
			x_coords = np.clip(x_coords + dx, 0, latitude)
			y_coords = np.clip(y_coords + dy, 0, longitude)
			user_locations.append(np.vstack((x_coords, y_coords)).T)

		return user_locations

	def los_probability_C2(self, distances):
		return np.minimum(18 / distances, 1) * (1 - np.exp(-distances / 63)) + np.exp(-distances / 63)

	def los_probability_D1(self, distances):
		return np.exp(-distances / 1000)

	def generate_correlated_LSPs(self, num_LSPs, num_samples, correlation_matrix):
		zeta = np.random.randn(num_LSPs, num_samples)
		L = np.linalg.cholesky(correlation_matrix)
		correlated_LSPs = np.dot(L, zeta)
		return correlated_LSPs

	def generate_azimuth_angles(num_clusters, power_norm, sigma_phi_aoa, sigma_phi_aod, c_aoa, c_aod, ricean_k=None, phi_los=0):
		"""
		Generate azimuth arrival angles (AoA) and azimuth departure angles (AoD) based on WINNER II generic model description

		Parameters:
		- num_clusters (int): Number of clusters.
		- power_norm (numpy.ndarray): Normalized power values for each cluster (P_n), which we generated from the previous subroutine
		- sigma_phi_aoa (float): Standard deviation of azimuth arrival angles (σ_samll_phi).
		- sigma_phi_aod (float): Standard deviation of azimuth departure angles (σ_ϕ).
		- c_aoa (int): choose one of the integer values given in the Cluster_ASA list of cluster angluar spread angles for arrival according to the propagation scenario
		- c_aod (int): choose one of the integer values given in the Cluster_ASD list of cluster angluar spread angles for departure according to the propagation scenario
		- ricean_k (float, optional): Ricean K-factor in dB for LOS case. If None, assumes NLOS.
		- phi_los (float, optional): LOS direction defined in the network layout. Default is 0.

		Returns:
		- aoa_angles (numpy.ndarray): Generated azimuth angles of arrival (phiSmall_n).
		- aod_angles (numpy.ndarray): Generated azimuth angles of departure (ϕ_n').
		- ray_aoa_angles (numpy.ndarray): Generated azimuth angles of arrival (ϕ_n,m) for rays within clusters.
		- ray_aod_angles (numpy.ndarray): Generated azimuth angles of departure (ϕ_n,m) for rays within clusters.
		"""

		# Step 1: Calculate σ_AoA and σ_AoD using σ_samll_phi and σ_ϕ and the scaling factor 1.4
		sigma_aoa = sigma_phi_aoa / 1.4
		sigma_aod = sigma_phi_aod / 1.4

		# Step 2: Determine constant C based on the number of clusters
		C_values = {4: 0.779, 5: 0.860, 8: 1.018, 10: 1.090, 11: 1.123, 12: 1.146, 14: 1.190, 15: 1.211, 16: 1.226, 20: 1.289}

		C = C_values.get(num_clusters, 1.0)  # Default C value if not in the table

		# Step 3: Adjust C for LOS case if ricean_k is provided
		if ricean_k is not None:
			C_los = C * (1.1035 - 0.028 * ricean_k - 0.002 * ricean_k**2 + 0.0001 * ricean_k**3)
		else:
			C_los = C  # No adjustment for NLOS case

		# Step 4: Calculate AoA angles (ϕ_n') using Equation 4.8
		max_power = np.max(power_norm)
		phi_n_prime = 2 * sigma_aoa * np.sqrt(-np.log(power_norm / max_power)) / C_los
		Phi_n_prime = 2 * sigma_aod * np.sqrt(-np.log(power_norm / max_power)) / C_los

		# Step 5: Assign positive or negative sign using uniform random variable X_n
		X_n = np.random.choice([1, -1], size=num_clusters, p=[0.5, 0.5])
		phi_n_signed = X_n * phi_n_prime
		Phi_n_signed = X_n * Phi_n_prime

		# Step 6: Add random component Y_n ~ N(0, σ_AoA/5) for variation
		y_n = np.random.normal(0, sigma_aoa / 5, num_clusters)
		Y_n = np.random.normal(0, sigma_aod / 5, num_clusters)

		if phi_los != 0:
			aoa_angles = phi_n_signed + y_n + phi_los - (X_n * phi_n_prime[0] + y_n[0])
			aod_angles = Phi_n_signed + Y_n + phi_los - (X_n * Phi_n_prime[0] + Y_n[0])
		else:
			aoa_angles = phi_n_signed + y_n + phi_los
			aod_angles = Phi_n_signed + Y_n + phi_los

		# Step 7: Generate ray-specific azimuth angles (ϕ_n,m) using offsets from the table given in the manual
		# Ray offset angles (α_m) from Table 4-1 for 1° RMS angle spread
		ray_offsets = {1: 0.0447, 2: 0.0447, 3: 0.1413, 4: 0.1413, 5: 0.2492, 6: 0.2492,
					7: 0.3715, 8: 0.3715, 9: 0.5129, 10: 0.5129, 11: 0.6797, 12: 0.6797,
					13: 0.8844, 14: 0.8844, 15: 1.1481, 16: 1.1481, 17: 1.5195, 18: 1.5195, 19: 2.1551, 20: 2.1551}

		# Compute azimuth angles for rays within each cluster
		M = len(ray_offsets)  # Number of rays per cluster (20)
		ray_aoa_angles = np.zeros((num_clusters, M))
		ray_aod_angles = np.zeros((num_clusters, M))

		for n in range(num_clusters):
			for m in range(1, M + 1):
				offset_angle = ray_offsets[m]  # Retrieve offset for ray m
				sign = 1 if m % 2 == 1 else -1
				ray_aoa_angles[n, m - 1] = aoa_angles[n] + c_aoa * sign * offset_angle
				ray_aod_angles[n, m - 1] = aod_angles[n] + c_aod * sign * offset_angle

		# Step 8: Randomly couple the departure ray angles Φ_n,m to the arrival ray angles ϕ_n,m
		coupled_ray_aod_angles = np.zeros((num_clusters, M))
		for n in range(num_clusters):
			# Randomly couple each AoD angle with an AoA angle within the same cluster
			coupled_ray_aod_angles[n] = ray_aoa_angles[n][np.random.permutation(M)]

		return aoa_angles, aod_angles, ray_aoa_angles, ray_aod_angles, coupled_ray_aod_angles

	def compute_doppler_frequency(self, ray_aoa_angles, ms_speed, travel_direction, subcarrier_frequency):
		wavelength = CSIDataGenerator.__SPEED_OF_LIGHT / subcarrier_frequency
		doppler_frequency = (ms_speed * np.cos(ray_aoa_angles - travel_direction)) / wavelength
		return doppler_frequency

	def calculate_los_scaling_constant(self, ricean_k):
		"""
		Calculates the LOS scaling constant D based on the Ricean K-factor for LOS adjustments.

		Parameters:
		- ricean_k (float): Ricean K-factor in dB.

		Returns:
		- float: Scaling constant D for LOS case.
		"""
		return 0.7705 - 0.0433 * ricean_k + 0.0002 * ricean_k**2 + 0.000017 * ricean_k**3


	def generate_CSI_element(self, num_clusters, num_rays, power_norm, ray_aoa_angles, ray_aod_angles, F_tx, F_rx, wavelength, velocity, time_stamps):
		"""
		Generates a CSI element matrix with dimensions (Nt, Nr) for each timestamp.
		"""
		# Initialize the CSI matrix with the correct dimensions
		h_csi = np.zeros((self.Nt, self.Nr), dtype=complex)

		# Define azimuth and elevation angles for interpolation
		azimuth_angles = np.linspace(-180, 180, F_tx.shape[1])   # Azimuth is second dimension in F_tx
		elevation_angles = np.linspace(-90, 90, F_tx.shape[0])	# Elevation is first dimension in F_tx

		# Set up 2D interpolation functions for F_tx and F_rx over azimuth and elevation
		F_tx_interp = interp2d(azimuth_angles, elevation_angles, F_tx, kind='linear')
		F_rx_interp = interp2d(azimuth_angles, elevation_angles, F_rx, kind='linear')

		# Ensure power_norm values are non-negative and set a minimum threshold
		# TODO: @EMAD, This is the concept that i made here to work with 2 different sizes of power_norm and num_clusters
		power_norm = np.tile(power_norm, num_clusters // len(power_norm) + 1)[:num_clusters]

		# Loop over each cluster
		for n in range(num_clusters):
			sqrt_Pn = np.sqrt(power_norm[n])
			
			# Loop over each ray within the cluster
			for m in range(num_rays):
				# Calculate the Doppler effect for this ray and timestamp
				doppler_term = np.exp(1j * 2 * np.pi * velocity[m] * time_stamps[0])

				# Get interpolated F_tx and F_rx values for each azimuth/elevation angle
				F_tx_term = F_tx_interp(ray_aod_angles[m], 0)[0]  # Interpolating at the azimuth AoD and elevation 0
				F_rx_term = F_rx_interp(ray_aoa_angles[m], 0)[0]  # Interpolating at the azimuth AoA and elevation 0

				# Loop through each transmit and receive antenna to populate h_csi
				for tx in range(self.Nt):
					for rx in range(self.Nr):
						# Sum contributions from each ray with small-scale fading
						h_csi[tx, rx] += sqrt_Pn * F_tx_term * F_rx_term * doppler_term

		return h_csi



	def generate_field_pattern(self, num_elements, frequency, azimuth_angles, elevation_angles):
		"""
		Generate a 2D field pattern for a Uniform Linear Array (ULA) with isotropic elements.

		Parameters:
		- num_elements (int): Number of elements in the ULA.
		- frequency (float): Operating frequency in Hz.
		- azimuth_angles (np.ndarray): Array of azimuth angles (in degrees) for which to calculate the pattern.
		- elevation_angles (np.ndarray): Array of elevation angles (in degrees) for which to calculate the pattern.

		Returns:
		- pattern_response_db (np.ndarray): 2D array of field pattern response (in dB) for azimuth and elevation angles.
		"""
		wavelength = CSIDataGenerator.__SPEED_OF_LIGHT / frequency  # Calculate wavelength
		d = wavelength / 2  # Element spacing set to half-wavelength

		# Convert azimuth and elevation angles from degrees to radians for calculation
		azimuth_angles_rad = np.radians(azimuth_angles)
		elevation_angles_rad = np.radians(elevation_angles)

		# Calculate wave number (k)
		k = 2 * np.pi / wavelength

		# Initialize the 2D array to store the pattern response
		pattern_response = np.zeros((len(elevation_angles), len(azimuth_angles)))

		# Loop over each azimuth and elevation angle to calculate the array factor
		for i, theta in enumerate(azimuth_angles_rad):
			for j, phi in enumerate(elevation_angles_rad):
				# Calculate the array factor for this azimuth and elevation angle
				array_factor = np.sum([
					np.exp(1j * k * d * n * (np.sin(theta) * np.cos(phi)))
					for n in range(num_elements)
				])
				
				# Store the absolute magnitude (linear scale) in the 2D pattern response array
				pattern_response[j, i] = np.abs(array_factor)

		# Convert the pattern response to decibels (dB) and normalize to the maximum value
		pattern_response_db = 20 * np.log10(pattern_response / np.max(pattern_response))

		return pattern_response_db


	def generate_CSI_data(self, timestamps, num_subcarriers, scenario='C2_NLOS', output_file='csi_data.h5'):
		"""
		Generates synthetic CSI data and stores it in an HDF5 file format.
		"""

		# Set up the correlation matrix for generating correlated large-scale parameters
		correlation_matrix = np.array([
			[1.0, 0.4, 0.6, -0.4],
			[0.4, 1.0, 0.4, -0.6],
			[0.6, 0.4, 1.0, -0.3],
			[-0.4, -0.6, -0.3, 1.0]
		])
		
		# Generate user mobility locations
		user_locations = self.generate_user_mobility(lambda_val=0.002, time_steps=len(timestamps),        # This 0.002 gave ~477 users in the area
													area_size=(500, 500), avg_speed=1.0, speed_std=0.1)  # TODO: Check with Omda on these values. I think they should depend on the scenario
		
		# Open the HDF5 file in write mode
		with h5py.File(output_file, 'w') as hf:
			# Determine the number of subcarriers to use
			subcarrier_count = max(1, num_subcarriers)
			
			for t_idx, timestamp in enumerate(timestamps):
				time_since_start = timestamp - timestamps[0]
				if not self.is_downlink_slot(time_since_start):
					continue  # Skip this timestamp if in UL slot
 
				# ------------ Proceed with DL slot ------------
				# Generate a timestamp string for the dataset
				timestamp_str = timestamp.strftime("%Y%m%d%H%M%S%f")  # create unique timestamp string with high percision for each dataset

				# Current user locations at the time step
				current_locations = user_locations[t_idx]
				distances = np.linalg.norm(current_locations, axis=1)

				# Large-scale fading and LOS probability calculations
				large_scale_fade = 10 ** (-self.large_scale_fading(distances, scenario) / 20)
				los_prob = self.los_probability_C2(distances) if scenario == 'C2_LOS' else (
					self.los_probability_D1(distances) if scenario == 'D1_LOS' else np.zeros_like(distances)
				)
	
				# Adjust the number of samples in correlated_LSPs to match the number of users
				num_users = len(distances)
				correlated_LSPs = self.generate_correlated_LSPs(num_LSPs=4, num_samples=num_users, correlation_matrix=correlation_matrix)
				
				# Loop through each slice and subcarrier
				for slice_idx, slice_bw in enumerate(self.slice_bandwidths):
					f_slice = self.slice_carrier_frequencies[slice_idx]
					wavelength = 3e8 / f_slice

					# Antenna pattern generation using ULA model
					# TODO: This should use the function generate_azimuth_angles() to generate AoA and AoD angles
					azimuth_angles = np.linspace(-180, 180, 360)
					elevation_angles = np.linspace(-90, 90, 180)
					pattern_response = self.generate_field_pattern(
						num_elements=self.Nt, frequency=f_slice, azimuth_angles=azimuth_angles, elevation_angles=elevation_angles
					)

					for f in range(subcarrier_count):
						h_total = np.zeros((self.Nt, self.Nr), dtype=complex)

						# Generate CSI for each user
						for user_idx, dist in enumerate(distances):
							# Adjust CSI for each user’s LOS probability and path loss
							los_adjustment = large_scale_fade[user_idx] * (1 if los_prob[user_idx] > 0.5 else 0.9)

							# Ensure power_norm values are non-negative and set a minimum threshold
							power_norm = np.maximum(correlated_LSPs[:, user_idx], 1e-9)

							# Generate Doppler frequencies
							doppler_freqs = self.compute_doppler_frequency(
								ray_aoa_angles=azimuth_angles, ms_speed=1.0, travel_direction=np.pi / 4, subcarrier_frequency=f_slice
							)

							# Generate CSI element for each user using correlated LSPs
							h_csi_elements = self.generate_CSI_element(
								num_clusters=self.N_clusters,
								num_rays=self.L_path,
								power_norm=power_norm,  # Use clipped power_norm to avoid sqrt of negative values
								ray_aoa_angles=azimuth_angles,
								ray_aod_angles=azimuth_angles,
								F_tx=pattern_response,
								F_rx=pattern_response,
								wavelength=wavelength,
								velocity=doppler_freqs,
								time_stamps=[t_idx]
							)

							# Apply LOS adjustment to CSI data
							h_total += h_csi_elements * los_adjustment


						# Store the CSI data in the HDF5 file
						dataset_name = f"timestamp_{timestamp_str}/slice_{slice_idx}/subcarrier_{f}"
						hf.create_dataset(dataset_name, data=h_total)
						hf[dataset_name].attrs['timestamp'] = timestamp_str
						hf[dataset_name].attrs['slice_index'] = slice_idx
						hf[dataset_name].attrs['subcarrier'] = f

		print(f"CSI data successfully written to {output_file}")

