############################################################################################################
############################################# WARNING ######################################################
# THIS IS NOT A WORKING VERSION OF THE CODE> RATHER IT IS ONLY A FRAMEWORK FOR REVIEW AND DISCUSSION.
# Please feel free to modify the concept but make sure that you document what you changed and why 
# so i can implement that framework later.
# Ahmed Aredah
############################################# WARNING ######################################################
############################################################################################################


import torch
import torch.nn as nn
import numpy as np
from sklearn.model_selection import train_test_split
from .channel_model import CSIDataGenerator

# This defines the Physics-Informed Neural Network (PINN) model as per Emad's recommendation for the CSI data on discord
class PINN(nn.Module):
	def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
		super(PINN, self).__init__()
		self.input_dim = input_dim
		self.hidden_dim = hidden_dim
		self.output_dim = output_dim
		self.num_layers = num_layers

		layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
		for _ in range(num_layers - 1):
			layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
		layers.append(nn.Linear(hidden_dim, output_dim))
		self.network = nn.Sequential(*layers)

	def forward(self, x):
		return self.network(x)

def compute_physics_loss_without_distances(predictions, inputs, scenario, generator):
	# Extract carrier frequency or other relevant features
	carrier_frequencies = inputs[:, 0]  # TODO: This is with the assumption that the first feature is carrier frequency in Hz. Double check this. I could just use the generator instance to get the carrier frequency. but is the slice carrier frequency the same as the carrier frequency? I dont think so

	# Estimate path loss based on carrier frequency and scenario
	effective_path_loss = 20 * torch.log10(carrier_frequencies) + 20 * torch.log10(4 * np.pi / generator.__SPEED_OF_LIGHT)

	# Large-scale fading (approximated with scenario-based assumptions)
 	# TODO: Double check with emad if this way could be correct
	if scenario == 'C2_LOS':
		shadowing_std_dev = generator.shadowing_std_dev
		large_scale_fading = torch.normal(mean=0.0, std=shadowing_std_dev, size=(predictions.size(0),))
	elif scenario == 'C2_NLOS':
		shadowing_std_dev = generator.shadowing_std_dev + 2  # increase the diversions from the LOS. we can get this from emad
		large_scale_fading = torch.normal(mean=0.0, std=shadowing_std_dev, size=(predictions.size(0),))
	else:
		large_scale_fading = torch.zeros(predictions.size(0))  # Do nothing when no scenari is selected

	# LOS Probability (scenario-dependent constant)
	if scenario == 'C2_LOS':
		los_prob = torch.ones(predictions.size(0)) * 0.8  # Higher LOS probability. we can get this from emad
	elif scenario == 'C2_NLOS':
		los_prob = torch.ones(predictions.size(0)) * 0.3  # Lower LOS probability
	else:
		los_prob = torch.ones(predictions.size(0))  # Do nothing if no scenario is selected

	# Predicted power
	predicted_power = torch.norm(predictions, dim=(1, 2))  # TODO: The predictions should have the size, Nt, Nr. We only need the power

	# Expected power approximation
	log_expected_power = (-effective_path_loss / 10) + (-large_scale_fading / 20) + torch.log10(los_prob)
	expected_power = 10 ** log_expected_power


	# Compute physics-based loss
	physics_loss = torch.mean((predicted_power - expected_power) ** 2)

	return physics_loss


class ProcessData:
	def __init__(self):
		self.mean = 0
		self.std = 0
		pass
	
	def normalize(self, data):
		self.mean = np.mean(data, axis=0)
		self.std = np.std(data, axis=0)
		return (data - self.mean) / self.std

	def denormalize(self, data):
		return data * self.std + self.mean


# TODOL LOAD THE CSI DATA HERE
def load_csi_data(file_path):
	# Discuss with emad the data that we get from the H% file
	pass

def train_pinn_model():
	# Load the data
	file_path = "file_path_to_csi_data"
	inputs, outputs = load_csi_data(file_path)

	# Normalize inputs and outputs
	# This is because the model will function better with normalize values
	# TODO: Do not forget to denormalize the outputs of the model.
	InputProcessData = ProcessData()
	OutputProcessData = ProcessData()
	inputs = InputProcessData.normalize(inputs)
	outputs = OutputProcessData.normalize(outputs)

	# Train-test split
	X_train, X_test, y_train, y_test = train_test_split(inputs, outputs, test_size=0.2, random_state=42)

	# Convert to PyTorch tensors
	X_train = torch.tensor(X_train, dtype=torch.float32)
	y_train = torch.tensor(y_train, dtype=torch.float32)
	X_test = torch.tensor(X_test, dtype=torch.float32)
	y_test = torch.tensor(y_test, dtype=torch.float32)
 
	# Initialize the PINN model
	model = PINN(input_dim=X_train.shape[1], hidden_dim=64, output_dim=y_train.shape[1], num_layers=4)
	criterion = nn.MSELoss()  # Data-driven loss
	optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)  # I am only using adam since it is popular only

	# Training loop
	epochs = 1000
	for epoch in range(epochs):
		optimizer.zero_grad()
		
		# Forward pass
		predictions = model(X_train)
		
		# Compute losses
		data_loss = criterion(predictions, y_train)  # calculate MSE between ground truth and predictions from the model
  
		# Compute physics-based loss
		# Baed on what Emad gave us
		generator = CSIDataGenerator(
			Nt=16, Nr=16, f_c=3.5e9, N_clusters=4, L_path=20,
			DL_TDD=2, UL_TDD=8, slice_bandwidths=[100e6],
			slice_carrier_frequencies=[3.5e9], shadowing_std_dev=8, channel_bandwidth=200e6
		)
		# Assume we work with C2_NLOS scenario
		scenario = 'C2_NLOS'
  
		# Compute physics-based loss without considering the distance as 
 		# per my understanding that the CSI data does not include distances
		physics_loss = compute_physics_loss_without_distances(predictions, X_train, scenario, generator)  # Physics-based loss
		loss = data_loss + physics_loss
		
		# Backpropagation and optimization
		loss.backward()
		optimizer.step()
		
		# just print the loss every 100 epochs
		if epoch % 100 == 0:
			print(f'Epoch {epoch}/{epochs}, Loss: {loss.item()}')

	# Evaluate on test set
	model.eval()
	test_predictions = model(X_test)
	test_loss = criterion(test_predictions, y_test).item()
	print(f'Test Loss: {test_loss}')
 
	return model, InputProcessData, OutputProcessData
