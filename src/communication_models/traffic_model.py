import numpy as np
import torch
from torch import nn
import torch
from sklearn.linear_model import LogisticRegression
import json
import logging
from typing import Optional, List, Union

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
DEFAULT_SPIKE_SCALE = 10.0
DEFAULT_NORMAL_LOC = 1.0
DEFAULT_NORMAL_SCALE = 0.5
DEFAULT_AGE_GROUP_THRESHOLD = 50
DEFAULT_ALPHA_MULTIPLIER = 1.5

class VariationalEncoder(nn.Module):
	def __init__(self, input_dim: int, latent_dim: int, dag_weights: Union[torch.Tensor, list]):
		"""
		Variational Encoder with DAG weights.

		:param input_dim: Number of input features.
		:param latent_dim: Latent space dimensions.
		:param dag_weights: DAG weights to scale input features.
		"""
		super(VariationalEncoder, self).__init__()
		self.fc1 = nn.Linear(input_dim, 128)
		self.fc_mu = nn.Linear(128, latent_dim)
		self.fc_logvar = nn.Linear(128, latent_dim)

		# Ensure DAG weights are a torch.Tensor with the correct shape
		if isinstance(dag_weights, list):
			dag_weights = torch.tensor(dag_weights, dtype=torch.float32)
		if dag_weights.shape[0] != input_dim:
			raise ValueError(f"DAG weights must have the same length as input_dim ({input_dim}).")
		self.dag_weights = dag_weights

	def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
		"""
		Forward pass through the Variational Encoder.

		:param x: Input tensor.
		:return: Mean (mu) and log-variance (logvar) tensors.
		"""
		# Apply DAG weights to input features
		x_weighted = x * self.dag_weights  # Element-wise scaling
		h = torch.relu(self.fc1(x_weighted))
		mu = self.fc_mu(h)
		logvar = self.fc_logvar(h)
		return mu, logvar


class TelemedicineDataTrafficModel:
	def __init__(
		self,
		alpha: float,
		lambda_coef: float,
		theta1: float,
		theta2: float,
		dag_weights: Optional[Union[List[float], torch.Tensor]] = None,
		transition_matrix: Optional[List[List[float]]] = None,
		states: Optional[List[str]] = None,
		spike_probabilities: Optional[List[float]] = None,
	):
		"""
		Initialize the model with behavioral, DAG, and Markovian process parameters.

		:param alpha: Baseline weighting coefficient for utility calculation (0 < alpha <= 1).
		:param lambda_coef: Loss aversion coefficient for utility calculation.
		:param theta1: Coefficient for perceived usefulness in TAM.
		:param theta2: Coefficient for perceived ease of use in TAM.
		:param dag_weights: Weights derived from the DAG structure.
		:param transition_matrix: Markov transition probabilities (rows must sum to 1).
		:param states: States in the Markovian process.
		:param spike_probabilities: Probabilities for event spikes in each state.
		"""
		if not (0 < alpha <= 1):
			raise ValueError("alpha should be in the range (0, 1].")
		if transition_matrix and not all(np.isclose(np.sum(row), 1.0) for row in transition_matrix):
			raise ValueError("Each row in the transition matrix must sum to 1.")

		self.alpha = alpha
		self.lambda_coef = lambda_coef
		self.theta1 = theta1
		self.theta2 = theta2

		# Ensure dag_weights is converted to a tensor if provided
		if dag_weights is not None:
			if isinstance(dag_weights, list):
				self.dag_weights = torch.tensor(dag_weights, dtype=torch.float32)
			elif isinstance(dag_weights, torch.Tensor):
				self.dag_weights = dag_weights.float()
			else:
				raise ValueError("dag_weights must be a list or a torch.Tensor.")
		else:
			self.dag_weights = None

		self.transition_matrix = transition_matrix
		self.states = states
		self.spike_probabilities = spike_probabilities
		self.current_state = np.random.choice(states) if states else None


	def calculate_utility(self, x_ij: float, r_i: float) -> float:
		"""
		Calculate utility based on deterministic formulation.

		:param x_ij: The input value for utility calculation.
		:param r_i: The reference point for comparison.
		:return: Calculated utility.
		"""
		return (x_ij - r_i) ** self.alpha if x_ij >= r_i else -self.lambda_coef * (r_i - x_ij) ** self.alpha

	def calculate_beta(self, pu: float, peou: float) -> float:
		"""
		Calculate TAM coefficient.

		:param pu: Perceived usefulness.
		:param peou: Perceived ease of use.
		:return: TAM coefficient.
		"""
		return np.exp(self.theta1 * pu + self.theta2 * peou)

	def calculate_weight(self, alpha_ij: float, beta_ij: float, utility_ij: float) -> float:
		"""
		Calculate weighting function.

		:param alpha_ij: Age-adjusted alpha coefficient.
		:param beta_ij: TAM coefficient.
		:param utility_ij: Calculated utility.
		:return: Calculated weight.
		"""
		return alpha_ij * beta_ij * utility_ij

	def next_markov_state(self) -> str:
		"""
		Transition to the next state in the Markov process.

		:return: Next state.
		"""
		if not self.states or not self.transition_matrix:
			raise ValueError("Markov states or transition matrix not initialized.")
		current_state_index = self.states.index(self.current_state)
		self.current_state = np.random.choice(
			self.states, p=self.transition_matrix[current_state_index]
		)
		logger.info(f"Transitioned to next Markov state: {self.current_state}")
		return self.current_state

	def generate_event_spike(self) -> float:
		"""
		Generate an event spike based on the current state's spike probability.

		:return: Generated spike value.
		"""
		if not self.states or not self.spike_probabilities:
			raise ValueError("Markov states or spike probabilities not initialized.")
		state_index = self.states.index(self.current_state)
		if np.random.rand() < self.spike_probabilities[state_index]:
			return np.random.exponential(scale=DEFAULT_SPIKE_SCALE)
		else:
			return np.random.normal(loc=DEFAULT_NORMAL_LOC, scale=DEFAULT_NORMAL_SCALE)

	# TODO: @OMDA, Not sure if what I have done here is correct,
	# How to do the income_group, education_level?
	def generate_synthetic_traffic(
		self,
		age_group: int,
		income_group: str,
		education_level: str,
		telemedicine_usage: bool,
	) -> float:
		"""
		Generate synthetic data traffic based on demographic attributes.

		:param age_group: Age group of the user.
		:param income_group: Income group of the user.
		:param education_level: Education level of the user.
		:param telemedicine_usage: Indicator for telemedicine usage.
		:return: Generated synthetic traffic.
		"""
		r_i = np.random.uniform(0.5, 1.5)
		x_ij = np.random.uniform(0, 2) if telemedicine_usage else 0
		utility_ij = self.calculate_utility(x_ij, r_i)
		pu, peou = np.random.uniform(0, 1), np.random.uniform(0, 1)
		beta_ij = self.calculate_beta(pu, peou)
		alpha_ij = self.alpha * (
			DEFAULT_ALPHA_MULTIPLIER if age_group > DEFAULT_AGE_GROUP_THRESHOLD else 1.0
		)
		weight_ij = self.calculate_weight(alpha_ij, beta_ij, utility_ij)
		logger.info(f"Generated synthetic traffic with weight: {weight_ij:.2f}")
		return weight_ij * np.random.uniform(1, 10)
	

	def generate_synthetic_outcomes(
		self, z: torch.Tensor, treatment: torch.Tensor, decoder: nn.Module
	) -> np.ndarray:
		"""
		Generate synthetic outcomes using latent features and treatments.

		:param z: Input latent features as a Torch tensor.
		:param treatment: Treatment indicators as a Torch tensor.
		:param decoder: VariationalEncoder model.
		:return: Synthetic outcomes as a NumPy array.
		"""
		if not isinstance(z, torch.Tensor) or not isinstance(treatment, torch.Tensor):
			raise ValueError("z and treatment must be Torch tensors.")
		if self.states is None or self.transition_matrix is None:
			raise ValueError("Markov states or transition matrix not initialized.")

		synthetic_data = []
		for i in range(len(z)):
			latent_input = z[i].unsqueeze(0)  # Add batch dimension
			mu, logvar = decoder.forward(latent_input)  # Get latent representation
			
			# Apply reparameterization trick
			std = torch.exp(0.5 * logvar)  # Compute standard deviation from log-variance
			latent_sample = mu + std * torch.randn_like(std)  # Sample latent variable
			
			if treatment[i].item() == 1:  # Check if treated
				self.next_markov_state()
				spike = self.generate_event_spike()
				outcome = spike + latent_sample.mean().item()  # Combine spike with latent sample
			else:  # Not treated
				outcome = latent_sample.mean().item() + np.random.normal(loc=0.0, scale=1.0)
			
			synthetic_data.append(outcome)

		logger.info(f"Generated {len(synthetic_data)} synthetic outcomes.")
		return np.array(synthetic_data)




	def save_config(self, filepath: str) -> None:
		"""
		Save the model configuration to a file.

		:param filepath: Path to save the configuration.
		"""
		config = {
			"alpha": self.alpha,
			"lambda_coef": self.lambda_coef,
			"theta1": self.theta1,
			"theta2": self.theta2,
			"dag_weights": self.dag_weights.tolist() if self.dag_weights is not None else None,
			"transition_matrix": self.transition_matrix,
			"states": self.states,
			"spike_probabilities": self.spike_probabilities,
		}
		with open(filepath, "w") as f:
			json.dump(config, f)
		logger.info(f"Configuration saved to {filepath}")

	@classmethod
	def load_config(cls, filepath: str) -> "TelemedicineDataTrafficModel":
		"""
		Load the model configuration from a file.

		:param filepath: Path to load the configuration.
		:return: Initialized model instance.
		"""
		with open(filepath, "r") as f:
			config = json.load(f)
		logger.info(f"Configuration loaded from {filepath}")
		return cls(**config)


# Example Usage
if __name__ == "__main__":
	dag_weights = torch.tensor([0.5, 0.8, 0.6], dtype=torch.float32)
	transition_matrix = [
		[0.7, 0.2, 0.1],
		[0.3, 0.4, 0.3],
		[0.2, 0.3, 0.5],
	]
	states = ["Low Traffic", "Medium Traffic", "High Traffic"]
	spike_probabilities = [0.1, 0.3, 0.6]

	model = TelemedicineDataTrafficModel(
		alpha=0.8,
		lambda_coef=2.0,
		theta1=1.2,
		theta2=0.8,
		dag_weights=dag_weights,
		transition_matrix=transition_matrix,
		states=states,
		spike_probabilities=spike_probabilities,
	)

	age_group = 55
	income_group = "300-500k"
	education_level = "Post Grade"
	telemedicine_usage = True

	# synthetic_traffic = model.generate_synthetic_traffic(
	# 	age_group, income_group, education_level, telemedicine_usage
	# )
	# print(f"Generated Synthetic Traffic: {synthetic_traffic:.2f}")


	# Generate outcomes with Markovian process
	n_samples = 100
	input_dim = 3
	latent_dim = 2
	z = torch.randn(n_samples, input_dim)
	treatment = torch.bernoulli(torch.tensor([0.5] * n_samples))
	decoder = VariationalEncoder(input_dim=input_dim, latent_dim=latent_dim, dag_weights=dag_weights)


	synthetic_outcomes = model.generate_synthetic_outcomes(z, treatment, decoder)
	print("Synthetic Outcomes:", synthetic_outcomes)