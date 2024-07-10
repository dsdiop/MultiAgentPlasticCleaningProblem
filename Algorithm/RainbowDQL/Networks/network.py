from ..NoisyLayers.layers import NoisyLinear
import torch
from torch import nn
import torch.nn.functional as F
from ..Networks.FeatureExtractors import FeatureExtractor


class Network(nn.Module):
	def __init__(
			self,
			in_dim: int,
			out_dim: int,
			atom_size: int,
			support: torch.Tensor
	):
		"""Initialization."""
		super(Network, self).__init__()

		self.support = support 
		self.out_dim = out_dim
		self.atom_size = atom_size

		# set common feature layer
		self.feature_layer = nn.Sequential(
			nn.Linear(in_dim, 128),
			nn.ReLU(),
		)

		# set advantage layer
		self.advantage_hidden_layer = NoisyLinear(128, 128)
		self.advantage_layer = NoisyLinear(128, out_dim * atom_size)

		# set value layer
		self.value_hidden_layer = NoisyLinear(128, 128)
		self.value_layer = NoisyLinear(128, atom_size)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		"""Forward method implementation."""
		dist = self.dist(x)
		q = torch.sum(dist * self.support, dim=2)

		return q

	def dist(self, x: torch.Tensor) -> torch.Tensor:
		"""Get distribution for atoms."""
		feature = self.feature_layer(x)
		adv_hid = F.relu(self.advantage_hidden_layer(feature))
		val_hid = F.relu(self.value_hidden_layer(feature))

		advantage = self.advantage_layer(adv_hid).view(
			-1, self.out_dim, self.atom_size
		)
		value = self.value_layer(val_hid).view(-1, 1, self.atom_size)
		q_atoms = value + advantage - advantage.mean(dim=1, keepdim=True)

		dist = F.softmax(q_atoms, dim=-1)
		dist = dist.clamp(min=1e-3)  # for avoiding nans

		return dist

	def reset_noise(self):
		"""Reset all noisy_layers layers."""
		self.advantage_hidden_layer.reset_noise()
		self.advantage_layer.reset_noise()
		self.value_hidden_layer.reset_noise()
		self.value_layer.reset_noise()

class DuelingVisualNetwork(nn.Module):

	def __init__(
			self,
			in_dim: tuple,
			out_dim: int,
			number_of_features: int,
	):
		"""Initialization."""
		super(DuelingVisualNetwork, self).__init__()

		self.out_dim = out_dim

		# set common feature layer
		self.feature_layer = nn.Sequential(
			FeatureExtractor(in_dim, number_of_features),
			nn.Linear(number_of_features, 256),
			nn.ReLU(),
			nn.Linear(256, 256),
			nn.ReLU(),
			nn.Linear(256, 256),
			nn.ReLU(),
		)

		# set advantage layer
		self.advantage_hidden_layer = nn.Linear(256, 64)
		self.advantage_layer = nn.Linear(64, out_dim)

		# set value layer
		self.value_hidden_layer = nn.Linear(256, 64)
		self.value_layer = nn.Linear(64, 1)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		"""Forward method implementation."""
		feature = self.feature_layer(x)

		adv_hid = F.relu(self.advantage_hidden_layer(feature))
		val_hid = F.relu(self.value_hidden_layer(feature))

		value = self.value_layer(val_hid)
		advantage = self.advantage_layer(adv_hid)

		q = value + advantage - advantage.mean(dim=-1, keepdim=True)

		return q

class NoisyDuelingVisualNetwork(nn.Module):

	def __init__(
			self,
			in_dim: tuple,
			out_dim: int,
			number_of_features: int,
	):
		"""Initialization."""
		super(NoisyDuelingVisualNetwork, self).__init__()

		self.out_dim = out_dim

		# set common feature layer
		self.feature_layer = nn.Sequential(
			FeatureExtractor(in_dim, number_of_features))

		self.common_layer_1 = NoisyLinear(number_of_features, 256)
		self.common_layer_2 = NoisyLinear(256, 256)
		self.common_layer_3 = NoisyLinear(256, 256)

		# set advantage layer
		self.advantage_hidden_layer = NoisyLinear(256, 64)
		self.advantage_layer = NoisyLinear(64, out_dim)

		# set value layer
		self.value_hidden_layer = NoisyLinear(256, 64)
		self.value_layer = NoisyLinear(64, 1)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		"""Forward method implementation."""
		feature = self.feature_layer(x)
		feature = F.relu(self.common_layer_1(feature))
		feature = F.relu(self.common_layer_2(feature))
		feature = F.relu(self.common_layer_3(feature))

		adv_hid = F.relu(self.advantage_hidden_layer(feature))
		val_hid = F.relu(self.value_hidden_layer(feature))

		value = self.value_layer(val_hid)
		advantage = self.advantage_layer(adv_hid)

		q = value + advantage - advantage.mean(dim=-1, keepdim=True)

		return q

	def reset_noise(self):

		self.common_layer_1.reset_noise()
		self.common_layer_2.reset_noise()
		self.common_layer_3.reset_noise()

		self.advantage_hidden_layer.reset_noise()
		self.advantage_layer.reset_noise()

		self.value_hidden_layer.reset_noise()
		self.value_layer.reset_noise()


class DistributionalVisualNetwork(nn.Module):

	def __init__(
			self,
			in_dim: tuple,
			out_dim: int,
			number_of_features: int,
			num_atoms: int,
			support: torch.Tensor,
	):
		"""Initialization."""
		super(DistributionalVisualNetwork, self).__init__()

		self.out_dim = out_dim
		self.support = support
		self.num_atoms = num_atoms

		# set common feature layer
		self.feature_layer = nn.Sequential(
			FeatureExtractor(in_dim, number_of_features),
			nn.Linear(number_of_features, 256),
			nn.ReLU(),
			nn.Linear(256, 256),
			nn.ReLU(),
			nn.Linear(256, 256),
			nn.ReLU(),
			nn.Linear(256, 256),
			nn.ReLU(),
			nn.Linear(256, out_dim * num_atoms),
		)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		"""Forward method implementation. First, obtain the distributions. Later, compute the mean """

		# distribution := [batch, |A|, n_atoms]
		distribution = self.dist(x)
		q = torch.sum(distribution * self.support, dim=2)
		return q


	def dist(self, x: torch.Tensor) -> torch.Tensor:
		""" Get the value distribution for atoms """

		# Propagate to obtain the distributions [batch, |A|, n_atoms]
		q_atoms = self.feature_layer(x).view(-1, self.out_dim, self.num_atoms)
		# Softmax to transform logits into probabilities #
		probs = torch.softmax(q_atoms, dim=-1)
		# Clamp the values to avoid nans #
		return probs.clamp(min=1E-4)

class DQFDuelingVisualNetwork(nn.Module):

	def __init__(
			self,
			in_dim: tuple,
			out_dims: list,
			number_of_features: int,
			archtype: str='v1',
            nettype: str='0'
	):
		"""Initialization."""
		super(DQFDuelingVisualNetwork, self).__init__()

		self.out_dims = out_dims
		self.archtype = archtype
		if self.archtype == 'v1':
			# set common feature layer
			self.feature_layer = nn.Sequential(
				FeatureExtractor(in_dim, number_of_features, nettype),
				nn.Linear(number_of_features, 256), #256
				nn.ReLU(),
				nn.Linear(256, 256),
				nn.ReLU(),
				nn.Linear(256, 256),
				nn.ReLU(),
			)

		if self.archtype == 'v2':
			# set common feature layer
			self.feature_layer = FeatureExtractor(in_dim, number_of_features, nettype)
			self.dense_layer1 = nn.Sequential(
				nn.Linear(number_of_features, 256), #256
				nn.ReLU(),
				nn.Linear(256, 256),
				nn.ReLU(),
				nn.Linear(256, 256),
				nn.ReLU(),
			)
			self.dense_layer2 = nn.Sequential(
				nn.Linear(number_of_features, 256), #256
				nn.ReLU(),
				nn.Linear(256, 256),
				nn.ReLU(),
				nn.Linear(256, 256),
				nn.ReLU(),
			)

		# set advantage layer
		self.advantage_hidden_layer1 = nn.Linear(256, 64)
		self.advantage_layer1 = nn.Linear(64, self.out_dims[0])

		# set value layer
		self.value_hidden_layer1 = nn.Linear(256, 64)
		self.value_layer1 = nn.Linear(64, 1)

		# set advantage layer
		self.advantage_hidden_layer2 = nn.Linear(256, 64)
		self.advantage_layer2 = nn.Linear(64, self.out_dims[1])

		# set value layer
		self.value_hidden_layer2 = nn.Linear(256, 64)
		self.value_layer2 = nn.Linear(64, 1)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		"""Forward method implementation."""
		feature = self.feature_layer(x)

		if self.archtype == 'v1':
			feature1 = feature2 = feature

		if self.archtype == 'v2':
			feature1 = self.dense_layer1(feature)
			feature2 = self.dense_layer2(feature)

		adv_hid1 = F.relu(self.advantage_hidden_layer1(feature1))
		val_hid1 = F.relu(self.value_hidden_layer1(feature1))

		value1 = self.value_layer1(val_hid1)
		advantage1 = self.advantage_layer1(adv_hid1)

		q1 = value1 + advantage1 - advantage1.mean(dim=-1, keepdim=True)

		adv_hid2 = F.relu(self.advantage_hidden_layer2(feature2))
		val_hid2 = F.relu(self.value_hidden_layer2(feature2))

		value2 = self.value_layer2(val_hid2)
		advantage2 = self.advantage_layer2(adv_hid2)

		q2 = value2 + advantage2 - advantage2.mean(dim=-1, keepdim=True)
		return torch.cat((q1, q2), 1)

	def shared_parameters(self):
		return [i for i in self.feature_layer.parameters()]

	def task_specific_parameters(self):
		return [i for j,i in self.named_parameters() if 'feature_layer' not in j]

class ConcatenatedDuelingVisualNetwork(nn.Module):
    def __init__(
			self,
			in_dim: tuple,
			out_dim: int,
			number_of_features: int,
	):
        super(ConcatenatedDuelingVisualNetwork, self).__init__()

        # Define two DuelingVisualNetworks
        self.dqn1 = DuelingVisualNetwork(in_dim, out_dim, number_of_features)
        self.dqn2 = DuelingVisualNetwork(in_dim, out_dim, number_of_features)

    def forward(self, x):
        # Forward pass for each DQN
        q_values1 = self.dqn1(x)
        q_values2 = self.dqn2(x)

        # Concatenate the Q-values
        concatenated_q_values = torch.cat((q_values1, q_values2), dim=1)

        return concatenated_q_values