import sys
import os
data_path = os.path.join(os.path.dirname(__file__), '..','..','..')
sys.path.append(data_path)
import numpy as np
from collections import deque
from typing import Deque, Dict, Tuple, List, Union
#from utils import MinSegmentTree, SumSegmentTree
from Algorithm.RainbowDQL.ReplayBuffers.utils import MinSegmentTree, SumSegmentTree
import random

class ReplayBuffer:
	"""A simple numpy replay buffer."""

	def __init__(self, obs_dim: Union[tuple, int, list], size: int, save_state_in_uint8: bool = True,  batch_size: int = 32, n_step: int = 1, gamma: float = 0.99, n_agents: int = 1):
		if save_state_in_uint8:
			self.obs_buf = np.zeros([size] + list(obs_dim), dtype=np.uint8)
			self.next_obs_buf = np.zeros([size] + list(obs_dim), dtype=np.uint8)
		else:
			self.obs_buf = np.zeros([size] + list(obs_dim), dtype=np.float32)
			self.next_obs_buf = np.zeros([size] + list(obs_dim), dtype=np.float32)
		self.acts_buf = np.zeros([size], dtype=np.float32)
		self.rews_buf = np.zeros([size], dtype=np.float32)
		self.done_buf = np.zeros(size, dtype=np.float32)
		self.info_buf = np.empty([size], dtype=dict)
		self.max_size, self.batch_size = size, batch_size
		self.ptr, self.size, = 0, 0

		# for N-step Learning
		self.n_agents = n_agents
		self.n_step_buffers = [deque(maxlen=n_step) for _ in range(n_agents)]
		self.n_step = n_step
		self.gamma = gamma

	def store(self, agent_id: int, obs: np.ndarray, act: np.ndarray, rew: float, next_obs: np.ndarray, done: bool, info: dict) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray, bool, dict]:

		# Check if the agent ID is valid
		assert 0 <= agent_id < self.n_agents, "Invalid agent ID"

		transition = (obs, act, rew, next_obs, done, info)
		
		self.n_step_buffers[agent_id].append(transition)

		# single step transition is not ready
		if len(self.n_step_buffers[agent_id]) < self.n_step:
			return ()

		# make a n-step transition
		rew, next_obs, done, info = self._get_n_step_info(self.n_step_buffers[agent_id], self.gamma)
		obs, act = self.n_step_buffers[agent_id][0][:2]

		self.obs_buf[self.ptr] = obs
		self.next_obs_buf[self.ptr] = next_obs
		self.acts_buf[self.ptr] = act
		self.rews_buf[self.ptr] = rew
		self.done_buf[self.ptr] = done
		self.info_buf[self.ptr] = info
		self.ptr = (self.ptr + 1) % self.max_size
		self.size = min(self.size + 1, self.max_size)

		return self.n_step_buffers[agent_id][0]

	def sample_batch(self) -> Dict[str, np.ndarray]:
		idxs = np.random.choice(self.size, size=self.batch_size, replace=False)

		return dict(
			obs=self.obs_buf[idxs],
			next_obs=self.next_obs_buf[idxs],
			acts=self.acts_buf[idxs],
			rews=self.rews_buf[idxs],
			done=self.done_buf[idxs],
			info=self.info_buf[idxs],
			# for N-step Learning
			indices=idxs,
		)

	def sample_batch_from_idxs(self, idxs: np.ndarray) -> Dict[str, np.ndarray]:
		# for N-step Learning

		return dict(
			obs=self.obs_buf[idxs],
			next_obs=self.next_obs_buf[idxs],
			acts=self.acts_buf[idxs],
			rews=self.rews_buf[idxs],
			done=self.done_buf[idxs],
			info=self.info_buf[idxs],
		)

	@staticmethod
	def _get_n_step_info(n_step_buffer: Deque, gamma: float) -> Tuple[np.int64, np.ndarray, bool, dict]:
		"""Return n step rew, next_obs, and done."""
		# info of the last transition
		rew, next_obs, done, info = n_step_buffer[-1][-4:]

		for transition in reversed(list(n_step_buffer)[:-1]):
			r, n_o, d = transition[-3:]
			rew = r + gamma * rew * (1 - d)
			next_obs, done = (n_o, d) if d else (next_obs, done)

		return rew, next_obs, done, info

	def __len__(self) -> int:
		return self.size


class ReplayBufferNrewards:
	"""A simple numpy replay buffer."""

	def __init__(self, obs_dim: Union[tuple, int, list], size: int, save_state_in_uint8: bool = True,  batch_size: int = 32, n_step: int = 1, gamma: float = 0.99, Nrewards: int = 2, n_agents: int = 1):
		if save_state_in_uint8:
			self.obs_buf = np.zeros([size] + list(obs_dim), dtype=np.uint8)
			self.next_obs_buf = np.zeros([size] + list(obs_dim), dtype=np.uint8)
		else:
			self.obs_buf = np.zeros([size] + list(obs_dim), dtype=np.float32)
			self.next_obs_buf = np.zeros([size] + list(obs_dim), dtype=np.float32)
		self.acts_buf = np.zeros([size], dtype=np.float32)
		self.rews_buf = np.zeros([size, Nrewards], dtype=np.float32)
		self.done_buf = np.zeros(size, dtype=np.float32)
		self.info_buf = np.empty([size], dtype=dict)
		self.max_size, self.batch_size = size, batch_size
		self.ptr, self.size, = 0, 0

		# for N-step Learning
		self.n_agents = n_agents
		self.n_step_buffers = [deque(maxlen=n_step) for _ in range(n_agents)]
		self.n_step = n_step
		self.gamma = gamma

	def store(self, agent_id: int, obs: np.ndarray, act: np.ndarray, rew: float, next_obs: np.ndarray, done: bool, info: dict) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray, bool, dict]:
		
		# Check if the agent ID is valid
		assert 0 <= agent_id < self.n_agents, "Invalid agent ID"

		transition = (obs, act, rew, next_obs, done, info)
		
		self.n_step_buffers[agent_id].append(transition)

		# single step transition is not ready
		if len(self.n_step_buffers[agent_id]) < self.n_step:
			return ()

		# make a n-step transition
		rew, next_obs, done, info = self._get_n_step_info(self.n_step_buffers[agent_id], self.gamma)
		obs, act = self.n_step_buffers[agent_id][0][:2]

		self.obs_buf[self.ptr] = obs
		self.next_obs_buf[self.ptr] = next_obs
		self.acts_buf[self.ptr] = act
		self.rews_buf[self.ptr] = rew
		self.done_buf[self.ptr] = done
		self.info_buf[self.ptr] = info
		self.ptr = (self.ptr + 1) % self.max_size
		self.size = min(self.size + 1, self.max_size)

		return self.n_step_buffers[agent_id][0]

	def sample_batch(self) -> Dict[str, np.ndarray]:
		idxs = np.random.choice(self.size, size=self.batch_size, replace=False)

		return dict(
			obs=self.obs_buf[idxs],
			next_obs=self.next_obs_buf[idxs],
			acts=self.acts_buf[idxs],
			rews=self.rews_buf[idxs],
			done=self.done_buf[idxs],
			info=self.info_buf[idxs],
			# for N-step Learning
			indices=idxs,
		)

	def sample_batch_from_idxs(self, idxs: np.ndarray) -> Dict[str, np.ndarray]:
		# for N-step Learning

		return dict(
			obs=self.obs_buf[idxs],
			next_obs=self.next_obs_buf[idxs],
			acts=self.acts_buf[idxs],
			rews=self.rews_buf[idxs],
			done=self.done_buf[idxs],
			info=self.info_buf[idxs],
		)

	@staticmethod
	def _get_n_step_info(n_step_buffer: Deque, gamma: float) -> Tuple[np.int64, np.ndarray, bool, dict]:
		"""Return n step rew, next_obs, and done."""
		# Find the index of the first step where done=True in the n_step_buffer.
		done_index = next((i for i, transition in enumerate(n_step_buffer) if transition[-2]), len(n_step_buffer))
		n_step_buffer_truncated = list(n_step_buffer)[:done_index+1]
		
		# info of the last transition
		rew, next_obs, done, info = n_step_buffer_truncated[-1][-4:]

		for transition in reversed(n_step_buffer_truncated[:-1]):
			r, n_o, d, _ = transition[-4:]
			rew = r + gamma * rew * (1 - d)
			next_obs, done = (n_o, d) if d else (next_obs, done)

		return rew, next_obs, done, info

	def __len__(self) -> int:
		return self.size



class PrioritizedReplayBuffer(ReplayBuffer):
	"""Prioritized Replay buffer.

	Attributes:
		max_priority (float): max priority
		tree_ptr (int): next index of tree
		alpha (float): alpha parameter for prioritized replay buffer
		sum_tree (SumSegmentTree): sum tree for prior
		min_tree (MinSegmentTree): min tree for min prior to get max weight

	"""

	def __init__(self, obs_dim: Union[tuple, int, list], size: int, save_state_in_uint8: bool = True, batch_size: int = 32, alpha: float = 0.6, n_step: int = 1, gamma: float = 0.99, n_agents: int = 1):
		"""Initialization."""
		assert alpha >= 0

		super().__init__(obs_dim, size, save_state_in_uint8, batch_size, n_step, gamma, n_agents)

		self.max_priority, self.tree_ptr = 1.0, 0
		self.alpha = alpha

		# capacity must be positive and a power of 2.
		tree_capacity = 1
		while tree_capacity < self.max_size:
			tree_capacity *= 2

		self.sum_tree = SumSegmentTree(tree_capacity)
		self.min_tree = MinSegmentTree(tree_capacity)

	def store(
			self,
			agent_id: int,
			obs: np.ndarray,
			act: int,
			rew: float,
			next_obs: np.ndarray,
			done: bool,
			info: dict,
	) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray, bool, dict]:
		"""Store experience and priority."""
		transition = super().store(agent_id, obs, act, rew, next_obs, done, info)

		if transition:
			self.sum_tree[self.tree_ptr] = self.max_priority ** self.alpha
			self.min_tree[self.tree_ptr] = self.max_priority ** self.alpha
			self.tree_ptr = (self.tree_ptr + 1) % self.max_size

		return transition

	def sample_batch(self, beta: float = 0.4) -> Dict[str, np.ndarray]:
		"""Sample a batch of experiences."""
		assert len(self) >= self.batch_size
		assert beta > 0

		indices = self._sample_proportional()

		obs = self.obs_buf[indices]
		next_obs = self.next_obs_buf[indices]
		acts = self.acts_buf[indices]
		rews = self.rews_buf[indices]
		done = self.done_buf[indices]
		info = self.info_buf[indices]
		weights = np.array([self._calculate_weight(i, beta) for i in indices])

		return dict(
			obs=obs,
			next_obs=next_obs,
			acts=acts,
			rews=rews,
			done=done,
			info=info,
			weights=weights,
			indices=indices,
		)

	def update_priorities(self, indices: List[int], priorities: np.ndarray):
		"""Update priorities of sampled transitions."""
		assert len(indices) == len(priorities)

		for idx, priority in zip(indices, priorities):
			assert priority > 0
			assert 0 <= idx < len(self)

			self.sum_tree[idx] = priority ** self.alpha
			self.min_tree[idx] = priority ** self.alpha

			self.max_priority = max(self.max_priority, priority)

	def _sample_proportional(self) -> List[int]:
		"""Sample indices based on proportions."""
		indices = []
		p_total = self.sum_tree.sum(0, len(self) - 1)
		segment = p_total / self.batch_size

		for i in range(self.batch_size):
			a = segment * i
			b = segment * (i + 1)
			upperbound = random.uniform(a, b)
			idx = self.sum_tree.retrieve(upperbound)
			indices.append(idx)

		return indices

	def _calculate_weight(self, idx: int, beta: float):
		"""Calculate the weight of the experience at idx."""
		# get max weight
		p_min = self.min_tree.min() / self.sum_tree.sum()
		max_weight = (p_min * len(self)) ** (-beta)

		# calculate weights
		p_sample = self.sum_tree[idx] / self.sum_tree.sum()
		weight = (p_sample * len(self)) ** (-beta)
		weight = weight / max_weight

		return weight


class PrioritizedReplayBufferNrewards(ReplayBufferNrewards):
	"""Prioritized Replay buffer.

	Attributes:
		max_priority (float): max priority
		tree_ptr (int): next index of tree
		alpha (float): alpha parameter for prioritized replay buffer
		sum_tree (SumSegmentTree): sum tree for prior
		min_tree (MinSegmentTree): min tree for min prior to get max weight

	"""

	def __init__(self, obs_dim: Union[tuple, int, list], size: int, save_state_in_uint8: bool = True, batch_size: int = 32, alpha: float = 0.6, n_step: int = 1, gamma: float = 0.99, Nrewards: int = 2, n_agents: int = 1):
		"""Initialization."""
		assert alpha >= 0

		super().__init__(obs_dim=obs_dim, size=size, save_state_in_uint8=save_state_in_uint8, batch_size=batch_size, n_step=n_step, gamma=gamma, Nrewards=Nrewards, n_agents=n_agents)

		self.max_priority, self.tree_ptr = 1.0, 0
		self.alpha = alpha

		# capacity must be positive and a power of 2.
		tree_capacity = 1
		while tree_capacity < self.max_size:
			tree_capacity *= 2

		self.sum_tree = SumSegmentTree(tree_capacity)
		self.min_tree = MinSegmentTree(tree_capacity)

	def store(
			self,
			agent_id: int,
			obs: np.ndarray,
			act: int,
			rew: Union[float, list, tuple, np.ndarray],
			next_obs: np.ndarray,
			done: bool,
			info: dict,
	) -> Tuple[np.ndarray, np.ndarray, Union[float, list, tuple, np.ndarray], np.ndarray, bool, dict]:
		"""Store experience and priority."""
		transition = super().store(agent_id, obs, act, rew, next_obs, done, info)

		if transition:
			self.sum_tree[self.tree_ptr] = self.max_priority ** self.alpha
			self.min_tree[self.tree_ptr] = self.max_priority ** self.alpha
			self.tree_ptr = (self.tree_ptr + 1) % self.max_size

		return transition

	def sample_batch(self, beta: float = 0.4) -> Dict[str, np.ndarray]:
		"""Sample a batch of experiences."""
		assert len(self) >= self.batch_size
		assert beta > 0

		indices = self._sample_proportional()

		obs = self.obs_buf[indices]
		next_obs = self.next_obs_buf[indices]
		acts = self.acts_buf[indices]
		rews = self.rews_buf[indices]
		done = self.done_buf[indices]
		info = self.info_buf[indices]
		weights = np.array([self._calculate_weight(i, beta) for i in indices])

		return dict(
			obs=obs,
			next_obs=next_obs,
			acts=acts,
			rews=rews,
			done=done,
			info=info,
			weights=weights,
			indices=indices,
		)

	def update_priorities(self, indices: List[int], priorities: np.ndarray):
		"""Update priorities of sampled transitions."""
		assert len(indices) == len(priorities)

		for idx, priority in zip(indices, priorities):
			assert priority > 0
			assert 0 <= idx < len(self)

			self.sum_tree[idx] = priority ** self.alpha
			self.min_tree[idx] = priority ** self.alpha

			self.max_priority = max(self.max_priority, priority)

	def _sample_proportional(self) -> List[int]:
		"""Sample indices based on proportions."""
		indices = []
		p_total = self.sum_tree.sum(0, len(self) - 1)
		segment = p_total / self.batch_size

		for i in range(self.batch_size):
			a = segment * i
			b = segment * (i + 1)
			upperbound = random.uniform(a, b)
			idx = self.sum_tree.retrieve(upperbound)
			indices.append(idx)

		return indices

	def _calculate_weight(self, idx: int, beta: float):
		"""Calculate the weight of the experience at idx."""
		# get max weight
		p_min = self.min_tree.min() / self.sum_tree.sum()
		max_weight = (p_min * len(self)) ** (-beta)

		# calculate weights
		p_sample = self.sum_tree[idx] / self.sum_tree.sum()
		weight = (p_sample * len(self)) ** (-beta)
		weight = weight / max_weight

		return weight


if __name__ == '__main__':
	my_buffer = PrioritizedReplayBufferNrewards((10,5),100,n_step=5,n_agents=4, Nrewards = 2)
	
	steps = 0
	for _ in range(60):
		 # Genera un número aleatorio entre 0 y 1
		numero_aleatorio = random.random() 
		# Compara el número aleatorio con la probabilidad dada
		if numero_aleatorio < 0.03:
			done = True
		else:
			done = False
		agent_id = _ % my_buffer.n_agents 
		if agent_id == 0:
			steps += 1

		my_buffer.store(agent_id,np.zeros((10,5)),random.randint(0,7),np.asarray([random.randint(0,20),random.randint(0,20)]),np.zeros((10,5)),done,{'agent_id': agent_id, 'steps': steps})
	print(my_buffer.sample_batch())
	pass



