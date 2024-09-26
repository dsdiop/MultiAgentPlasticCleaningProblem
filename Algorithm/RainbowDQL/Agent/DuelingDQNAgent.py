import sys
import os
data_path = os.path.join(os.path.dirname(__file__), '..','..','..')
sys.path.append(data_path)
from typing import Dict, List, Tuple
import gym
import numpy as np
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from Algorithm.RainbowDQL.ReplayBuffers.ReplayBuffers import PrioritizedReplayBuffer, ReplayBuffer,  PrioritizedReplayBufferNrewards
from Algorithm.RainbowDQL.Networks.network import DuelingVisualNetwork, NoisyDuelingVisualNetwork, DistributionalVisualNetwork
from Algorithm.RainbowDQL.Networks.network import DQFDuelingVisualNetwork, ConcatenatedDuelingVisualNetwork
from Algorithm.RainbowDQL.ActionMasking.ActionMaskingUtils import SafeActionMasking, ConsensusSafeActionMasking
import torch.nn.functional as F
from tqdm import trange
from copy import copy
import json
import os
from collections import deque
from Algorithm.RainbowDQL.Agent.methods.weight_methods import WeightMethods
from Evaluation.Utils.EvaluationUtils import prewarm_buffer
class MultiAgentDuelingDQNAgent:

	def __init__(
			self,
			env: gym.Env,
			memory_size: int,
			batch_size: int,
			target_update: int,
			soft_update: bool = False,
			tau: float = 0.0001,
			epsilon_values: List[float] = [1.0, 0.0],
			epsilon_interval: List[float] = [0.0, 1.0],
			learning_starts: int = 10,
			gamma: float = 0.99,
			lr: float = 1e-4,
			# PER parameters
			alpha: float = 0.2,
			beta: float = 0.6,
			prior_eps: float = 1e-6,
			n_steps: int = 1,
			prewarmed_memory: None = None,
			# NN parameters
			number_of_features: int = 1024,
			noisy: bool = False,
            nettype: str='0',
			archtype: str='v1',
			device='cuda:1',
			weighted: bool=False,
			# Distributional parameters #
			distributional: bool = False,
			num_atoms: int = 51,
			v_interval: Tuple[float, float] = (0.0, 100.0),
			logdir=None,
			log_name="Experiment",
			save_every=None,
			train_every=1,
			# Choose Q-function
			use_nu: bool = False,
			nu_intervals=[[0., 1], [0.5, 1.], [0.5, 0.], [1., 0.]],
			concatenatedDQN = False,
			# Evaluation
			eval_every=None,
			eval_episodes=1000,
			# Masked actions
			masked_actions= True,
			consensus= True,
      weighting_method=None,
      weight_methods_parameters = None
	):
		"""

		:param env: Environment to optimize
		:param memory_size: Size of the experience replay
		:param batch_size: Mini-batch size for SGD steps
		:param target_update: Number of episodes between updates of the target
		:param soft_update: Flag to activate the Polyak update of the target
		:param tau: Polyak update constant
		:param gamma: Discount Factor
		:param lr: Learning Rate
		:param alpha: Randomness of the sample in the PER
		:param beta: Bias compensating constant in the PER weights
		:param prior_eps: Minimal probability for every experience to be samples
		:param number_of_features: Number of features after the visual extractor
		:param logdir: Directory to save the tensorboard log
		:param log_name: Name of the tb log
		"""

		""" Logging parameters """
		self.logdir = logdir
		self.experiment_name = log_name
		self.writer = None
		self.save_every = save_every
		self.eval_every = eval_every
		self.eval_episodes = eval_episodes

		""" Observation space dimensions """
		obs_dim = env.observation_space.shape
		action_dim = env.action_space.n
		self.action_dim = [8, action_dim - 8]
		""" Agent embeds the environment """
		self.env = env
		self.batch_size = batch_size
		self.target_update = target_update
		self.soft_update = soft_update
		self.tau = tau
		self.gamma = gamma
		self.learning_rate = lr
		self.epsilon_values = epsilon_values
		self.epsilon_interval = epsilon_interval
		self.epsilon = self.epsilon_values[0]
		self.learning_starts = learning_starts
		self.noisy = noisy
		self.distributional = distributional
		self.v_interval = v_interval
		self.num_atoms = num_atoms
		self.train_every = train_every
		self.nettype = nettype
		self.archtype = archtype
		self.weighted = weighted
		self.masked_actions = masked_actions
		self.consensus = consensus
		self.n_steps = n_steps
		self.use_nu = use_nu
		if self.use_nu:
			self.nu_intervals = nu_intervals
			self.nu = self.nu_intervals[0][1]
		self.concatenatedDQN = concatenatedDQN
		""" Automatic selection of the device """
		self.device = torch.device(device if torch.cuda.is_available() else "cpu")
		#self.device = torch.device("cpu")

		print("Selected device: ", self.device)

		""" Prioritized Experience Replay """
		self.beta = beta
		self.beta_init = beta
		self.prior_eps = prior_eps
		self.save_state_in_uint8 = self.env.convert_to_uint8
		if self.use_nu:
			self.memory = PrioritizedReplayBufferNrewards(obs_dim, memory_size, save_state_in_uint8=self.save_state_in_uint8,
                                                 		batch_size=batch_size, alpha=alpha, n_step=n_steps, gamma=gamma,
                                                   n_agents=self.env.number_of_agents)
		else:
			self.memory = PrioritizedReplayBuffer(obs_dim, memory_size, save_state_in_uint8=self.save_state_in_uint8,
                                                 		batch_size=batch_size, alpha=alpha, n_step=n_steps, gamma=gamma,
                                                   n_agents=self.env.number_of_agents)
		if prewarmed_memory is not None:
			import pickle
			if os.path.exists(prewarmed_memory):
				with open(prewarmed_memory, 'rb') as f:	
					prewarmed_buffer = pickle.load(f)
			else:
				prewarmed_buffer = prewarm_buffer("", self.env, 1500, self.env.number_of_agents, 
									self.env.ground_truth_type, nu_intervals=self.nu_intervals, memory=self.memory)
				os.makedirs(os.path.dirname(prewarmed_memory), exist_ok=True)
				with open(prewarmed_memory, 'wb') as f:
					pickle.dump(prewarmed_buffer, f)
			self.memory=prewarmed_buffer
		""" Create the DQN and the DQN-Target (noisy if selected) """
		if self.noisy:
			self.dqn = NoisyDuelingVisualNetwork(obs_dim, action_dim, number_of_features).to(self.device)
			self.dqn_target = NoisyDuelingVisualNetwork(obs_dim, action_dim, number_of_features).to(self.device)
		elif self.distributional:
			self.support = torch.linspace(self.v_interval[0], self.v_interval[1], self.num_atoms).to(self.device)
			self.dqn = DistributionalVisualNetwork(obs_dim, action_dim, number_of_features, num_atoms, self.support).to(self.device)
			self.dqn_target = DistributionalVisualNetwork(obs_dim, action_dim, number_of_features, num_atoms, self.support).to(self.device)
		elif self.use_nu:
			if not self.concatenatedDQN:
				self.dqn = DQFDuelingVisualNetwork(obs_dim, self.action_dim, number_of_features,archtype,nettype).to(self.device)
				self.dqn_target = DQFDuelingVisualNetwork(obs_dim, self.action_dim, number_of_features,archtype,nettype).to(self.device)
			else:
				self.dqn = ConcatenatedDuelingVisualNetwork(obs_dim, self.action_dim, number_of_features).to(self.device)
				self.dqn_target = ConcatenatedDuelingVisualNetwork(obs_dim, self.action_dim, number_of_features).to(self.device)
			self.loss_expl = deque(maxlen=3)
			self.loss_inf = deque(maxlen=3)
			self.weighting_method_name = weighting_method
			self.weighting_method = None
			if weighting_method is not None:
				self.weighting_method = WeightMethods(weighting_method, n_tasks=2, device=self.device, **weight_methods_parameters)
		else:
			self.dqn = DuelingVisualNetwork(obs_dim, action_dim, number_of_features).to(self.device)
			self.dqn_target = DuelingVisualNetwork(obs_dim, action_dim, number_of_features).to(self.device)

		self.dqn_target.load_state_dict(self.dqn.state_dict())
		self.dqn_target.eval()

		""" Optimizer """
		self.optimizer = optim.Adam(self.dqn.parameters(), lr=self.learning_rate)

		""" Actual list of transitions """
		self.transition = list()

		""" Evaluation flag """
		self.is_eval = False

		""" Data for logging """
		self.episodic_reward = []
		self.episodic_loss = []
		self.episodic_length = []
		self.episode = 0

		# Sample new noisy parameters
		if self.noisy:
			self.dqn.reset_noise()
			self.dqn_target.reset_noise()

		# Masking utilities #
		if self.masked_actions:
			self.safe_masking_module = SafeActionMasking(action_space_dim = self.action_dim[0], movement_length = self.env.movement_length)
		
		if self.consensus:
			self.consensus_safe_action_masking = ConsensusSafeActionMasking(self.env.scenario_map, action_space_dim = self.action_dim[0], movement_length = self.env.movement_length)
			self.q_values4consensus = {agent_id: np.zeros((1, self.env.action_space.n)) for agent_id in range(self.env.number_of_agents)}
	# TODO: Implement an annealed Learning Rate (see:
	#  https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.ReduceLROnPlateau.html#torch.optim.lr_scheduler.ReduceLROnPlateau)
	def choose_q_function(self, state: np.ndarray):

		"""Select an action from the input state. If deterministic, no noise is applied. """

		if self.epsilon > np.random.rand() and not self.noisy:
			selected_action = self.env.action_space.sample()

		else:
			q_values = self.dqn(torch.FloatTensor(state).unsqueeze(0).to(self.device)).detach().cpu().numpy()
			if self.nu > np.random.rand():
				selected_action = np.argmax(q_values.squeeze(0)[:self.action_dim[0]])
			else:
				selected_action = np.argmax(q_values.squeeze(0)[self.action_dim[0]:])


		return selected_action

	def predict_action(self, state: np.ndarray):

		"""Select an action from the input state. If deterministic, no noise is applied. """

		if self.epsilon > np.random.rand() and not self.noisy:
			selected_action = self.env.action_space.sample()

		else:
			q_values = self.dqn(torch.FloatTensor(state).unsqueeze(0).to(self.device)).detach().cpu().numpy()
			selected_action = np.argmax(q_values)

		return selected_action

	def choose_q_function_masked_action(self, state: np.ndarray, agent_id: int, position: np.ndarray, condition: bool):	
		""" Select an action masked to avoid collisions and so """

		# Update the state of the safety module #
		self.safe_masking_module.update_state(position = position, new_navigation_map = self.env.scenario_map)

		if self.epsilon > np.random.rand() and not self.noisy:

			# Compute randomly the action #
			if condition:
				q_values, selected_action = self.safe_masking_module.mask_action(q_values =np.random.rand(self.action_dim[0]))
			else:
				q_values, selected_action = self.safe_masking_module.mask_action(q_values =np.random.rand(self.action_dim[1]))
				"""# if there is trash in the position of the agent, the agent will pick it up
				if self.env.gt.map[int(position[0]), int(position[1])] > 0:
					selected_action = -1
					q_values[selected_action] = np.inf"""
		else:
			q_values = self.dqn(torch.FloatTensor(state).unsqueeze(0).to(self.device)).detach().cpu().numpy()
			if condition:
				q_values = q_values.squeeze(0)[:self.action_dim[0]]
			else:
				q_values = q_values.squeeze(0)[self.action_dim[0]:]
    
			q_values, selected_action = self.safe_masking_module.mask_action(q_values = q_values.flatten())
		"""# do not clean when there is no trash
		if not condition and False:
			if not self.env.gt.map[int(position[0]), int(position[1])] > 0:
				selected_action = -1
				q_values[selected_action] = -np.inf 
			else:
				selected_action = -1
				q_values[selected_action] = np.inf """
                                                        
		if self.consensus:
			self.q_values4consensus[agent_id] = q_values
   
		return selected_action

	def predict_masked_action(self, state: np.ndarray, agent_id: int, position: np.ndarray):
		""" Select an action masked to avoid collisions and so """

		# Update the state of the safety module #
		self.safe_masking_module.update_state(position = position, new_navigation_map = self.env.scenario_map)

		if self.epsilon > np.random.rand() and not self.noisy:

			# Compute randomly the action #
			q_values, selected_action = self.safe_masking_module.mask_action(q_values = None)

		else:
			q_values = self.dqn(torch.FloatTensor(state).unsqueeze(0).to(self.device)).detach().cpu().numpy()
			q_values, selected_action = self.safe_masking_module.mask_action(q_values = q_values.flatten())
		if self.consensus:
			self.q_values4consensus[agent_id] = q_values
  
		return selected_action



	def select_action(self, states: dict) -> dict:

		if self.use_nu:
			condition = self.nu > np.random.rand()
			actions = {agent_id: self.choose_q_function(state, condition=condition) for agent_id, state in states.items()}
		else:
			actions = {agent_id: self.predict_action(state) for agent_id, state in states.items()}

		return actions

	def select_masked_action(self, states: dict, positions: np.ndarray):
    
		if self.consensus:
			self.q_values4consensus = {agent_id: np.zeros((1, self.env.action_space.n)) for agent_id, state in states.items()}
   
		if self.use_nu:
			# See which phase we are by choosing self.nu > np.random.rand()
			condition = self.nu > np.random.rand()
			actions = {agent_id: self.choose_q_function_masked_action(state, agent_id=agent_id, position=positions[agent_id], condition=condition) for agent_id, state in states.items()}
		else:
			actions = {agent_id: self.predict_masked_action(state=state, agent_id=agent_id, position=positions[agent_id]) for agent_id, state in states.items()}
   
		if self.consensus:
			actions = self.consensus_safe_action_masking.query_actions(self.q_values4consensus,positions)
			actions = {agent_id: actions[agent_id] for agent_id, state in states.items()}

		return actions

	def step(self, action: dict) -> Tuple[np.ndarray, np.float64, bool]:
		"""Take an action and return the response of the env."""

		next_state, reward, done, _ = self.env.step(action)

		return next_state, reward, done

	def update_model(self) -> torch.Tensor:
		# Update the model by gradient descent. #

		# PER needs beta to calculate weights
		samples = self.memory.sample_batch(self.beta)
		weights = torch.FloatTensor(samples["weights"].reshape(-1, 1)).to(self.device)
		indices = samples["indices"]

		# Compute gradients and apply them
		self.optimizer.zero_grad()
		# PER: importance sampling before average
		if self.use_nu:
			elementwise_loss= self._compute_ddqn_multihead_loss(samples)
			if self.weighting_method is not None:
				losses = torch.cat([torch.mean(elementwise_loss[i] * weights).reshape(1) for i in range(len(elementwise_loss))])
				loss, extra_outputs = self.weighting_method.backward(
                losses=losses,
                shared_parameters=list(self.dqn.shared_parameters()),
                task_specific_parameters=list(self.dqn.task_specific_parameters()),
                last_shared_parameters=None,
                representation=None
            )
				if self.weighting_method_name=='cagrad' or self.weighting_method_name=='pcgrad':
					loss = sum([torch.mean(elementwise_loss[i] * weights).reshape(1) for i in range(len(elementwise_loss))])
		
			else:
				loss = sum([torch.mean(elementwise_loss[i] * weights[self.action_mask[i]]).reshape(1) for i in range(len(elementwise_loss))])
				loss.backward()
			self.writer.add_scalar('pruebas/loss_exp', torch.mean(elementwise_loss[0] * weights[self.action_mask[0]]).detach().cpu(), self.episode)
			self.writer.add_scalar('pruebas/loss_clean', torch.mean(elementwise_loss[1] * weights[self.action_mask[1]]).detach().cpu(), self.episode)
			if self.weighting_method is not None:
				elementwise_loss = extra_outputs['weights'][0]*elementwise_loss[0] + extra_outputs['weights'][1]*elementwise_loss[1]
			else:
				elementwise_loss = elementwise_loss[0] + elementwise_loss[1]
				
		else:
			elementwise_loss = self._compute_dqn_loss(samples)
			loss = torch.mean(elementwise_loss * weights)
			loss.backward()


		self.optimizer.step()

		# PER: update priorities
		"""weights_ = [1,1]
		for i in range(len(elementwise_loss)):
			loss_for_prior = elementwise_loss[i].detach().cpu().numpy()*weights_[i]
			new_priorities = loss_for_prior + self.prior_eps
			self.memory.update_priorities(np.asarray(indices)[self.action_mask[i]].tolist(), new_priorities)"""
   
		loss_for_prior = elementwise_loss.detach().cpu().numpy()
		new_priorities = loss_for_prior + self.prior_eps
		self.memory.update_priorities(indices, new_priorities)
		
		# Sample new noisy distribution
		if self.noisy:
			self.dqn.reset_noise()
			self.dqn_target.reset_noise()

		return loss.item()

	@staticmethod
	def anneal_nu(p, p1=[0., 1], p2=[0.5, 1.], p3=[0.5, 0.], p4=[1., 0.]):

		if p <= p2[0] and p1[0]!=p2[0]:
			first_p = p1
			second_p = p2
		elif p <= p3[0] and p2[0]!=p3[0]:
			first_p = p2
			second_p = p3
		elif p <= p4[0]:
			first_p = p3
			second_p = p4

		return (second_p[1] - first_p[1]) / (second_p[0] - first_p[0]) * (p - first_p[0]) + first_p[1]

	@staticmethod
	def anneal_epsilon(p, p_init=0.1, p_fin=0.9, e_init=1.0, e_fin=0.0):

		if p < p_init:
			return e_init
		elif p > p_fin:
			return e_fin
		else:
			return (e_fin - e_init) / (p_fin - p_init) * (p - p_init) + 1.0

	@staticmethod
	def anneal_beta(p, p_init=0.1, p_fin=0.9, b_init=0.4, b_end=1.0):

		if p < p_init:
			return b_init
		elif p > p_fin:
			return b_end
		else:
			return (b_end - b_init) / (p_fin - p_init) * (p - p_init) + b_init

	def train(self, episodes):
		# TODO: Change for multiagent

		""" Train the agent. """

		# Optimization steps #
		steps = 0
		# Create train logger #
		if self.writer is None:
			assert not os.path.exists(self.logdir), "El directorio ya existe. He evitado que se sobrescriba"
			self.writer = SummaryWriter(log_dir=self.logdir, filename_suffix=self.experiment_name)
			self.write_experiment_config()
			self.env.save_environment_configuration(self.logdir if self.logdir is not None else './')
		
		# Agent in training mode #
		self.is_eval = False
		# Reset episode count #
		self.episode = 1
		# Reset metrics #
		episodic_reward_vector = []
		record = np.array([-np.inf, -np.inf])
		mean_clean_record = -np.inf
		max_movements = self.env.distance_budget
		for episode in trange(1, int(episodes) + 1):

			done = {i:False for i in range(self.env.number_of_agents)}
			state = self.env.reset()
			score = 0
			length = 0
			losses = []
			#self.env.max_collisions = np.max([int(20-(episode/episodes)*20*1.5), 5])
			#self.writer.add_scalar('pruebas/max_collisions', self.env.max_collisions, self.episode)
			# Initially sample noisy policy #
			if self.noisy:
				self.dqn.reset_noise()
				self.dqn_target.reset_noise()

			# PER: Increase beta temperature
			self.beta = self.anneal_beta(p=episode / episodes, p_init=0, p_fin=0.9, b_init=self.beta_init, b_end=1.0)

			# Epsilon greedy annealing
			self.epsilon = self.anneal_epsilon(p=episode / episodes,
			                                   p_init=self.epsilon_interval[0],
			                                   p_fin=self.epsilon_interval[1],
			                                   e_init=self.epsilon_values[0],
			                                   e_fin=self.epsilon_values[1])
			# Run an episode #
			#print('Episode: ', episode, 'Memory used: ', self.memory.ptr)

			while not all(done.values()):
				if self.use_nu:
					distance = np.min([np.max(self.env.fleet.get_distances()), max_movements])
					self.nu = self.anneal_nu(p= distance / max_movements,
											 p1=self.nu_intervals[0],
											 p2=self.nu_intervals[1],
											 p3=self.nu_intervals[2],
											 p4=self.nu_intervals[3])
				# Increase the played steps #
				steps += 1
    
				# Select the action using the current policy
				state_float32 = {i:None for i in state.keys()}
				if self.save_state_in_uint8:
					for agent_id in state.keys():
						state_float32[agent_id] = (state[agent_id] / 255.0).astype(np.float32)
				else:
					state_float32 = state
     
				if not self.masked_actions:
					actions = self.select_action(state_float32)
				else:
					actions = self.select_masked_action(states=state_float32, positions=self.env.fleet.get_positions())


				actions = {agent_id: action for agent_id, action in actions.items() if not done[agent_id]}
				# Process the agent step #
				next_state, reward, done = self.step(actions)

				for key, lista in reward.items():
					if -1 in lista and False:
						print(f"En la lista '{key}' se encontró el valor -1.")
						print('Actions: ',actions)
						print('Positions: ', self.env.fleet.get_positions())
						print('Distances: ', self.env.fleet.get_distances())
				for agent_id in actions.keys():
					"""agent_id = np.random.randint(0, self.env.number_of_agents) ##########################################
					while agent_id not in actions.keys():
						agent_id = np.random.randint(0, self.env.number_of_agents)"""
					# Store every observation for every agent
					self.transition = [ agent_id,
                        				state[agent_id],
										actions[agent_id],
										reward[agent_id],
										next_state[agent_id],
										done[agent_id],
										{'nu': self.nu}]

					self.memory.store(*self.transition)
				# Update the state
				state = next_state
				# Accumulate indicators
				score += np.mean(list(reward.values()),axis=0)  # The mean reward among the agents
				length += 1

				# if episode ends
				if all(done.values()):

					# Append loss metric #
					if losses:
						self.episodic_loss = np.mean(losses)

					# Compute average metrics #
					self.episodic_reward = score
					self.episodic_length = length
					episodic_reward_vector.append(self.episodic_reward)
					self.episode += 1

					self.writer.add_scalar('train/trash_cleaned', self.env.percentage_of_trash_cleaned, self.episode)
					# Log progress
					self.log_data()

				# If training is ready
				if len(self.memory) >= self.batch_size and episode >= self.learning_starts:

					# Update model parameters by backprop-bootstrapping #
					if steps % self.train_every == 0:
						
						loss = self.update_model()
						# Append loss #
						losses.append(loss)

					# Update target soft/hard #
					if self.soft_update:
						self._target_soft_update()
					elif episode % self.target_update == 0 and all(done.values()):
						self._target_hard_update()

			if self.save_every is not None and False:
				if episode % self.save_every == 0:
					self.save_model(name=f'Episode_{episode}_Policy.pth')

			if self.eval_every is not None:
				if episode % self.eval_every == 0:
					mean_reward_exploration, mean_reward_cleaning, mean_reward, mean_length, total_n_trash_cleaned = self.evaluate_agents(self.eval_episodes)
					self.writer.add_scalar('test/accumulated_reward_exploration', mean_reward_exploration, self.episode)
					self.writer.add_scalar('test/accumulated_reward_cleaning', mean_reward_cleaning, self.episode)
					self.writer.add_scalar('test/accumulated_reward', mean_reward, self.episode)
					self.writer.add_scalar('test/accumulated_length', mean_length, self.episode)
					self.writer.add_scalar('test/trash_cleaned', total_n_trash_cleaned, self.episode)
					# Save policy if is better on average
					mean_episodic_reward = np.mean(episodic_reward_vector[-50:], axis=0)
					mean_reward = np.mean(mean_episodic_reward)
     
					if mean_reward_exploration > record[0]:
						print(f"New best policy with mean exploration reward of {mean_reward_exploration}")
						print("Saving model in " + self.writer.log_dir)
						record[0] = mean_reward_exploration
						self.save_model(name='BestPolicy_reward_exploration.pth')

					if mean_reward_cleaning > record[1]:
						print(f"New best policy with mean cleaning reward of {mean_reward_cleaning}")
						print("Saving model in " + self.writer.log_dir)
						record[1] = mean_reward_cleaning
						self.save_model(name='BestPolicy_reward_cleaning.pth')
					if total_n_trash_cleaned > mean_clean_record:
						print(f"New best policy with mean trash cleaned of {total_n_trash_cleaned} \%")
						print("Saving model in " + self.writer.log_dir)
						mean_clean_record = total_n_trash_cleaned
						self.save_model(name='BestCleaningPolicy.pth')
     
			

		# Save the final policy #
		self.save_model(name='Final_Policy.pth')

	def _compute_ddqn_multihead_loss(self, samples: Dict[str, np.ndarray]) -> torch.Tensor:

		"""Return dqn loss."""
		device = self.device  # for shortening the following lines
		if self.save_state_in_uint8:
			samples_obs = (samples["obs"] / 255.0).astype(np.float32)
			samples_next_obs = (samples["next_obs"] / 255.0).astype(np.float32)
			state = torch.FloatTensor(samples_obs).to(device)
			next_state = torch.FloatTensor(samples_next_obs).to(device)
		else:
			state = torch.FloatTensor(samples["obs"]).to(device)
			next_state = torch.FloatTensor(samples["next_obs"]).to(device)
		#action = torch.LongTensor(samples["acts"]).to(device)
		#reward = torch.FloatTensor(samples["rews"]).to(device)
		done = torch.FloatTensor(samples["done"].reshape(-1, 1)).to(device)
		nu_ = np.fromiter((d['nu'] for d in samples["info"]), dtype=int)
		nu_ = torch.FloatTensor(nu_.reshape(-1, 1)).to(device)

		# G_t   = r + gamma * v(s_{t+1})  if state != Terminal
		#       = r                       otherwise
		num_of_rewards = samples["rews"].shape[1]
		with torch.no_grad():
			next_maxq_values = self.dqn(next_state)
			#next_maxq_values = next_maxq_values.view((-1, num_of_rewards, self.action_dim))
			#next_max_action_values = next_maxq_values.max(dim=2, keepdim=True)[1]
			next_maxq_values_1 = next_maxq_values[:, :self.action_dim[0]]
			next_maxq_values_2 = next_maxq_values[:, self.action_dim[0]:]
			next_max_action_values_1 = next_maxq_values_1.max(dim=1, keepdim=True)[1]
			next_max_action_values_2 = next_maxq_values_2.max(dim=1, keepdim=True)[1]
			next_max_action_values = torch.cat([next_max_action_values_1, next_max_action_values_2], dim=1).view(128,2,-1)
		elementwise_loss = [0, 0]
		self.action_mask = [None, None]
		samples_aux = copy(samples)

		if not self.distributional:
			for i in range(num_of_rewards):
				samples_aux["rews"] = samples["rews"][:, i]
				offset = i*self.action_dim[0]
				# remove the actions that are not in the action space of the agent
				self.action_mask[i] = samples["acts"] < self.action_dim[i]
				samples_aux["acts"] = samples["acts"][self.action_mask[i]] + offset

				action = torch.LongTensor(samples_aux["acts"]).to(device)
				reward = torch.FloatTensor(samples_aux["rews"][self.action_mask[i]].reshape(-1, 1)).to(device)

				action = action.reshape(-1, 1)
				curr_q_value = self.dqn(state[self.action_mask[i]]).gather(1, action)
				next_max_action_value = next_max_action_values[:, i][self.action_mask[i]] + offset
				done_mask = 1 - done[self.action_mask[i]]

				with torch.no_grad():
					next_q_value = self.dqn_target(next_state[self.action_mask[i]]).gather(1, next_max_action_value)
					target = (reward + self.gamma**self.n_steps * next_q_value * done_mask).to(self.device)

				# calculate element-wise dqn loss
				#
				
				if self.weighted and i == 1:
					elementwise_loss[i]= nu_*F.mse_loss(curr_q_value, target, reduction="none")
				else:
					elementwise_loss[i] =  F.mse_loss(curr_q_value, target, reduction="none")
		
		return elementwise_loss


	def _compute_dqn_loss(self, samples: Dict[str, np.ndarray]) -> torch.Tensor:

		"""Return dqn loss."""
		device = self.device  # for shortening the following lines
		state = torch.FloatTensor(samples["obs"]).to(device)
		next_state = torch.FloatTensor(samples["next_obs"]).to(device)
		action = torch.LongTensor(samples["acts"]).to(device)
		reward = torch.FloatTensor(samples["rews"].reshape(-1, 1)).to(device)
		done = torch.FloatTensor(samples["done"].reshape(-1, 1)).to(device)

		# G_t   = r + gamma * v(s_{t+1})  if state != Terminal
		#       = r                       otherwise

		if not self.distributional:

			action = action.reshape(-1, 1)
			curr_q_value = self.dqn(state).gather(1, action)
			done_mask = 1 - done

			with torch.no_grad():
				next_q_value = self.dqn_target(next_state).max(dim=1, keepdim=True)[0]
				target = (reward + self.gamma**self.n_steps * next_q_value * done_mask).to(self.device)

			# calculate element-wise dqn loss
			elementwise_loss = F.mse_loss(curr_q_value, target, reduction="none")

		else:
			# Distributional Q-Learning - Here is where the fun begins #
			delta_z = float(self.v_interval[1] - self.v_interval[0]) / (self.num_atoms - 1)

			with torch.no_grad():

				# max_a = argmax_a' Q'(s',a')
				next_action = self.dqn_target(next_state).argmax(1)
				# V'(s', max_a)
				next_dist = self.dqn_target.dist(next_state)
				next_dist = next_dist[range(self.batch_size), next_action]

				# Compute the target distribution by adding the
				t_z = reward + (1 - done) * self.gamma**self.n_steps * self.support
				t_z = t_z.clamp(min=self.v_interval[0], max=self.v_interval[1])
				b = (t_z - self.v_interval[0]) / delta_z
				lower_bound = b.floor().long()
				upper_bound = b.ceil().long()

				offset = (torch.linspace(0, (self.batch_size - 1) * self.num_atoms, self.batch_size
					).long()
					.unsqueeze(1)
					.expand(self.batch_size, self.num_atoms)
					.to(self.device)
				)

				proj_dist = torch.zeros(next_dist.size(), device=self.device)
				proj_dist.view(-1).index_add_(
					0, (lower_bound + offset).view(-1), (next_dist * (upper_bound.float() - b)).view(-1)
				)
				proj_dist.view(-1).index_add_(
					0, (upper_bound + offset).view(-1), (next_dist * (b - lower_bound.float())).view(-1)
				)

			dist = self.dqn.dist(state)
			log_p = torch.log(dist[range(self.batch_size), action])

			elementwise_loss = -(proj_dist * log_p).sum(1)

		return elementwise_loss

	def _target_hard_update(self):
		"""Hard update: target <- local."""
		print(f"Hard update performed at episode {self.episode}!")
		self.dqn_target.load_state_dict(self.dqn.state_dict())

	def _target_soft_update(self):
		"""Soft update: target_{t+1} <- local * tau + target_{t} * (1-tau)."""
		for target_param, local_param in zip(self.dqn_target.parameters(), self.dqn_target.parameters()):
			target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)

	def log_data(self):

		if self.episodic_loss:
			self.writer.add_scalar('train/loss', self.episodic_loss, self.episode)

		self.writer.add_scalar('train/epsilon', self.epsilon, self.episode)
		self.writer.add_scalar('train/beta', self.beta, self.episode)

		percentage_visited = np.count_nonzero(self.env.fleet.historic_visited_mask) / np.count_nonzero(self.env.scenario_map)
		self.writer.add_scalar('train/percentage_visited', percentage_visited, self.episode)
		if self.use_nu:
			self.writer.add_scalar('train/nu', self.nu, self.episode)

		self.writer.add_scalar('train/accumulated_reward_exploration', self.episodic_reward[0], self.episode)
		self.writer.add_scalar('train/accumulated_reward_cleaning', self.episodic_reward[1], self.episode)
		self.writer.add_scalar('train/accumulated_length', self.episodic_length, self.episode)

		self.writer.flush()

	def load_model(self, path_to_file):

		self.dqn.load_state_dict(torch.load(path_to_file, map_location=self.device))

	def save_model(self, name='experiment.pth'):
		torch.save(self.dqn.state_dict(), self.writer.log_dir + '/' + name)

	def evaluate_agents(self, eval_episodes, render=False):
		""" Evaluate the agent on the environment for a given number of episodes with a deterministic policy """

		self.dqn.eval()
		total_reward = 0
		total_reward_information = 0
		total_reward_exploration = 0
		total_length = 0
		total_n_trash_cleaned = 0
		max_movements = self.env.distance_budget
		max_coll_ant=self.env.max_collisions
		self.env.max_collisions=np.inf
		epsilon=self.epsilon
		self.epsilon = 0
		for _ in trange(eval_episodes):

			# Reset the environment #
			state = self.env.reset()
			if render:
				self.env.render()
			done = {agent_id: False for agent_id in range(self.env.number_of_agents)}

			while not all(done.values()):

				total_length += 1
				if self.use_nu:
					distance = np.min([np.max(self.env.fleet.get_distances()), max_movements])
					self.nu = self.anneal_nu(p= distance / max_movements,
											 p1=self.nu_intervals[0],
											 p2=self.nu_intervals[1],
											 p3=self.nu_intervals[2],
											 p4=self.nu_intervals[3])
				# Select the action using the current policy
				state_float32 = {i:None for i in state.keys()}
				if self.save_state_in_uint8:
					for agent_id in state.keys():
						state_float32[agent_id] = (state[agent_id] / 255.0).astype(np.float32)
				else:
					state_float32 = state
				if not self.masked_actions:
					actions = self.select_action(state_float32)
				else:
					actions = self.select_masked_action(states=state_float32, positions=self.env.fleet.get_positions())


				actions = {agent_id: action for agent_id, action in actions.items() if not done[agent_id]}

				# Process the agent step #
				next_state, reward, done = self.step(actions)

				if render:
					self.env.render()

				# Update the state #
				state = next_state
				rewards = np.asarray(list(reward.values()))
				total_reward_information += np.sum(rewards[:,0])
				total_reward_exploration += np.sum(rewards[:,1])
			total_n_trash_cleaned += self.env.percentage_of_trash_cleaned
		total_reward += total_reward_exploration + total_reward_information
		self.dqn.train()
		self.env.max_collisions = max_coll_ant
		self.epsilon = epsilon
		# Return the average reward, average length

		return total_reward_information / eval_episodes, total_reward_exploration / eval_episodes, total_reward / eval_episodes, total_length / eval_episodes, total_n_trash_cleaned/eval_episodes

	def write_experiment_config(self):
		""" Write experiment and environment variables in a json file """

		self.experiment_config = {
			"save_every": self.save_every,
			"eval_every": self.eval_every,
			"eval_episodes": self.eval_episodes,
			"batch_size": self.batch_size,
			"gamma": self.gamma,
			"tau": self.tau,
			"lr": self.learning_rate,
			"epsilon": self.epsilon,
			"epsilon_values": self.epsilon_values,
			"epsilon_interval": self.epsilon_interval,
			"beta": self.beta,
			"num_atoms": self.num_atoms,
			"use_nu": self.use_nu,
			"nu": self.nu,
			"nu_intervals": self.nu_intervals,
			"concatenatedDQN": self.concatenatedDQN,
            "nettype": self.nettype,
            "archtype": self.archtype,
            "masked_actions": self.masked_actions,
            "consensus": self.consensus
		}

		with open(self.writer.log_dir + '/experiment_config.json', 'w') as f:
			json.dump(self.experiment_config, f, indent=4)

