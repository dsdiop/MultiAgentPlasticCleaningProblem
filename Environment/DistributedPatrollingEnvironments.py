import gym
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix
from sklearn.neighbors import KNeighborsRegressor
import copy

class DistributedVehicle:

	def __init__(self, agent_id, default_config: dict):
		# Unpack config values #
		self.navigation_map = default_config["navigation_map"]
		self.detection_radius = default_config["radius"]
		self.forget_factor = default_config["forget_factor"]
		self.initial_position = default_config["initial_position"]
		self.movement_length = default_config["movement_length"]
		self.agent_id = agent_id

		# Create the matrices #
		self.position = self.initial_position.copy()
		self.fleet_position_observation = {}
		self.information_matrix = np.zeros_like(self.navigation_map)
		self.idleness_matrix = np.ones_like(self.navigation_map)
		self.precision_matrix = np.zeros_like(self.navigation_map)
		self.detection_mask = self.compute_detection_mask()
		self.fleet_positional_map = np.zeros_like(self.navigation_map)
		self.redundancy_matrix = np.zeros_like(self.navigation_map)
		self.visitable_positions = np.column_stack(np.where(self.navigation_map == 1.0))
		self.change_in_information = 0.0

		# Vahicle values #
		self.distance = 0
		self.number_of_collisions = 0
		self.state = None
		self.waypoints = None
		self.d = None

	def reset(self,
			  initial_position: np.ndarray,
			  new_ground_truth_field: np.ndarray):
		""" Reset the state of the matrixes and the vehicle.
         Update the state taking in account the other vehicles model """

		# Reset the ground_truth #
		self.ground_truth_field = new_ground_truth_field
		# Reset the position
		self.position = initial_position
		self.waypoints = np.atleast_2d(self.position)
		# Reset the matrixes
		self.information_matrix = np.zeros_like(self.navigation_map)
		self.idleness_matrix = np.ones_like(self.navigation_map)
		self.precision_matrix = np.zeros_like(self.navigation_map)
		self.detection_mask = self.compute_detection_mask()
		self.fleet_positional_map = np.zeros_like(self.navigation_map)
		self.redundancy_matrix = self.detection_mask.copy()

		# Reset the fleet position dict #
		self.fleet_position_observation = {self.agent_id: self.position}

		self.distance = 0
		self.number_of_collisions = 0

		# Update the model
		self.update_model()


	def compute_detection_mask(self, position = None):

		known_mask = np.zeros_like(self.navigation_map)

		if position is None:
			px, py = self.position.astype(int)
		else:
			px, py = position.astype(int)

		# State - coverage area #
		x = np.arange(0, self.navigation_map.shape[0])
		y = np.arange(0, self.navigation_map.shape[1])

		# Compute the circular mask (area) of the state 3 #
		mask = (x[np.newaxis, :] - px) ** 2 + (y[:, np.newaxis] - py) ** 2 <= self.detection_radius ** 2

		known_mask[mask.T] = 1.0

		# known_mask[px-self.detection_radius : px+self.detection_radius+1, py-self.detection_radius : py+self.detection_radius+1] = 1.0

		return known_mask

	def update_model(self):
		""" Update the model using only the information from the self vehicle """

		# Update the detection mask #
		self.detection_mask = self.compute_detection_mask()

		# Update the precision map [P() OR Mask] and the redundancy matrix #
		self.precision_matrix = np.logical_or(self.detection_mask, self.precision_matrix) * self.navigation_map
		self.redundancy_matrix = self.detection_mask.copy()

		# Update the self information map #
		self.information_matrix = self.ground_truth_field * self.precision_matrix

		# Update own position in the fleet position dict #
		self.fleet_position_observation[self.agent_id] = self.position

	def update_idleness(self):
		# Update the idleness matrix
		self.idleness_matrix += self.forget_factor
		self.idleness_matrix = self.idleness_matrix - self.detection_mask
		self.idleness_matrix = np.clip(self.idleness_matrix, 0.0, 1.0) * self.navigation_map

	def update_detection_mask(self):
		""" Update the detection mask - analogous as taking a sample """
		self.detection_mask = self.compute_detection_mask()

	def fuse_model_information(self, external_information: dict):

		# Update the precision matrix #
		self.redundancy_matrix = np.sum(
			[external_information[i]["detection_mask"] for i in external_information.keys()], axis=0)
		self.precision_matrix = np.logical_or.reduce(
			[external_information[i]["precision_matrix"] for i in external_information.keys()])

		# Update the information matrix by agreement #
		sum_information_matrix = np.max(
			[external_information[i]["information_matrix"] for i in external_information.keys()], axis=0)

		# Register the information change for reward # 
		self.change_in_information = np.sum(np.abs(self.information_matrix - sum_information_matrix))

		# Update the change in information #
		self.information_matrix = sum_information_matrix

		# Update the idleness matrix by minimum value #
		self.idleness_matrix = np.min([external_information[i]["idleness_matrix"] for i in external_information.keys()], axis=0)

		# Update the fleet position #
		for agent_id in external_information.keys():
			self.fleet_position_observation[agent_id] = external_information[agent_id]["position"]

	def move(self, action):
		""" Move a vehicle in the direction of the action. If valid is False, the action is not performed. """

		# Compute the next attempted position #
		angle = 2 * np.pi / 8.0 * action
		movement = np.array([self.movement_length * np.cos(angle), self.movement_length * np.sin(angle)])
		next_position = self.position + movement

		if self.check_collision(next_position):
			# With a collision we increase the count #
			collide = True
			self.number_of_collisions += 1
		else:
			# Without any collisions we can update the position #
			collide = False
			self.distance += np.linalg.norm(self.position - next_position)
			self.position = next_position
			self.waypoints = np.vstack((self.waypoints, [self.position]))

		return collide

	def check_collision(self, next_position):

		outbounds_condition = next_position[0] > self.navigation_map.shape[0] or \
							  next_position[0] < 0 or \
							  next_position[1] > self.navigation_map.shape[1] or \
							  next_position[1] < 0

		if outbounds_condition:
			return True  # There is a collision
		elif self.navigation_map[int(next_position[0]), int(next_position[1])] == 0:
			return True
		else:
			return False

	def get_valid_mask(self):

		# Compute the next attempted position #
		mask = []
		for action in range(8):
			angle = 2 * np.pi / 8.0 * action
			movement = np.array([self.movement_length * np.cos(angle), self.movement_length * np.sin(angle)])
			next_position = self.position + movement
			if not self.check_collision(next_position=next_position):
				mask.append(True)
			else:
				mask.append(False)

		return np.array(mask)

	def render(self):

		if self.d is None:
			self.d = []
			self.fig, self.axs = plt.subplots(1, 5)
			for i, ax in enumerate(self.axs):
				self.d.append(ax.imshow(self.state[i]))
		else:
			for i, d in enumerate(self.d):
				d.set_data(self.state[i])

		self.fig.canvas.draw()
		plt.draw()
		plt.pause(0.1)


class DistributedFleet:

	def __init__(self, default_config):

		agent_config = default_config["vehicle_config"]
		agent_config["navigation_map"] = default_config["navigation_map"]

		self.number_of_agents = default_config["number_of_agents"]
		self.initial_positions = default_config["initial_positions"]
		self.random_initial_positions = default_config["random_initial_positions"]
		self.navigation_map = default_config["navigation_map"]
		self.ground_truth = None
		self.agents = [DistributedVehicle(agent_id=i, default_config=agent_config) for i in range(self.number_of_agents)]
		self.valid_positions = np.column_stack(np.where(self.navigation_map == 1))
		self.max_connection_distance = default_config["max_connection_distance"]
		self.connectivity_enabled = default_config["connectivity_enabled"]

		self.fig = None

	def reset(self):

		# Reset every agent #
		if self.random_initial_positions:
			new_positions = self.valid_positions[
				np.random.choice(np.arange(0, len(self.valid_positions)), self.number_of_agents, replace=False)]
		else:
			new_positions = self.initial_positions.copy()

		for i, agent in enumerate(self.agents):
			agent.reset(new_positions[i], self.ground_truth)

		# Forward the information between agents #
		for agent in self.agents:
			# Update the idleness #
			agent.update_model()
			agent.update_idleness()

		self.update_distributed_models()

	def step(self, actions: dict):
		""" Move every agent and update their models """

		collisions = []
		relative_interest_sum = []

		for agent_id, action in actions.items():
			# Process the movement #
			collisions.append(self.agents[agent_id].move(action))
			# Update the detection mask #  
			self.agents[agent_id].update_detection_mask()

		# Update the model if possible #
		if self.connectivity_enabled:
			self.update_distributed_models()

		# Compute the sum of relative interest collected #
		for agent_id in actions.keys():
			relative_interest_matrix = self.ground_truth * self.agents[agent_id].idleness_matrix * self.agents[agent_id].detection_mask
			relative_interest_values = relative_interest_matrix / np.clip(self.agents[agent_id].redundancy_matrix, 1.0, 9999.0)
			relative_interest_sum.append(relative_interest_values.sum() / (self.agents[agent_id].detection_radius**2))
		
		for agent in self.agents:
			agent.update_idleness()
			agent.update_model()

		self.update_distributed_models()
			
		return np.asarray(relative_interest_sum), np.asarray(collisions)

	def get_connectivity_matrix(self):
		""" Obtain the adjacency matrix of the fleet and compute the complete connectivity matrix """

		# First, obtain the adjacency matrix #
		positions = np.asarray([agent.position for agent in self.agents])
		available_agents_ids = np.asarray([agent.agent_id for agent in self.agents])
		adjacency_matrix = distance_matrix(positions, positions) < self.max_connection_distance

		# Obtain the connectivity matrix by computing the A^N power of the matrix
		connectivity_matrix = np.linalg.matrix_power(adjacency_matrix, len(available_agents_ids))

		return connectivity_matrix, available_agents_ids

	def get_positions(self):

		return np.array([agent.position for agent in self.agents])

	def update_distributed_models(self):
		""" Update the distributed models of every agent using the information available in the network """

		# Obtain the connectivity matrix for the fleet #
		connectivity_matrix, agents_ids = self.get_connectivity_matrix()

		# Fuse every agent model with each other information #
		for agent_id, connectivity_row in zip(agents_ids, connectivity_matrix):
			new_information_dict = {agent_id: {"information_matrix": self.agents[agent_id].information_matrix,
											   "detection_mask": self.agents[agent_id].detection_mask,
											   "precision_matrix": self.agents[agent_id].precision_matrix,
											   "redundancy_matrix": self.agents[agent_id].redundancy_matrix,
											   "idleness_matrix": self.agents[agent_id].idleness_matrix,
											   "position": self.agents[agent_id].position}
									for agent_id in agents_ids if connectivity_row[agent_id]}

			self.agents[agent_id].fuse_model_information(new_information_dict)

	def get_safe_actions(self):
		""" Obtain a dictionary with random safe actions to follow """

		safe_actions = {}
		for agent_id, agent in enumerate(self.agents):
			valid_mask = agent.get_valid_mask().astype(int)
			safe_actions[agent_id] = np.random.choice(np.arange(8), p=valid_mask/np.sum(valid_mask))

		return safe_actions


class DistributedDiscretePatrollingEnv(gym.Env):

	default_config_dict = {

		"fleet_configuration": {

			"vehicle_config": {

				"radius": 2,
				"forget_factor": 0.02,
				"initial_position": np.array([10, 20]),
				"movement_length": 3,
			},

			"navigation_map": None,
			"random_initial_positions": False,
			"initial_positions": np.array([[30, 20], [32, 20], [34, 20], [30, 22]]),
			"number_of_agents": 4,
			"max_connection_distance": 5000,
			"connectivity_enabled": True,
		},

		"ground_truth_generator": None,
		"max_collisions": 10,
		"collision_penalization": -1.0,
		"reward_new_information": None,
		"distance_budget": 100,

	}

	def __init__(self, config_dict: dict):

		self.fleet = DistributedFleet(config_dict["fleet_configuration"])
		self.navigation_map = config_dict["fleet_configuration"]["navigation_map"]
		self.distance_budget = config_dict["distance_budget"]
		self.number_of_collisions = 0

		self.gt = config_dict["ground_truth_generator"]
		
		self.max_collisions = config_dict["max_collisions"]
		self.collision_penalization = config_dict["collision_penalization"]
		self.reward_new_information = config_dict["reward_new_information"]
		self.state = None
		self.fig = None

		self.action_space = gym.spaces.Discrete(8)
		self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(5, *self.navigation_map.shape), dtype=np.float32)
		self.number_of_agents = self.navigation_map = config_dict["fleet_configuration"]["number_of_agents"]
		self.individual_action_state = gym.spaces.Discrete(8)


	def reset(self):

		# Reset the ground truth
		self.gt.reset()
		self.fleet.ground_truth = self.gt.read()

		# Reset the fleet state #
		self.fleet.reset()
		# Reset termination conditions #
		self.number_of_collisions = 0
		# Generate new state
		self.state = self.process_state()

		return self.state

	def reward_function(self, relative_interest_sum, collisions, new_information_quantity = None):
		""" Compute the reward function """

		rewards = relative_interest_sum.copy()

		if new_information_quantity is not None:
			rewards += new_information_quantity * self.reward_for_new_information

		rewards[np.where(collisions)] = self.collision_penalization

		return {i: rewards[i] for i in range(len(rewards))}

	def step(self, actions: dict):
		""" Process one step of the environment """

		# Move the fleet #
		relative_interest_sum, collisions = self.fleet.step(actions)

		# Compute reward #
		if self.reward_new_information is None:
			rewards = self.reward_function(relative_interest_sum, collisions)
		else:			
			# Add the shared information gain to the reward if specified 
			rewards = self.reward_function(relative_interest_sum, collisions, [agent.change_in_information for agent in self.fleet.agents])

		# Accumulate collisions
		self.number_of_collisions += np.sum(collisions)

		# Process state
		self.state = self.process_state()

		# Compute the final condition #
		end_condition = self.number_of_collisions > self.max_collisions or any([agent.distance > self.distance_budget for agent in self.fleet.agents])
		dones = {agent.agent_id: end_condition for agent in self.fleet.agents}

		return self.state, rewards, dones, {"collisions": self.number_of_collisions}


	def process_state(self):
		""" Create a batch of observations of the agents """
		return {agent.agent_id: self.process_individual_obs(agent) for agent in self.fleet.agents}

	def process_individual_obs(self, agent: DistributedVehicle):
		""" Gather the matrices of the agent to conform a state """

		position_map = np.zeros_like(agent.navigation_map)
		self_obs_position = agent.fleet_position_observation[agent.agent_id].astype(int)
		position_map[self_obs_position[0], self_obs_position[1]] = 1.0

		others_position_map = np.zeros_like(agent.navigation_map)
		others_positions = np.asarray([agent.fleet_position_observation[agent_id] for agent_id in agent.fleet_position_observation.keys() if agent_id != agent.agent_id]).astype(int)
		if len(others_positions) > 0:
			others_position_map[others_positions[:, 0], others_positions[:, 1]] = 1.0

		return np.concatenate((agent.navigation_map[np.newaxis],
							   position_map[np.newaxis],
							   agent.redundancy_matrix[np.newaxis]/self.number_of_agents,
							   agent.information_matrix[np.newaxis],
							   agent.idleness_matrix[np.newaxis]), axis=0)

	def render(self, mode="human"):

		if self.fig is None:
			self.d = []
			self.fig, self.axs = plt.subplots(self.number_of_agents, 5)

			self.axs = np.atleast_2d(self.axs)

			for i, agent in enumerate(self.fleet.agents):

				state = self.process_individual_obs(agent)
				self.d.append(self.axs[i, 0].imshow(state[0], cmap='gray', vmin=0, vmax=1))
				self.d.append(self.axs[i, 1].imshow(state[1], cmap='gray', vmin=0, vmax=1))
				self.d.append(self.axs[i, 2].imshow(state[2], cmap='gray', vmin=0, vmax=1))
				self.d.append(self.axs[i, 3].imshow(state[3], cmap='jet', vmin=0, vmax=1))
				self.d.append(self.axs[i, 4].imshow(state[4], cmap='plasma', vmin=0, vmax=1))

		else:
			
			for i, agent in enumerate(self.fleet.agents):

				state = self.process_individual_obs(agent)
				self.d[i * 5 + i].set_data(state[0])
				self.d[i * 5 + 1].set_data(state[1])
				self.d[i * 5 + 2].set_data(state[2])
				self.d[i * 5 + 3].set_data(state[3])
				self.d[i * 5 + 4].set_data(state[4])

		self.fig.canvas.draw()
		plt.draw()
		plt.pause(0.01)


if __name__ == '__main__':
	
	from GroundTruthsModels.ShekelGroundTruth import GroundTruth
	import time


	N = 2
	nav_map = np.genfromtxt('Environment/Maps/example_map.csv', delimiter=',')
	gt = GroundTruth(nav_map, max_number_of_peaks=6)


	env_config = {

		"fleet_configuration": {

			"vehicle_config": {
				
				"radius": 3,
				"forget_factor": 0.02,
				"initial_position": np.array([10, 20]),
				"movement_length": 2,
			},

			"navigation_map": nav_map,
			"random_initial_positions": False,
			"initial_positions": np.asarray([[30, 24],[25,24]]),
			"number_of_agents": N,
			"max_connection_distance": 5000,
			"connectivity_enabled": True,
		},

		"ground_truth_generator": gt,
		"max_collisions": 10000,
		"collision_penalization": -1.0,
		"reward_new_information": None,
		"distance_budget": 1500,

	}

	env = DistributedDiscretePatrollingEnv(env_config)
	env.reset()
	env.render()

	dones = {i: False for i in range(N)}

	time0 = time.time()
	times = []
	
	actions = {i: 0 for i in range(N)}  # Get safe actions

	while not all(dones.values()):
		
		actions = {i:action for i, action in actions.items() if not dones[i]}
		_, reward, dones, _ = env.step(actions)

		for agent_id, action in actions.items():
			if not env.fleet.agents[agent_id].get_valid_mask()[action]:
				actions[agent_id] = env.fleet.get_safe_actions()[agent_id]


		print("Reward:", reward)

		times.append(time.time() - time0)
		time0 = time.time()
		env.render()


	print(np.mean(times))



