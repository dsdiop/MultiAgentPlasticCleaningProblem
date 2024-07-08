import numpy as np
import pandas as pd 
from tqdm import trange
import matplotlib.pyplot as plt

from Evaluation.Utils.metrics_wrapper import MetricsDataCreator
from Evaluation.Utils.anneal_nu import anneal_nu
import os

class GreedyAgent:

	def __init__(self, world: np.ndarray, movement_length: float, detection_length: float, number_of_actions: int, seed = 0):
		
		self.world = world
		self.move_length = movement_length
		self.detection_length = detection_length
		self.number_of_actions = number_of_actions
		self.seed = seed
		self.rng = np.random.default_rng(seed=self.seed)
	
	def move(self, current_position, other_positions, idleness_matrix, interest_map):

		# Compute if there is an obstacle or reached the border #
		new_possible_positions = [current_position + self.action_to_vector(i) for i in range(self.number_of_actions)]
		OBS = self.check_possible_collisions(current_position, new_possible_positions, other_positions)
		if OBS.all():
			return 0, current_position, idleness_matrix

		rewards_exploration = []
		rewards_information = []
		new_idleness_matrices = []

		for i in range(len(OBS)):
			if OBS[i]:
				rewards_exploration.append(-np.inf)
				rewards_information.append(-np.inf)
				new_idleness_matrices.append(idleness_matrix)
			else:
				reward, new_idleness_matrix = self.reward_function(new_possible_positions[i], idleness_matrix, interest_map)
				rewards_exploration.append(reward[1])
				rewards_information.append(reward[0])
				new_idleness_matrices.append(new_idleness_matrix)

		max_exploration_index = np.argmax(rewards_exploration)
		max_information_index = np.argmax(rewards_information)

		action_exploration = self.rng.choice(np.where(np.array(rewards_exploration) == rewards_exploration[max_exploration_index])[0])
		action_information = self.rng.choice(np.where(np.array(rewards_information) == rewards_information[max_information_index])[0])
		#print(f'exploration: {rewards_exploration[action_exploration]}')
		#print(f'information: {rewards_information[action_information]}')
		return [action_exploration, new_possible_positions[action_exploration], new_idleness_matrices[action_exploration]], [action_information, new_possible_positions[action_information], new_idleness_matrices[action_information]]

	def reward_function(self, position, idleness_matrix, interest_map):
		""" Compute the reward function given the position, the idleness matrix and the interest map. """
		detection_mask = self.compute_detection_mask(position)
		rewards_information = np.sum((interest_map[detection_mask.astype(bool)] * idleness_matrix[detection_mask.astype(bool)])/
                               (1*self.detection_length)) 
		
		rewards_exploration = np.sum((idleness_matrix[detection_mask.astype(bool)])/
                               (1*self.detection_length)) 

		new_idleness_matrix = np.clip(idleness_matrix - detection_mask,0,1)
  
		return np.asarray([rewards_information, rewards_exploration]), new_idleness_matrix
	
	def interest_recollected(self, agent_position, interest_map):
		""" Given the agent position and the interest map, compute the interest recollected. """
		interest_map = np.ones_like(interest_map)
		masked_interest_map = interest_map * self.compute_detection_mask(agent_position) * self.world
		interest_recollected = np.sum(masked_interest_map)
		return interest_recollected

	def compute_detection_mask(self, agent_position):
		""" Compute the circular mask """
  
		px, py = agent_position.astype(int)
  		# State - coverage area #
		x = np.arange(0, self.world.shape[0])
		y = np.arange(0, self.world.shape[1])

		# Compute the circular mask (area) #
		mask = (x[np.newaxis, :] - px) ** 2 + (y[:, np.newaxis] - py) ** 2 <= self.detection_length ** 2

		known_mask = np.zeros_like(self.world)
		known_mask[mask.T] = 1.0
		known_mask = known_mask*self.world
		for px, py in np.argwhere(known_mask == 1):
			if self.isnot_reachable(agent_position, [px, py]):
				known_mask[px, py] = 0
		return known_mask*self.world

	def action_to_vector(self, action):
		""" Transform an action to a vector """
		angle_set = np.linspace(0, 2 * np.pi, self.number_of_actions, endpoint=False)
		angle = angle_set[action]
		movement = np.round(np.array([self.move_length * np.cos(angle), self.move_length * np.sin(angle)])).astype(int)
		return movement.astype(int)

	def check_possible_collisions(self, current_position, new_possible_positions, other_positions):
		""" Check if the agent collides with an obstacle """
		agent_collisions = [list(elemento) in other_positions for elemento in new_possible_positions]
		world_collisions = [(new_position[0] < 0) or (new_position[0] >= self.world.shape[0]) or (new_position[1] < 0) or (new_position[1] >= self.world.shape[1])
                      		for new_position in new_possible_positions]
		border_collisions = [self.isnot_reachable(current_position, new_position)  if not world_collisions[i] else True 
                       		for i,new_position in enumerate(new_possible_positions)]
		OBS = np.logical_or(agent_collisions, np.logical_or(world_collisions,border_collisions))	
		return OBS
	def isnot_reachable(self, current_position, next_position):
		""" Check if the next position is reachable or navigable """
		if self.world[int(next_position[0]), int(next_position[1])] == 0:
			return True 
		x, y = next_position
		dx = x - current_position[0]
		dy = y - current_position[1]
		steps = max(abs(dx), abs(dy))
		dx = dx / steps if steps != 0 else 0
		dy = dy / steps if steps != 0 else 0
		reachable_positions = True
		for step in range(1, steps + 1):
			px = round(current_position[0] + dx * step)
			py = round(current_position[1] + dy * step)
			if self.world[px, py] != 1:
				reachable_positions = False
				break

		return not reachable_positions

def run_evaluation(path: str, env, algorithm: str, runs: int, n_agents: int, ground_truth_type: str, render = False):

	metrics = MetricsDataCreator(metrics_names=['Policy Name',
											'Accumulated Reward Intensification',
											'Accumulated Reward Exploration',
											'Total Accumulated Reward',
											'Total Length',
											'Total Collisions',
											'Average global idleness Intensification',
											'Average global idleness Exploration',
											'Sum global idleness Intensification',
											'Percentage Visited Exploration',
											'Percentage Visited'],
							algorithm_name=algorithm,
							experiment_name=f'{algorithm}_Results',
							directory=path)
	if os.path.exists(path + algorithm + '_Results.csv'):
		metrics.load_df(path + algorithm + '_Results.csv')
        
	paths = MetricsDataCreator(metrics_names=['vehicle', 'x', 'y'],
                            algorithm_name=algorithm,
                            experiment_name=f'{algorithm}_paths',
                            directory=path)
    
	if os.path.exists(path + algorithm + '_paths.csv'):
		paths.load_df(path + algorithm + '_paths.csv')
  

	greedy_agents = [GreedyAgent(world = env.scenario_map, number_of_actions = 8, movement_length = env.movement_length, detection_length=env.detection_length, seed=0) for i in range(n_agents)]

	
	distance_budget = env.distance_budget

	for run in trange(runs):
		# Increment the step counter #
		step = 0
		
		# Reset the environment #
		s = env.reset()

		if render:
			env.render()

		# Reset dones #
		done = {agent_id: False for agent_id in range(env.number_of_agents)}
		#plt.savefig(f'{path}_{algorithm}_{run}.png')
		# Update the metrics #
		total_reward = 0
		total_reward_information = 0
		total_reward_exploration = 0
		total_length = 0
		total_collisions = 0
		percentage_visited = np.count_nonzero(env.fleet.historic_visited_mask) / np.count_nonzero(env.scenario_map)
		percentage_visited_exp = np.count_nonzero(env.fleet.historic_visited_mask) / np.count_nonzero(env.scenario_map)
		average_global_idleness_exp = env.average_global_idleness_exp
		sum_global_interest = env.sum_global_idleness
		sum_instantaneous_global_idleness = 0
		steps_int = 0
		average_global_idleness_int = sum_instantaneous_global_idleness
		metrics_list = [algorithm, total_reward_information,
						total_reward_exploration,
						total_reward, total_length,
						total_collisions,
						average_global_idleness_int,
						average_global_idleness_exp,
						sum_global_interest,
						percentage_visited_exp,
						percentage_visited]
		# Initial register #
		metrics.register_step(run_num=run, step=total_length, metrics=metrics_list)
		for veh_id, veh in enumerate(env.fleet.vehicles):
			paths.register_step(run_num=run, step=total_length, metrics=[veh_id, veh.position[0], veh.position[1]])
		imm = []
		while not all(done.values()):
			st = {i:None for i in s.keys()}
			if env.convert_to_uint8:
				for agent_id in s.keys():
					st[agent_id] = (s[agent_id] / 255.0).astype(np.float32)
			else:
				st = s
			total_length += 1
			other_positions = []
			acts = []
			idleness_matrix =  st[np.argmax(env.active_agents)][0]
			interest_map =st[np.argmax(env.active_agents)][1]
			distance = np.min([np.max(env.fleet.get_distances()), distance_budget])
			nu = anneal_nu(p= distance / distance_budget, p1=[0., 1], p2=[0.5, 1.], p3=[0.5, 1.], p4=[1., 1.])
			print(f'nu: {nu}')
			# Compute the actions #
			for i in range(n_agents):
				action_exp, action_inf = greedy_agents[i].move(env.fleet.vehicles[i].position, other_positions, idleness_matrix, interest_map)
				if nu > np.random.rand():
					action = action_exp
				else:
					action = action_inf


				acts.append(action[0])
				if list(action[1]) in other_positions:
					OBS = False
				other_positions.append(list(action[1]))
				idleness_matrix =  action[2]
			actions = {i: acts[i] for i in range(n_agents)}
			# Process the agent step #
			s, reward, done, _ = env.step(actions)

			if render:
				env.render()
			rewards = np.asarray(list(reward.values()))
			percentage_visited = np.count_nonzero(env.fleet.historic_visited_mask) / np.count_nonzero(env.scenario_map)
			if nu<0.5:
				steps_int += 1
				sum_instantaneous_global_idleness += env.instantaneous_global_idleness
				average_global_idleness_int = sum_instantaneous_global_idleness/steps_int
				total_reward_information += np.sum(rewards[:,0])
			else:
				average_global_idleness_exp = np.copy(env.average_global_idleness_exp)
				percentage_visited_exp = np.count_nonzero(env.fleet.historic_visited_mask) / np.count_nonzero(env.scenario_map)
				total_reward_exploration += np.sum(rewards[:,1])

			total_collisions += env.fleet.fleet_collisions    
			total_reward = total_reward_exploration + total_reward_information

			instantaneous_global_idleness = env.instantaneous_global_idleness
			imm.append(instantaneous_global_idleness)
			sum_global_interest = env.sum_global_idleness
			metrics_list = [algorithm, total_reward_information,
							total_reward_exploration,
							total_reward, total_length,
							total_collisions,
							average_global_idleness_int,
							average_global_idleness_exp,
							sum_global_interest,
							percentage_visited_exp,
							percentage_visited]
			metrics.register_step(run_num=run, step=total_length, metrics=metrics_list)
			for veh_id, veh in enumerate(env.fleet.vehicles):
				paths.register_step(run_num=run, step=total_length, metrics=[veh_id, veh.position[0], veh.position[1]])

		

	plt.figure()
	plt.plot(imm)
	plt.show()
	if not render:
		metrics.register_experiment()
		paths.register_experiment()
	else:
		plt.close()



def compute_greedy_action(agent_position, interest_map, max_distance, navigation_map):

	""" Given the agent position and the interest map, compute the greedy action. """

	px, py = agent_position.astype(int)

	# State - coverage area #
	x = np.arange(0, navigation_map.shape[0])
	y = np.arange(0, navigation_map.shape[1])

	# Compute the circular mask (area) #
	mask = (x[np.newaxis, :] - px) ** 2 + (y[:, np.newaxis] - py) ** 2 <= max_distance ** 2

	known_mask = np.zeros_like(navigation_map)
	known_mask[mask.T] = 1.0

	masked_interest_map = interest_map * known_mask * navigation_map

	# Compute the action that moves the agent in the direction of the maximum value of masked_interest_map #
	best_position_x, best_position_y = np.unravel_index(np.argmax(masked_interest_map), masked_interest_map.shape)

	direction = np.arctan2(best_position_y - py, best_position_x - px)

	direction = direction + 2*np.pi if direction < 0 else direction

	greedy_action = np.argmin(np.abs(direction - np.linspace(0, 2*np.pi, 8)))

	return greedy_action, np.asarray([best_position_y, best_position_x])


