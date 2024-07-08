import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import trange
from Evaluation.Utils.metrics_wrapper import MetricsDataCreator
from Evaluation.Utils.anneal_nu import anneal_nu
import os
class LawnMowerAgent:

	def __init__(self, world: np.ndarray, number_of_actions: int, movement_length: int, forward_direction: int, seed=0):

		""" Finite State Machine that represents a lawn mower agent. """
		self.world = world
		self.action = None
		self.number_of_actions = number_of_actions
		self.move_length = movement_length
		self.state = 'FORWARD'
		self.turn_count = 0
		self.initial_action = forward_direction
		self.seed = seed
		self.rng = np.random.default_rng(seed=self.seed)
		if self.initial_action is None:
			self.initial_action = self.rng.integers(0, self.number_of_actions)

	
	def compute_obstacles(self, current_position, position, other_positions):
		# Compute if there is an obstacle or reached the border #
		OBS = position[0] < 0 or position[0] >= self.world.shape[0] or position[1] < 0 or position[1] >= self.world.shape[1]
		
		if not OBS:
			OBS = OBS or self.isnot_reachable(current_position, position)# or (list(position) in other_positions)
		if list(position) in other_positions:
			OBS = True
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



	def move(self, current_position,others_positions):
		""" Compute the new state """

		# Compute the new position #
		new_position = current_position + self.action_to_vector(self.state_to_action(self.state)) 
		# Compute if there is an obstacle or reached the border #
		OBS = self.compute_obstacles(current_position, new_position,others_positions)

		if self.state == 'FORWARD':
			
			if not OBS:
				self.state = 'FORWARD'
			else:

				# Check if with the new direction there is an obstacle #
				new_position = current_position + self.action_to_vector(self.state_to_action('TURN')) 
				OBS = self.compute_obstacles(current_position, new_position,others_positions)

				if not OBS:
					self.state = 'TURN'
				else:
					self.state = 'RECEED'

		elif self.state == 'RECEED':
			# Stay in receed state until there is no obstacle #

			# Check if with the new direction there is an obstacle #
			new_position = current_position + self.action_to_vector(self.state_to_action('TURN')) 
			OBS = self.compute_obstacles(current_position, new_position,others_positions)
			if OBS:
				self.state = 'RECEED'
			else:
				self.state = 'TURN'

		elif self.state == 'TURN':

			if self.turn_count == 1 or OBS:
				self.state = 'REVERSE'
				self.turn_count = 0
			else:
				self.state = 'TURN'
				self.turn_count += 1

		elif self.state == 'REVERSE':

			if not OBS:
				self.state = 'REVERSE'
			else:

				# Check if with the new direction there is an obstacle #
				new_position = current_position + self.action_to_vector(self.state_to_action('TURN2'))
				OBS = self.compute_obstacles(current_position, new_position,others_positions)

				if not OBS:
					self.state = 'TURN2'
				else:
					self.state = 'RECEED2'

		elif self.state == 'RECEED2':
			# Stay in receed state until there is no obstacle #
			new_position = current_position + self.action_to_vector(self.state_to_action('TURN2')) 
			OBS = self.compute_obstacles(current_position, new_position,others_positions)
			if OBS:
				self.state = 'RECEED2'
			else:
				self.state = 'TURN2'

		elif self.state == 'TURN2':
				
				if self.turn_count == 1 or OBS:
					self.state = 'FORWARD'
					self.turn_count = 0
				else:
					self.state = 'TURN2'
					self.turn_count += 1

		# Compute the new position #
		new_position = current_position + self.action_to_vector(self.state_to_action(self.state)) 
		# Compute if there is an obstacle or reached the border #
		OBS = self.compute_obstacles(current_position, new_position,others_positions)
		ina = self.initial_action
		while OBS:
			self.initial_action = self.perpendicular_action(self.initial_action)
			if ina == self.initial_action:
				break
			self.state = 'FORWARD'
			# Compute the new position #
			new_position = current_position + self.action_to_vector(self.state_to_action(self.state)) 
			# Compute if there is an obstacle or reached the border #
			OBS = self.compute_obstacles(current_position, new_position,others_positions)
   
		new_position = current_position + self.action_to_vector(self.state_to_action(self.state)) 
		return self.state_to_action(self.state), new_position
	
	def state_to_action(self, state):

		if state == 'FORWARD':
			return self.initial_action
		elif state == 'TURN':
			return self.perpendicular_action(self.initial_action)
		elif state == 'REVERSE':
			return self.opposite_action(self.initial_action)
		elif state == 'TURN2':
			return self.perpendicular_action(self.initial_action)
		elif state == 'RECEED':
			return self.opposite_action(self.initial_action)
		elif state == 'RECEED2':
			return self.initial_action

	def action_to_vector(self, action):
		""" Transform an action to a vector """
		angle_set = np.linspace(0, 2 * np.pi, self.number_of_actions, endpoint=False)
		angle = angle_set[action]
		movement = np.round(np.array([self.move_length * np.cos(angle), self.move_length * np.sin(angle)])).astype(int)
		return movement.astype(int)
	
	def perpendicular_action(self, action):
		""" Compute the perpendicular action """
		return (action - self.number_of_actions//4) % self.number_of_actions
	
	def opposite_action(self, action):
		""" Compute the opposite action """
		return (action + self.number_of_actions//2) % self.number_of_actions
	
	def reset(self, initial_action):
		""" Reset the state of the agent """
		self.state = 'FORWARD'
		self.initial_action = initial_action
		self.turn_count = 0
		if self.initial_action is None:
			self.initial_action = self.rng.integers(0, self.number_of_actions)


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
  
	initial_directions = np.random.choice([0,1,2,3,4], size=n_agents, replace=False)
	initial_directions = [None for _ in range(n_agents)]
	lawn_mower_agents = [LawnMowerAgent(world = env.scenario_map, number_of_actions = 8, movement_length = env.movement_length, forward_direction = initial_directions[i], seed=0) for i in range(n_agents)]

	
	distance_budget = env.distance_budget

	for run in trange(runs):
		initial_directions = np.random.choice([0,1,2,3,4], size=n_agents, replace=False)
		initial_directions = [None for _ in range(n_agents)]
		#lawn_mower_agents = [LawnMowerAgent(world = env.scenario_map, number_of_actions = 8, movement_length = env.movement_length, forward_direction = initial_directions[i], seed=0) for i in range(n_agents)]
		for i in range(n_agents):
			lawn_mower_agents[i].reset(initial_directions[i])
		# Increment the step counter #
		step = 0
		
		# Reset the environment #
		env.reset()

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

		while not all(done.values()):

			total_length += 1
			other_positions = []
			acts = []
			# Compute the actions #
			for i in range(n_agents):
				action, new_position = lawn_mower_agents[i].move(env.fleet.vehicles[i].position,other_positions)
				acts.append(action)
				if list(new_position) in other_positions:
					OBS = False
				other_positions.append(list(new_position))
			actions = {i: acts[i] for i in range(n_agents)}
			# Process the agent step #
			_, reward, done, _ = env.step(actions)

			if render:
				env.render()
			distance = np.min([np.max(env.fleet.get_distances()), distance_budget])
			nu = anneal_nu(p= distance / distance_budget)
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

		


	if not render:
		metrics.register_experiment()
		paths.register_experiment()
	else:
		plt.close()


		
		

