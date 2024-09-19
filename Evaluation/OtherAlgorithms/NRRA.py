import numpy as np
import pandas as pd 
from tqdm import trange
import matplotlib.pyplot as plt

from Evaluation.Utils.metrics_wrapper import MetricsDataCreator
from Evaluation.Utils.anneal_nu import anneal_nu
import os

class WanderingAgent:

	def __init__(self, world: np.ndarray, movement_length: float, number_of_actions: int, consecutive_movements = None, seed = 0):
		
		self.world = world
		self.move_length = movement_length
		self.number_of_actions = number_of_actions
		self.consecutive_movements = consecutive_movements
		self.t = 0
		self.action = None
		self.seed = seed
		self.rng = np.random.default_rng(seed=self.seed)
	
	def move(self, current_position, other_positions):

		if self.action is None:
			self.action = self.select_action_without_collision(current_position, other_positions)
		
		# Compute if there is an obstacle or reached the border #
		OBS = self.check_collision(self.action, current_position, other_positions)

		if OBS:
			self.action = self.select_action_without_collision(current_position, other_positions)

		if self.consecutive_movements is not None:
			if self.t == self.consecutive_movements:
				self.action = self.select_action_without_collision(current_position, other_positions)
				self.t = 0

		self.t += 1
		return self.action , current_position + self.action_to_vector(self.action)
	
	
	def action_to_vector(self, action):
		""" Transform an action to a vector """
		angle_set = np.linspace(0, 2 * np.pi, self.number_of_actions, endpoint=False)
		angle = angle_set[action]
		movement = np.round(np.array([self.move_length * np.cos(angle), self.move_length * np.sin(angle)])).astype(int)
		return movement.astype(int)
	
	def opposite_action(self, action):
		""" Compute the opposite action """
		return (action + self.number_of_actions//2) % self.number_of_actions
	
	def check_collision(self, action, current_position, other_positions):
		""" Check if the agent collides with an obstacle """
		new_position = current_position + self.action_to_vector(action)
		new_position = np.ceil(new_position).astype(int)
		
		OBS = (new_position[0] < 0) or (new_position[0] >= self.world.shape[0]) or (new_position[1] < 0) or (new_position[1] >= self.world.shape[1])
		if not OBS:
			OBS = self.isnot_reachable(current_position, new_position) #or (list(new_position) in other_positions)
		if (list(new_position) in other_positions):
			OBS = True
		return OBS

	def select_action_without_collision(self, current_position, other_positions):
		""" Select an action without collision """
		action_caused_collision = [self.check_collision(action, current_position, other_positions) for action in range(self.number_of_actions)]

		# Select a random action without collision and that is not the oppositve previous action #
		if self.action is not None:
			opposite_action = self.opposite_action(self.action)
			action_caused_collision[opposite_action] = True
		if np.all(action_caused_collision):
			opposite_action = self.opposite_action(self.action)
			action_caused_collision[opposite_action] = False
		action = self.rng.choice(np.where(np.logical_not(action_caused_collision))[0])

		return action
	
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
    
	
	distance_budget = env.distance_budget
	random_wandering_agents = [WanderingAgent(world = env.scenario_map, number_of_actions = 8, movement_length = env.movement_length, seed=0) for _ in range(n_agents)]
	total_length = 0
	for run in trange(runs):

		#Increment the step counter #
		step = 0
		# Reset the environment #
		env.reset()

		if render:
			env.render()
		# Reset dones #
		done = {agent_id: False for agent_id in range(env.number_of_agents)}
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
				action, new_position = random_wandering_agents[i].move(env.fleet.vehicles[i].position,other_positions)
				acts.append(action)
				if list(new_position) in other_positions:
					OBS = False
				other_positions.append(list(new_position))
			actions = {i: acts[i] for i in range(n_agents)}
			#actions = {i: random_wandering_agents[i].move(env.fleet.vehicles[i].position) for i in range(n_agents)}

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
