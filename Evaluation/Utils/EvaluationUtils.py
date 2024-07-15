import sys
import os
data_path = os.path.join(os.path.dirname(__file__), '..')
sys.path.append(data_path)

import numpy as np
import scipy.ndimage as ndimage
import scipy.ndimage.filters as filters
import matplotlib.pyplot as plt
from tqdm import trange
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C


from Evaluation.Utils.metrics_wrapper import MetricsDataCreator
from Evaluation.Utils.anneal_nu import anneal_nu

from Evaluation.OtherAlgorithms.NRRA import WanderingAgent
from Evaluation.OtherAlgorithms.LawnMower import LawnMowerAgent
from Evaluation.OtherAlgorithms.GreedyAgent import GreedyAgent

from Environment.GroundTruthsModels.AlgaeBloomGroundTruth import algae_colormap,background_colormap
def find_peaks(data:np.ndarray, neighborhood_size: int = 5, threshold: float = 0.1) -> np.ndarray:
	""" Find the peaks in a 2D image using the local maximum filter. """


	data_max = filters.maximum_filter(data, neighborhood_size)
	maxima = (data == data_max)
	data_min = filters.minimum_filter(data, neighborhood_size)
	diff = ((data_max - data_min) > threshold)
	maxima[diff == 0] = 0

	labeled, num_objects = ndimage.label(maxima)
	slices = ndimage.find_objects(labeled)
	x, y = [], []
	for dy,dx in slices:
		x_center = (dx.start + dx.stop - 1)/2
		x.append(x_center)
		y_center = (dy.start + dy.stop - 1)/2    
		y.append(y_center)

	peaks = np.array([y,x]).T.astype(int)

	return peaks, data[peaks[:,0], peaks[:,1]]


	

def prewarm_buffer(path: str, env, runs: int, n_agents: int, ground_truth_type: str, 
			nu_intervals=[[0., 1], [0.30, 1], [0.60, 0.], [1., 0.]],
			memory = None,
      		info = {}):
	
	distance_budget = env.distance_budget
	algorithms = ['RandomWandering']
	for algorithm in algorithms:
		if algorithm == 'RandomWandering':
			if 'seed' in info.keys():
				seed = info['seed']
			else:
				seed = 0
			agents = [WanderingAgent(world = env.scenario_map,
									number_of_actions = env.fleet.n_actions,
									movement_length = env.movement_length, seed=seed+i) for i in range(n_agents)]
		elif algorithm == 'LawnMower':	
			if 'seed' in info.keys():
				seed = info['seed']
			else:
				seed = 0
			if 'initial_directions' in info.keys():
				initial_directions = info['initial_directions']
			else:
				initial_directions = [None for _ in range(n_agents)]
			agents = [LawnMowerAgent(world = env.scenario_map,
									number_of_actions = env.fleet.n_actions,
									movement_length = env.movement_length,
									forward_direction = initial_directions[i],
									seed=seed) for i in range(n_agents)]
		elif algorithm == 'GreedyAgent':
			if 'seed' in info.keys():
				seed = info['seed']
			else:
				seed = 0
			agents = [GreedyAgent(world = env.scenario_map,
									number_of_actions = env.fleet.n_actions,
									movement_length = env.movement_length,
									detection_length = env.detection_length,
									seed=seed) for _ in range(n_agents)]
	
		else:
			raise NotImplementedError('The algorithm {} is not implemented'.format(algorithm))


		total_length = 0
		for run in trange(runs):
			#Increment the step counter #
			step = 0
			# Reset the environment #
			state = env.reset()
			# Reset dones #
			done = {agent_id: False for agent_id in range(env.number_of_agents)}

			while not all(done.values()):

				total_length += 1
				other_positions = []
				acts = []
				distance = np.min([np.max(env.fleet.get_distances()), distance_budget])
				nu = anneal_nu(p= distance / distance_budget)
				# Compute the actions #
				for i in range(n_agents):
					if algorithm == 'GreedyAgent':
						continue	
						action_exp, action_inf = agents[i].move(env.fleet.vehicles[i].position, other_positions, idleness_matrix, interest_map)
						if nu > np.random.rand():
							action_mat = action_exp
						else:
							action_mat = action_inf

						idleness_matrix =  action_mat[2]
						new_position = action_mat[1]
						action = action_mat[0]
					else:
						if env.gt.map[int(env.fleet.vehicles[i].position[0]), int(env.fleet.vehicles[i].position[1])] > 0:
							action = 8
							new_position = env.fleet.vehicles[i].position
						else:
							action, new_position = agents[i].move(env.fleet.vehicles[i].position,other_positions)

					acts.append(action)
					if list(new_position) in other_positions:
						OBS = False
					other_positions.append(list(new_position))
				actions = {i: acts[i] for i in range(n_agents) if not done[i]}
				#actions = {i: random_wandering_agents[i].move(env.fleet.vehicles[i].position) for i in range(n_agents)}

				# Process the agent step #
				next_state, reward, done, _ = env.step(actions)
				#env.render()
				for agent_id in actions.keys():
					"""agent_id = np.random.randint(0, self.env.number_of_agents) ##########################################
					while agent_id not in actions.keys():
						agent_id = np.random.randint(0, self.env.number_of_agents)"""
					# Store every observation for every agent
					#print(agent_id)
					transition = [  agent_id,
									state[agent_id],
									actions[agent_id],
									reward[agent_id],
									next_state[agent_id],
									done[agent_id],
									{'nu': nu}]

					memory.store(*transition)
	return memory

def run_path_planners_evaluation(path: str, env, algorithm: str, runs: int, n_agents: int, ground_truth_type: str, 
			nu_intervals=[[0., 1], [0.30, 1], [0.60, 0.], [1., 0.]],
   			render = False,
      		save = True,	
      		info = {}):
    
    
	metrics = MetricsDataCreator(metrics_names=['Policy Name',
                                                'Accumulated Reward Intensification',
                                                'Accumulated Reward Exploration',
                                                'Total Accumulated Reward',
                                                'Total Length',
                                                'nu',
                                                'Instantaneous Global Idleness Intensification',
                                                'Instantaneous Global Idleness Exploration',
                                                'Average Global Idleness Intensification',
                                                'Average Global Idleness Exploration',
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

	if algorithm == 'RandomWandering':
		if 'seed' in info.keys():
			seed = info['seed']
		else:
			seed = 0
		agents = [WanderingAgent(world = env.scenario_map,
								number_of_actions = env.fleet.n_actions,
								movement_length = env.movement_length, seed=seed+i) for i in range(n_agents)]
	elif algorithm == 'LawnMower':	
		if 'seed' in info.keys():
			seed = info['seed']
		else:
			seed = 0
		if 'initial_directions' in info.keys():
			initial_directions = info['initial_directions']
		else:
			initial_directions = [None for _ in range(n_agents)]
		agents = [LawnMowerAgent(world = env.scenario_map,
								number_of_actions = env.fleet.n_actions,
								movement_length = env.movement_length,
								forward_direction = initial_directions[i],
								seed=seed) for i in range(n_agents)]
	elif algorithm == 'GreedyAgent':
		if 'seed' in info.keys():
			seed = info['seed']
		else:
			seed = 0
		agents = [GreedyAgent(world = env.scenario_map,
								number_of_actions = env.fleet.n_actions,
								movement_length = env.movement_length,
        						detection_length = env.detection_length,
              					seed=seed) for _ in range(n_agents)]
 
	else:
		raise NotImplementedError('The algorithm {} is not implemented'.format(algorithm))


	total_length = 0
	for run in trange(runs):
		#Increment the step counter #
		step = 0
		# Reset the environment #
		st = env.reset()

		if render:
			env.render()
		# Reset dones #
		done = {agent_id: False for agent_id in range(env.number_of_agents)}
		# Update the metrics #
		total_reward = 0
		total_reward_information = 0
		total_reward_exploration = 0
		total_length = 0
		instantaneous_global_idleness = 0
		percentage_visited = np.count_nonzero(env.fleet.historic_visited_mask) / np.count_nonzero(env.scenario_map)
		average_global_idleness_int = env.average_global_idleness
		average_global_idleness_exp = env.average_global_idleness_exp
		instantaneous_global_idleness = env.instantaneous_global_idleness
		instantaneous_global_idleness_exp = env.instantaneous_global_idleness_exp
		distance = np.min([np.max(env.fleet.get_distances()), distance_budget])
		nu = anneal_nu(distance / distance_budget, *nu_intervals)
  
		metrics_list = [algorithm, total_reward_information,
                        total_reward_exploration,
                        total_reward, total_length, nu,
                        instantaneous_global_idleness,
                        instantaneous_global_idleness_exp,
                        average_global_idleness_int,
                        average_global_idleness_exp,
                        percentage_visited]
		# Initial register #
		metrics.register_step(run_num=run, step=total_length, metrics=metrics_list)
		for veh_id, veh in enumerate(env.fleet.vehicles):
			paths.register_step(run_num=run, step=total_length, metrics=[veh_id, veh.position[0], veh.position[1]])

		fig_vis = []
		dd = False
		imm = []
		while not any(done.values()):

			total_length += 1
			other_positions = []
			acts = []
   
			s = {i:None for i in st.keys()}
			if env.convert_to_uint8:
				for agent_id in st.keys():
					s[agent_id] = (st[agent_id] / 255.0).astype(np.float32)
			else:
				s = st
    
			idleness_matrix =  s[np.argmax(env.active_agents)][0]
			#interest_map = env.importance_matrix
			interest_map =s[np.argmax(env.active_agents)][1]
			distance = np.min([np.max(env.fleet.get_distances()), distance_budget])
			nu = anneal_nu(p= distance / distance_budget)
			# Compute the actions #
			for i in range(n_agents):
       
				if algorithm == 'GreedyAgent':
					action_exp, action_inf= agents[i].move(env.fleet.vehicles[i].position, other_positions, idleness_matrix, interest_map)
					if nu > np.random.rand():
						list_exp = action_exp
					else:
						list_exp = action_inf
					idleness_matrix =  list_exp[2]
					new_position = list(list_exp[1])
					action = list_exp[0]
				else:
					action, new_position = agents[i].move(env.fleet.vehicles[i].position,other_positions)
     
				acts.append(action)
				if list(new_position) in other_positions:
					OBS = False
				other_positions.append(list(new_position))
			actions = {i: acts[i] for i in range(n_agents)}
			#actions = {i: random_wandering_agents[i].move(env.fleet.vehicles[i].position) for i in range(n_agents)}

			if render:
				print(nu)
			# Process the agent step #
			st, reward, done, _ = env.step(actions)
			if nu == 1 and dd :
				
				#print('exp: ', percentage_visited_exp)
				fig1,ax = plt.subplots()
				model = env.scenario_map*np.nan
				model[np.where(env.scenario_map)] = env.model[np.where(env.scenario_map)]
				pos1 = ax.imshow(model,  cmap=algae_colormap, vmin=0.0, vmax=1.0)
				fig1.colorbar(pos1, ax=ax, orientation='vertical')
				fig_vis.append([total_length,nu,model])   
				plt.title('Contamination Model')
				plt.close()
				#plt.savefig(f'C:\\Users\\dames\\OneDrive\\Documentos\\GitHub\\MultiAgentPatrollingProblem\\Results_seed30_firstpaper1/contamination_model.png')
				#plt.show()
				fig0,ax = plt.subplots()
				pos0= ax.imshow(env.node_visit, cmap='rainbow')
				fig0.colorbar(pos0, ax=ax, orientation='vertical')
				#plt.show()
				fig_vis.append([total_length,nu,env.node_visit])
				#plt.savefig(f'C:\\Users\\dames\\OneDrive\\Documentos\\GitHub\\MultiAgentPatrollingProblem\\Results_seed30_firstpaper1/{policy_name}_node_visit_exp.png')
				plt.close()
			elif nu == 0 and dd:   
				fig1,ax = plt.subplots()
				model = env.scenario_map*np.nan
				model[np.where(env.scenario_map)] = env.model[np.where(env.scenario_map)]
				pos1 = ax.imshow(model,  cmap=algae_colormap, vmin=0.0, vmax=1.0)
				fig1.colorbar(pos1, ax=ax, orientation='vertical')
				fig_vis.append([total_length,nu,model])   
				plt.title('Contamination Model')
				plt.close()
				fig3,ax = plt.subplots()
				pos3= ax.imshow(env.node_visit, cmap='rainbow')
				fig3.colorbar(pos3, ax=ax, orientation='vertical')
				#plt.show()
				fig_vis.append([total_length,nu,env.node_visit])
				#plt.savefig(f'C:\\Users\\dames\\OneDrive\\Documentos\\GitHub\\MultiAgentPatrollingProblem\\Results_seed30_firstpaper1/{policy_name}_node_visit_exp.png')
				plt.close() 
			elif dd:
				env.node_visit=np.zeros_like(env.scenario_map)  
			if render:
				env.render()
	
			distance = np.min([np.max(env.fleet.get_distances()), distance_budget])
			nu = anneal_nu(distance / distance_budget, *nu_intervals)
			rewards = np.asarray(list(reward.values()))
			percentage_visited = np.count_nonzero(env.fleet.historic_visited_mask) / np.count_nonzero(env.scenario_map)
			total_reward_information += np.sum(rewards[:,0])
			total_reward_exploration += np.sum(rewards[:,1]) 
			total_reward = total_reward_exploration + total_reward_information

			average_global_idleness_int = env.average_global_idleness
			average_global_idleness_exp = env.average_global_idleness_exp
			instantaneous_global_idleness = env.instantaneous_global_idleness
			instantaneous_global_idleness_exp = env.instantaneous_global_idleness_exp

			metrics_list = [algorithm, total_reward_information,
							total_reward_exploration,
							total_reward, total_length, nu,
							instantaneous_global_idleness,
							instantaneous_global_idleness_exp,
							average_global_idleness_int,
							average_global_idleness_exp,
							percentage_visited]
			metrics.register_step(run_num=run, step=total_length, metrics=metrics_list)
			for veh_id, veh in enumerate(env.fleet.vehicles):
				paths.register_step(run_num=run, step=total_length, metrics=[veh_id, veh.position[0], veh.position[1]])

		# Reset the algorithm #
		if algorithm == 'LawnMower':
			for i in range(n_agents):
				agents[i].reset(initial_directions[i])
		

	imm.append({algorithm:fig_vis})
	if save:
		if not os.path.exists(path):
			os.makedirs(path)
	
		metrics.register_experiment()
		paths.register_experiment()

	if render:
		pass
		#plt.close()

def run_evaluation_Ver_old(path: str, agent, algorithm: str, reward_type: str, runs: int, n_agents: int, ground_truth_type: str, render = False):

	
	metrics = {'Algorithm': [], 
			'Reward type': [],  
			'Run': [], 
			'Step': [],
			'N_agents': [],
			'Ground Truth': [],
			'Mean distance': [],
			'Accumulated Reward': [],
			'$\Delta \mu$': [], 
			'$\Delta \sigma$': [], 
			'Total uncertainty': [],
			'Error $\mu$': [],
			'Max. Error in $\mu_{max}$': [], 
			'Mean Error in $\mu_{max}$': [],
			'SOP GLOBAL GP':[],
			'MSE GLOBAL GP':[],
			'R2 GLOBAL GP':[],}
	
	for i in range(n_agents): 
		metrics['Agent {} X '.format(i)] = []
		metrics['Agent {} Y'.format(i)] = []
		metrics['Agent {} reward'.format(i)] = []

	for run in trange(runs):

		#Increment the step counter #
		step = 0
		
		# Reset the environment #
		state = agent.env.reset()

		if render:
			agent.env.render()

		# Reset dones #
		done = {agent_id: False for agent_id in range(agent.env.number_of_agents)}

		# Reset modules
		for module in agent.nogoback_masking_modules.values():
			module.reset()

		# Update the metrics #
		metrics['Algorithm'].append(algorithm)
		metrics['Reward type'].append(reward_type)
		metrics['Run'].append(run)
		metrics['Step'].append(step)
		metrics['N_agents'].append(n_agents)
		metrics['Ground Truth'].append(ground_truth_type)
		metrics['Mean distance'].append(0)
		metrics['Total uncertainty'].append(agent.env.gp_coordinator.sigma_map[agent.env.gp_coordinator.X[:,0], agent.env.gp_coordinator.X[:,1]].mean())
		metrics['$\Delta \mu$'].append(0)
		metrics['$\Delta \sigma$'].append(0)
		metrics['Error $\mu$'].append(agent.env.get_error())
		metrics['Max. Error in $\mu_{max}$'].append(1)
		metrics['Mean Error in $\mu_{max}$'].append(1)

		peaks, vals = find_peaks(agent.env.gt.read())
		if peaks.shape[0] == 0:
			peaks, vals = find_peaks(agent.env.gt.read(), threshold=0.3)

		positions = agent.env.fleet.get_positions()
		for i in range(n_agents): 
			metrics['Agent {} X '.format(i)].append(positions[i,0])
			metrics['Agent {} Y'.format(i)].append(positions[i,1])
			metrics['Agent {} reward'.format(i)].append(0)

		metrics['Accumulated Reward'].append(0)
		
		acc_reward = 0

		while not all(done.values()):

			step += 1

			# Select the action using the current policy
			if not 	agent.masked_actions:
				actions = agent.select_action(state, deterministic=True)
			else:
				actions = agent.select_masked_action(states=state, positions=agent.env.fleet.get_positions(), deterministic=True)
				
			actions = {agent_id: action for agent_id, action in actions.items() if not done[agent_id]}

			# Process the agent step #
			next_state, reward, done, _ = agent.env.step(actions)

			if render:
				agent.env.render()

			acc_reward += sum(reward.values())

			# Update the state #
			state = next_state

			# Datos de estado
			metrics['Algorithm'].append(algorithm)
			metrics['Reward type'].append(reward_type)
			metrics['Run'].append(run)
			metrics['Step'].append(step)
			metrics['N_agents'].append(n_agents)
			metrics['Ground Truth'].append(ground_truth_type)
			metrics['Mean distance'].append(agent.env.fleet.get_distances().mean())

			# Datos de cambios en la incertidumbre y el mu
			changes_mu, changes_sigma = agent.env.gp_coordinator.get_changes()
			metrics['$\Delta \mu$'].append(changes_mu.sum())
			metrics['$\Delta \sigma$'].append(changes_sigma.sum())
			# Incertidumbre total aka entropía
			metrics['Total uncertainty'].append(agent.env.gp_coordinator.sigma_map[agent.env.gp_coordinator.X[:,0], agent.env.gp_coordinator.X[:,1]].mean())
			# Error en el mu
			metrics['Error $\mu$'].append(agent.env.get_error())
			# Error en el mu max
			peaks, vals = find_peaks(agent.env.gt.read())
			if peaks.shape[0] == 0:

				peaks = np.unravel_index(np.argmax(agent.env.gt.read()), agent.env.gt.read().shape) 
				vals = agent.env.gt.read()[peaks[0], peaks[1]]
				#peaks, vals = find_peaks(agent.env.gt.read(), threshold=0.8)
				estimated_vals = agent.env.gp_coordinator.mu_map[peaks[0], peaks[1]]
			else:
				estimated_vals = agent.env.gp_coordinator.mu_map[peaks[:,0], peaks[:,1]]
			error = np.abs(estimated_vals - vals)
			metrics['Max. Error in $\mu_{max}$'].append(np.max(error))
			metrics['Mean Error in $\mu_{max}$'].append(np.mean(error.mean()))

			positions = agent.env.fleet.get_positions()
			for i in range(n_agents): 
				metrics['Agent {} X '.format(i)].append(positions[i,0])
				metrics['Agent {} Y'.format(i)].append(positions[i,1])
				metrics['Agent {} reward'.format(i)].append(0)

			metrics['Accumulated Reward'].append(acc_reward)

		if render:
			plt.show()

		# Compute the final error using all the points in the map #
		gp_unique = GaussianProcessRegressor(kernel=C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-2, 1e3)), n_restarts_optimizer=10)
		gp_unique.fit(agent.env.gp_coordinator.x, agent.env.gp_coordinator.y)
		mu_global = gp_unique.predict(agent.env.gp_coordinator.X)
		real_values = agent.env.gt.read()[agent.env.gp_coordinator.X[:,0], agent.env.gp_coordinator.X[:,1]]
		sop_GLOBAL_GP = np.abs(mu_global - real_values).sum()
		mse_GLOBAL_GP = mean_squared_error(mu_global, real_values)
		r2_GLOBAL_GP = r2_score(mu_global, real_values)

		# Add the final error to the metrics #
		metrics['SOP GLOBAL GP'].extend([sop_GLOBAL_GP] * (step + 1))
		metrics['MSE GLOBAL GP'].extend([mse_GLOBAL_GP] * (step + 1))
		metrics['R2 GLOBAL GP'].extend([r2_GLOBAL_GP] * (step + 1))


	df = pd.DataFrame(metrics)



	df.to_csv(path + '/{}_{}_{}_{}.csv'.format(algorithm, ground_truth_type, reward_type, n_agents))

