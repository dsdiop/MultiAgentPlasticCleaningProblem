
import sys
import os
data_path = os.path.join(os.path.dirname(__file__), '..')
sys.path.append(data_path)
from Environment.PatrollingEnvironments import MultiAgentPatrolling
import numpy as np
import argparse
from tqdm import trange
from Evaluation.Utils.EvaluationUtils import run_path_planners_evaluation
from matplotlib import pyplot as plt


N = 4
sc_map = np.genfromtxt(f'{data_path}/Environment/Maps/example_map.csv', delimiter=',')
visitable_locations = np.vstack(np.where(sc_map != 0)).T
random_index = np.random.choice(np.arange(0,len(visitable_locations)), N, replace=False)
initial_positions = np.asarray([[24, 21],[28,24],[27,19],[24,24]])

N_EPISODES = 200
reward_type = 'metrics global'
ground_truth_type = 'algae_bloom'
seed = 30
env = MultiAgentPatrolling(scenario_map=sc_map,
                        fleet_initial_positions=initial_positions,
                        distance_budget=200,
                        number_of_vehicles=N,
                        seed=seed,
                        miopic=True,
                        detection_length=2,
                        movement_length=2,
                        max_collisions=np.inf,
                        forget_factor=0.5,
                        attrition=0.1,
                        reward_type='metrics global',
                        ground_truth_type=ground_truth_type,
                        obstacles=False,
                        frame_stacking=1,
                        state_index_stacking=(2, 3, 4),
                        reward_weights=(1.0, 0.1)
                        )
algorithms = ['RandomWandering', 'LawnMower', 'GreedyAgent']
for algorithm in algorithms:
    if 'LawnMower' not in algorithm:
        continue
    run_path_planners_evaluation(path=f'./Evaluation/Results/Results_seed_{seed}_Heuristics/', 
                    env=env, 
                    algorithm = algorithm,
                    runs = N_EPISODES, 
                    n_agents = N, 
                    ground_truth_type = ground_truth_type, 
                    render = False,
                    save=True,
                    info={'seed': 30 })