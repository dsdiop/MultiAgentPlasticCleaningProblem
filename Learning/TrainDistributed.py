import sys

sys.path.append('.')

from Environment.DistributedPatrollingEnvironments import DistributedDiscretePatrollingEnv
import numpy as np
from Algorithm.RainbowDQL.Agent.DuelingDQNAgent import MultiAgentDuelingDQNAgent
from Environment.GroundTruthsModels.ShekelGroundTruth import GroundTruth


N = 4
sc_map = np.genfromtxt('Environment/example_map.csv', delimiter=',')
gt = GroundTruth(sc_map, max_number_of_peaks=6)


env_config = 	env_config = {

		"fleet_configuration": {

			"vehicle_config": {
				
				"radius": 2,
				"forget_factor": 0.02,
				"initial_position": np.array([10, 20]),
				"movement_length": 2,
			},

			"navigation_map": sc_map,
			"random_initial_positions": True,
			"initial_positions": np.zeros((1, 2)),
			"number_of_agents": 4,
			"max_connection_distance": 5,
			"connectivity_enabled": True,
		},

		"ground_truth_generator": gt,
		"max_collisions": 10,
		"collision_penalization": -1.0,
		"reward_new_information": None,
		"distance_budget": 150,

	}


env = DistributedDiscretePatrollingEnv(env_config)

multiagent = MultiAgentDuelingDQNAgent(env=env,
                                       memory_size=int(1E6),
                                       batch_size=64,
                                       target_update=1000,
                                       soft_update=False,
                                       tau=0.0001,
                                       epsilon_values=[1.0, 0.05],
                                       epsilon_interval=[0.0, 0.5],
                                       learning_starts=0,
                                       gamma=0.99,
                                       lr=1e-4,
                                       noisy=False,
                                       train_every=5,
                                       save_every=5000,
                                       distributional=True,
                                       num_atoms=51,
                                       v_interval=(-1, 900),
                                       logdir="./runs/Distributional",
                                       log_name="Distributional"
                                       )

multiagent.train(episodes=100000)
