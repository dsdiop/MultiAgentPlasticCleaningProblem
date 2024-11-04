import sys
import os
data_path = os.path.join(os.path.dirname(__file__), '..')
sys.path.append(data_path)
from Environment.PatrollingEnvironments import MultiAgentPatrolling
from Algorithm.RainbowDQL.Agent.DuelingDQNAgent import MultiAgentDuelingDQNAgent
import numpy as np
import torch
import optuna
from optuna.trial import TrialState
import joblib

import argparse


parser = argparse.ArgumentParser(description='Train a multiagent DQN agent to solve the multiobjective cleaning problem.')
parser.add_argument('--map', type=str, default='malaga_port', choices=['malaga_port','alamillo_lake','ypacarai_map'], help='The map to use.')
parser.add_argument('--distance_budget', type=int, default=200, help='The maximum distance of the agents.')
parser.add_argument('--n_agents', type=int, default=4, help='The number of agents to use.')
parser.add_argument('--seed', type=int, default=0, help='The seed to use.')
parser.add_argument('--miopic', type=bool, default=True, help='If True the scenario is miopic.')
parser.add_argument('--detection_length', type=int, default=2, help='The influence radius of the agents.')
parser.add_argument('--movement_length', type=int, default=1, help='The movement length of the agents.')
parser.add_argument('--reward_type', type=str, default='Distance Field', help='The reward type to train the agent.')
parser.add_argument('--convert_to_uint8', type=bool, default=False, help='If convert the state to unit8 to store it (to save memory).')
parser.add_argument('--benchmark', type=str, default='macro_plastic', choices=['shekel', 'algae_bloom','macro_plastic'], help='The benchmark to use.')

parser.add_argument('--model', type=str, default='vaeUnet', choices=['miopic', 'vaeUnet'], help='The model to use.')
parser.add_argument('--device', type=int, default=0, help='The device to use.', choices=[-1, 0, 1])
parser.add_argument('--dynamic', type=bool, default=True, help='Simulate dynamic')

# Compose a name for the experiment
args = parser.parse_args()

N = args.n_agents

sc_map = np.genfromtxt(f"{data_path}/Environment/Maps/{args.map}.csv", delimiter=',')

if args.map == 'malaga_port':
    initial_positions = np.array([[12, 7], [14, 5], [16, 3], [18, 1]])[:N, :]
elif args.map == 'alamillo_lake':
    initial_positions = np.array([[68, 26], [64, 26], [60, 26], [56, 26]])[:N, :]
elif args.map == 'ypacarai_map':
    initial_positions = np.asarray([[24, 21],[28,24],[27,19],[24,24]])


device = 'cpu' if args.device == -1 else f'cuda:{args.device}'

nettype = '0'

arch = 'v1'
policy_name = 'Experimento_clean21_Distance_Field_model_filtered_map_alamillo_nsteps5'
logdir=f'./Learning/runs/Vehicles_{N}/SecondPaper/Optuna_{args.map}_{args.benchmark}'

def objective(trial: optuna.Trial):
    """ Optuna objective. """

    # Batch size
    batch_size = trial.suggest_int("batch_size", low = 4, high = 8, step=1)
    batch_size = 2 ** batch_size
    # Discount factor
    gamma = trial.suggest_float("gamma", low = 0.9, high = 0.999, step=0.001)
    # Target update
    target_update = trial.suggest_int("target_update", low = 1000, high = 5000, step=1000)
    # tau
    tau = trial.suggest_float("tau", low = 0.001, high = 0.01, step=0.001)
    # Epsilon values
    epsilon_values_final = trial.suggest_float("epsilon_final_value", low = 0.05, high = 0.2, step=0.05)
    # Epsilon interval
    epsilon_interval_final = trial.suggest_float("epsilon_final_interval", low = 0.25, high = 0.85, step=0.05)
    # Learning starts
    learning_starts = trial.suggest_int("learning_starts", low = 50, high = 500, step=50)
    # alpha
    alpha = trial.suggest_float("alpha", low = 0.1, high = 0.9, step=0.1)
    # beta
    beta = trial.suggest_float("beta", low = 0.1, high = 0.9, step=0.1)
    # N steps return
    n_steps = trial.suggest_int("n_steps", low = 3, high = 7, step=1)
    # Learning rate
    lr = trial.suggest_float("lr", low = 1e-5, high = 1e-3, step=1e-5)
    # Train every
    train_every = trial.suggest_int("train_every", low = 5, high = 20, step=1)

    experiment_name = "DRL_OPTUNA_TRIAL_" + str(trial.number)

    env = MultiAgentPatrolling(scenario_map=sc_map,
                                    fleet_initial_positions=initial_positions,
                                    distance_budget=args.distance_budget,
                                    number_of_vehicles=N,
                                    seed=args.seed,
                                    miopic=args.miopic,
                                    dynamic=args.dynamic,
                                    detection_length=args.detection_length,
                                    movement_length=args.movement_length,
                                    max_collisions=15,
                                    reward_type=args.reward_type,
                                    convert_to_uint8=args.convert_to_uint8,
                                    ground_truth_type=args.benchmark,
                                    obstacles=False,
                                    frame_stacking=1,
                                    state_index_stacking=(1, 2, 3)
                                    )
    multiagent = MultiAgentDuelingDQNAgent(env=env,
                                        memory_size=int(1E6),
                                        batch_size=batch_size,#128
                                        target_update=target_update,
                                        soft_update=True,
                                        tau=tau,
                                        epsilon_values=[1.0, epsilon_values_final],
                                        epsilon_interval=[0.0, epsilon_interval_final],
                                        learning_starts=learning_starts, # 100
                                        gamma=gamma,
                                        alpha=alpha,
                                        beta = beta,
                                        n_steps=n_steps,
                                        lr=lr,
                                        number_of_features=1024,
                                        noisy=False,
                                        nettype=nettype,
                                        archtype=arch,
                                        device=device,
                                        weighted=False,
                                        train_every=train_every,
                                        save_every=1000,
                                        distributional=False,
                                        logdir=logdir+'/'+experiment_name,
                                        prewarmed_memory=None,
                                        use_nu=True,
                                        nu_intervals=[[0., 1], [0.3, 1], [.6, 0.], [1., 0.]],
                                        concatenatedDQN = False,
                                        eval_episodes=100,
                                        masked_actions= True,
                                        consensus = True,
                                        eval_every=np.inf,
                                        weighting_method=None,
                                        weight_methods_parameters=dict()
                                        )

    mean_metrics = 0.0
    optuna_hyperparameter_optimization ={'trial': trial,
                                         'train_step': 10,
                                         'episode_per_train_step': 1000,
                                         'eval_episodes': 50}
    mean_metrics = multiagent.train(episodes=None, optuna_hyperparameter_optimization=optuna_hyperparameter_optimization)
    # for train_step in range(10):
        
    #     multiagent.train(episodes=1000)

    #     _, _, _, _, percentage_of_trash_cleaned, percentage_of_map_visited = multiagent.evaluate_agents(50)

    #     # Report the trial
    #     mean_metrics = percentage_of_trash_cleaned + percentage_of_map_visited
    #     trial.report(mean_metrics,train_step)

    #     # Handle pruning based on the intermediate value.
    #     if trial.should_prune():
    #         raise optuna.TrialPruned()

    return mean_metrics

if __name__ == "__main__":

	# Create a directory for the study
	if not os.path.exists(logdir):
		os.makedirs(logdir)

	study = optuna.create_study(direction="maximize", pruner=optuna.pruners.SuccessiveHalvingPruner() , study_name="DQN_hyperparametrization") # SuccessiveHalvingPruner() 

	study.optimize(objective, n_trials=50, show_progress_bar=True)

	pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
	complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

	print("Study statistics: ")
	print("  Number of finished trials: ", len(study.trials))
	print("  Number of pruned trials: ", len(pruned_trials))
	print("  Number of complete trials: ", len(complete_trials))

	print("Best trial:")
	trial = study.best_trial

	print("  Value: ", trial.value)

	print("  Params: ")
	for key, value in trial.params.items():
		print("	{}: {}".format(key, value))

	joblib.dump(study, f"{logdir}/DRL_hyperparam_study.pkl")

