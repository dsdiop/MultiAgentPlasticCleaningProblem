import sys
import os
data_path = os.path.join(os.path.dirname(__file__), '..')
sys.path.append(data_path)
from Environment.PatrollingEnvironments import MultiAgentPatrolling
from Algorithm.RainbowDQL.Agent.DuelingDQNAgent import MultiAgentDuelingDQNAgent
import numpy as np
import torch

N = 4
sc_map = np.genfromtxt(data_path+'/Environment/Maps/example_map.csv', delimiter=',')
visitable_locations = np.vstack(np.where(sc_map != 0)).T
random_index = np.random.choice(np.arange(0,len(visitable_locations)), N, replace=False)
initial_positions = np.asarray([[24, 21],[28,24],[27,19],[24,24]])

#initial_positions = np.array([[30, 20], [32, 20], [34, 20], [30, 22]])[:N, :]
#visitable = np.column_stack(np.where(sc_map == 1))
# initial_positions = visitable[np.random.randint(0,len(visitable), size=N), :]

#frame_stack
nettype = '0'

N=4
ww = None
wm = dict()
archs = ['v1']
i=0
rew = 'v5'
num_of_eval_episodes = 200
seed_eval = 30
#windows_datapath = os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', '..','Downloads') #"C:\\Users\\dames\\Downloads\\"
prewarm_buffer = data_path+'/prewarm_buffer/buffer.pkl'
for batch_size in [128]:
    prewarm_buffer = data_path+'/prewarm_buffer/buffer.pkl'
    for arch in archs:
        eval_dir = f'{data_path}/Evaluation/Results/'
        if not os.path.exists(eval_dir):
            os.makedirs(eval_dir)
        policy_name = 'Experimento_serv_2DQN'+'_rew_'+rew+'_bsize_'+str(batch_size)
        logdir=f'./Learning/runs/Vehicles_{N}/FirstPaper/'+policy_name
        env = MultiAgentPatrolling(scenario_map=sc_map,
                                fleet_initial_positions=initial_positions,
                                distance_budget=200,
                                number_of_vehicles=N,
                                seed=0,
                                miopic=True,
                                detection_length=2,
                                movement_length=2,
                                max_collisions=15,
                                forget_factor=0.5,
                                minimum_importance=0.1,
                                reward_type='Double reward'+rew,
                                convert_to_uint8=True,
                                ground_truth_type='algae_bloom',
                                obstacles=False,
                                frame_stacking=1,
                                state_index_stacking=(1, 2, 3)
                                )
        multiagent = MultiAgentDuelingDQNAgent(env=env,
                                            memory_size=int(1E6),
                                            batch_size=batch_size,#64
                                            target_update=1000,
                                            soft_update=True,
                                            tau=0.001,
                                            epsilon_values=[1.0, 0.05],
                                            epsilon_interval=[0.0, 0.5],
                                            learning_starts=100, # 100
                                            gamma=0.99,
                                            alpha= 0.2,
                                            beta = 0.4,
                                            n_steps=1,
                                            lr=1e-4,
                                            number_of_features=1024,
                                            noisy=False,
                                            nettype=nettype,
                                            archtype=arch,
                                            device='cuda:1',
                                            weighted=False,
                                            train_every=7,
                                            save_every=1000,
                                            distributional=False,
                                            logdir=logdir,
                                            prewarmed_memory=None,
                                            use_nu=True,
                                            nu_intervals=[[0., 1], [0.30, 1], [0.60, 0.], [1., 0.]],
                                            concatenatedDQN = True,
                                            eval_episodes=100,
                                            masked_actions= True,
                                            consensus = True,
                                            eval_every=200,
                                            weighting_method=ww,
                                            weight_methods_parameters=wm
                                            )

        multiagent.train(episodes=10000)
        torch.cuda.empty_cache()

