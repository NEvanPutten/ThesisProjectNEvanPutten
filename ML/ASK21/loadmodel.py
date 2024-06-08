import gym
import numpy as np
from stable_baselines3 import PPO
from CustomScalableENV import WingOptimEnv
import os
from Ac_constants import AcConstants



# models_dir_new = "models/PPO_SR22_V2"

model_dir= 'models/PPO_ASK21_CL'
AC = AcConstants()

import time

env = WingOptimEnv()
env.reset()

time_step =488 * 2000# 998000# 890000# 782000 #684000

model_path = f"{model_dir}/{time_step}.zip"
model_path = 'models/PPO_ASK21_TL/TL_ASK21_1648000.zip'
model = PPO.load(model_path, env=env, device = 'cuda')

episodes = 500
start_time = time.time()
best_CLCD = [] # List of final objective values
best_designs = [] #Total list of final design vectors
total_reward = [] #List of final total reward value of optimization
reward = [] # #List of final reward value of optimization
penalties = [] #List of final penalties value of optimization
# action_list = [] # action list
# reward_list = []
# cumm_reward_list = [] # each reward per step
# cumm_penalty_list = [] # each penalty per step
# total_reward_list = [] # each total reward per step
# CL_CD_list  =[] # each CL_CD list
# for episode in range(episodes):
# 	done = False
# 	obs = env.reset()
# 	while True and not done:
# 		random_action = env.action_space.sample()
# 		print("action",random_action)
# 		obs, reward, done, truncated, info = env.step(random_action)
# 		print('reward',reward)
# 		print('INFO:', info)
# 		print('done', done)


# for ep in range(episodes):
#     obs, _ = env.reset()
#     done = False
#     action = [ 0.7714547 ,  1.        ,  0.12912989,  1.        , -0.09735483,
#         0.25012958, -0.42262655, -1.        , -1.        ]
#     # action, _states = model.predict(obs)
#     obs, rewards, done,_, info = env.step(action)
#     action_list.append(action)
#     print('INFO:', info)
#     print('Reward:',rewards)
#     best_CLCD.append(info.get('Best CL/CD'))
#     best_designs.append(info.get('Best Design'))
#     total_reward.append(info.get('Cummulative reward'))
#     reward.append(info.get('Total reward'))
#     penalties.append(info.get('Total penalty'))
#     action_list.append('done')

for ep in range(episodes):
    obs, _ = env.reset()
    done = False
    while not done:
        action, _states = model.predict(obs)
        obs, rewards, done,_, info = env.step(action)
        # action_list.append(action)
        print('INFO:', info)
        print('Reward:',rewards)
        # reward_list.append(rewards)
        # cumm_reward_list.append(info.get('Total reward'))
        # cumm_penalty_list.append(info.get('Total penalty'))
        # total_reward_list.append(info.get('Cummulative reward'))
        # CL_CD_list.append(info.get('Best CL/CD'))

    best_CLCD.append(info.get('Best CL/CD'))
    best_designs.append(info.get('Best Design'))
    total_reward.append(info.get('Cummulative reward'))
    reward.append(info.get('Total reward'))
    penalties.append(info.get('Total penalty'))
    # action_list.append('done')

print("##################RESULTS##################################")
print('model:', model_path)
print('Aircraft config', AC.aircraft_config)
print(f'timestep: {time_step}')
end_time = time.time()
print("--- %s seconds ---" % (time.time() - start_time))


best_objective = max(best_CLCD)
best_index = best_CLCD.index(best_objective)

# print('Action lists', action_list)
print('\n')
print('Maximum Objective reached: ', best_objective, 'Best Wing: ', best_designs[best_index] )
print('\n')
print('Average objective', sum(best_CLCD)/episodes)
print('\n')
print('Standard Deviation', np.std(best_CLCD))
print('\n')
print('Time per optim', (end_time - start_time)/episodes)
print('\n')
print('List of objectives',best_CLCD)
print('\n')

print_dv = []
for i in range(len(best_designs)):
    temp_dv = []
    for dv in range(9):
        temp_dv.append(best_designs[i][dv])
    print_dv.append(temp_dv)

print('list of design vectors:')
print(print_dv)
print('\n')
print('rewards',reward)
print('\n')
print('penalties', penalties)
print('\n')
print('total reward', total_reward)
# print(cumm_reward_list)
# print(cumm_penalty_list)
# print(total_reward_list)
# print(CL_CD_list)


#model.save(f"{models_dir}/{TIMESTEPS * 5}")
#"tensorboard --logdir=logs"