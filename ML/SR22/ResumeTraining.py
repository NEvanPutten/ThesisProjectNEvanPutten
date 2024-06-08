import gym
from stable_baselines3 import PPO
from CustomScalableENV import WingOptimEnv
import os
import time

models_dir = "models/PPO_SR22_V2"
logdir = "logs_SR22_base_V2"



env = WingOptimEnv()
env.reset()

model_path = f"{models_dir}/1000000.zip"
model = PPO.load(model_path, env=env,  verbose=1, tensorboard_log=logdir)

TIMESTEPS = 2000
start_time = time.time()
for i in range(501, 751):
	timestamp = time.time()
	model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="PPO")
	model.save(f"{models_dir}/{TIMESTEPS * i}")
	print("--- %s seconds ---" % (time.time() - timestamp))

	print(f'Done one {i}')
print('Completely done')
print("--- %s seconds ---" % (time.time() - start_time))

#tensorboard --logdir=logs