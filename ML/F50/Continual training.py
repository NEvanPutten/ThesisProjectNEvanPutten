from stable_baselines3 import PPO
import os
from CustomScalableENV import WingOptimEnv
import time



#Setup directories for training
log_CL_dir = "logs_F50_CL"
model_CL_dir = 'models/PPO_F50_CL'

if not os.path.exists(model_CL_dir):
	os.makedirs(model_CL_dir)
if not os.path.exists(log_CL_dir):
	os.makedirs(log_CL_dir)

# Init Environment
env = WingOptimEnv()
env.reset()

# Load base TL model
model_base_dir = "models/reference models"
model_base_file = '1272000'
model_base_CL_path = f"{model_base_dir}/{model_base_file}.zip"
model = PPO.load(model_base_CL_path, env=env, device='cuda', tensorboard_log=log_CL_dir)

#Retraining
TIMESTEPS = 2000
start_time = time.time()
for i in range(637, 901):
	timestamp = time.time()
	model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="PPO")
	model.save(f"{model_CL_dir}/CL_F50_{TIMESTEPS * i}")
	print("--- %s seconds ---" % (time.time() - timestamp))

	print(f'Done one {i}')
print('Completely done')
print("--- %s seconds ---" % (time.time() - start_time))