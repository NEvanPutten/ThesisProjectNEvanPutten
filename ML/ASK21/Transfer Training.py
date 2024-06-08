from stable_baselines3 import PPO
import os
from CustomScalableENV import WingOptimEnv
import time



#Setup directories for training
log_TL_dir = "logs_ASK21_TL"
model_TL_dir = 'models/PPO_ASK21_TL'

if not os.path.exists(model_TL_dir):
	os.makedirs(model_TL_dir)
if not os.path.exists(log_TL_dir):
	os.makedirs(log_TL_dir)

# Init Environment
env = WingOptimEnv()
env.reset()

# Load base TL model
model_base_dir = "models/reference models"
model_base_file = 'base_TL_model_1272000'
model_base_TL_path = f"{model_base_dir}/{model_base_file}.zip"
model = PPO.load(model_base_TL_path, env=env, device='cuda', tensorboard_log=log_TL_dir)

#Retraining
TIMESTEPS = 2000
start_time = time.time()
for i in range(637, 901):
	timestamp = time.time()
	model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="PPO")
	model.save(f"{model_TL_dir}/TL_ASK21_{TIMESTEPS * i}")
	print("--- %s seconds ---" % (time.time() - timestamp))

	print(f'Done one {i}')
print('Completely done')
print("--- %s seconds ---" % (time.time() - start_time))