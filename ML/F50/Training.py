from stable_baselines3 import PPO
import os
from CustomScalableENV import WingOptimEnv
import time
import torch as T


models_dir = "models/PPO_F50_base"
logdir = "logs_F50_base"

if not os.path.exists(models_dir):
	os.makedirs(models_dir)

if not os.path.exists(logdir):
	os.makedirs(logdir)

env = WingOptimEnv()
env.reset()

# model_path = f"{models_dir}/2000.zip"
# model = PPO.load(model_path, env=env)
policy_kwargs = dict(activation_fn=T.nn.ReLU,
                     net_arch=dict(pi=[64, 64, 64], vf=[64, 64, 64]))

model = PPO("MlpPolicy", env, policy_kwargs=policy_kwargs, verbose=1, tensorboard_log=logdir)


TIMESTEPS = 2000
start_time = time.time()
for i in range(1,501):
	timestamp = time.time()
	model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="PPO", progress_bar=True)
	model.save(f"{models_dir}/{TIMESTEPS * i}")
	print("--- %s seconds ---" % (time.time() - timestamp))


	print(f'Done one {i}')
print('Completely done')
print("--- %s seconds ---" % (time.time() - start_time))

