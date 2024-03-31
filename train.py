from stable_baselines3.common.env_util import make_atari_env
from src.dqn.mega_dqn import MegaDQN
from src.mega_atari.env import MegaAtariEnv
import argparse
import cv2
import numpy as np
import torch as th
import datetime

argparser = argparse.ArgumentParser()
argparser.add_argument("--envs", type=str, default="PongNoFrameskip-v4,BreakoutNoFrameskip-v4,SpaceInvadersNoFrameskip-v4,QbertNoFrameskip-v4")
argparser.add_argument("--total_timesteps", type=int, default=int(1e6))
argparser.add_argument("--verbose", type=int, default=0)
argparser.add_argument("--seed", type=int, default=0)
args = argparser.parse_args()

# seed everything
np.random.seed(args.seed)
th.manual_seed(args.seed)

# create envs
env_strs = args.envs.split(",")
print("Training on ", env_strs )
env = MegaAtariEnv(env_strs, render_mode="rgb_array")

# logging
date_time_str = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
logdir = f"logs/{date_time_str}"

# training
model = MegaDQN("MegaCnnPolicy", env, verbose=args.verbose, learning_starts=500, tensorboard_log=logdir)
model.learn(total_timesteps=args.total_timesteps, progress_bar=True)

# save
model.save(f"{logdir}/model")

# create renderer
img = env.render()
width, height = img.shape[1], img.shape[0]
out = cv2.VideoWriter(f'{logdir}/TrainingResult.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (width, height))

# run episodes
obs = env.reset()
for timestep in range(10_000):
    actions, _states = model.predict(obs, deterministic=False)
    obs, rewards, done, info = env.step(actions)
    img = env.render()
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    out.write(img)
out.release()