from stable_baselines3.common.env_util import make_atari_env
from src.dqn.mega_dqn import MegaDQN
from src.mega_atari.env import MegaAtariEnv
import argparse
import cv2
import numpy as np
import torch as th


argparser = argparse.ArgumentParser()
argparser.add_argument("--envs", type=str, default="PongNoFrameskip-v4,BreakoutNoFrameskip-v4,SpaceInvadersNoFrameskip-v4,QbertNoFrameskip-v4")
argparser.add_argument("--total_timesteps", type=int, default=1000)
argparser.add_argument("--verbose", type=int, default=0)
argparser.add_argument("--seed", type=int, default=0)
args = argparser.parse_args()

# seed everything
np.random.seed(args.seed)
th.manual_seed(args.seed)

# create envs
env_strs = args.envs.split(",")
env = MegaAtariEnv(env_strs, render_mode="rgb_array")

# training
model = MegaDQN("MegaCnnPolicy", env, verbose=args.verbose, learning_starts=500)
before = model.policy.q_net.heads[0][0].weight.data.clone()
model.learn(total_timesteps=args.total_timesteps, progress_bar=True)
after = model.policy.q_net.heads[0][0].weight.data.clone()

# create renderer
img = env.render()
width, height = img.shape[1], img.shape[0]
out = cv2.VideoWriter('TrainingResult.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (width, height))

obs = env.reset()
for timestep in range(1000):
    actions, _states = model.predict(obs, deterministic=False)
    obs, rewards, done, info = env.step(actions)
    img = env.render()
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    out.write(img)
out.release()