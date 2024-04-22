from stable_baselines3.common.env_util import make_atari_env

from src.dqn.SpecialProgressBar import SpecialProgressBarCallback
from src.dqn.mega_dqn import MegaDQN
from src.mega_atari.env import MegaAtariEnv
import argparse
import cv2
import numpy as np
import torch as th
import datetime
import os

argparser = argparse.ArgumentParser()
argparser.add_argument("--env", type=str, default="BoxingNoFrameskip-v4")
argparser.add_argument("--original_envs", type=str, default="PongNoFrameskip-v4,BreakoutNoFrameskip-v4,SpaceInvadersNoFrameskip-v4,QbertNoFrameskip-v4")
argparser.add_argument("--total_timesteps", type=int, default=int(1e6))
argparser.add_argument("--verbose", type=int, default=0)
argparser.add_argument("--seed", type=int, default=0)
argparser.add_argument("--head_type", type=str, default="linear")
argparser.add_argument('--load_dir', type=str, default=None)
args = argparser.parse_args()

# seed everything
np.random.seed(args.seed)
th.manual_seed(args.seed)
assert args.head_type in ["linear", "nn"]

# create envs
env_strs = [args.env]
print("Training on ", env_strs )
env = MegaAtariEnv(env_strs, render_mode="rgb_array")

# logging
date_time_str = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
logdir = f"./logs/{date_time_str}"
os.makedirs(logdir, exist_ok=True)

# training
cb = SpecialProgressBarCallback()

if args.load_dir is not None:
    print("Testing Transfer")
    dumy_env_strs = args.original_envs.split(",")
    dummy_env = MegaAtariEnv(dumy_env_strs, render_mode="rgb_array")
    model = MegaDQN("MegaCnnPolicy", dummy_env,
                    verbose=args.verbose,
                    learning_starts=500,
                    tensorboard_log=logdir,
                    policy_kwargs={"head_type": args.head_type},
                    device="auto")
    model.set_parameters(args.load_dir)
    model.set_env(env, force_reset=True)
    model.create_new_head(env.action_space)
    dummy_env.close()
    del dummy_env, dumy_env_strs
else:
    model = MegaDQN("MegaCnnPolicy", env,
                    verbose=args.verbose,
                    learning_starts=500,
                    tensorboard_log=logdir,
                    policy_kwargs={"head_type": args.head_type},
                    device="auto")
model.learn(total_timesteps=args.total_timesteps, callback=cb)

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