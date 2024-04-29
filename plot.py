from typing import List

import pandas as pd
from tensorboard.backend.event_processing import event_accumulator
import os
import matplotlib.pyplot as plt



def smooth(xs, ys, output_len=100):
    # Smooths all data sets to have output_len points
    # xs: list of x values
    # ys: list of y values
    # output_len: number of points to output
    # returns: list of x values, list of y values
    assert len(xs) == len(ys)
    if len(xs) <= output_len:
        return xs, ys
    new_xs = []
    new_ys = []
    for i in range(output_len):
        start = i * len(xs) // output_len
        end = (i + 1) * len(xs) // output_len
        new_xs.append(xs[start])
        new_ys.append(sum(ys[start:end]) / (end - start))
    return new_xs, new_ys



def parse_tensorboard(path, scalars:List[str]):
    """returns a dictionary of pandas dataframes for each requested scalar"""
    ea = event_accumulator.EventAccumulator(
        path,
        size_guidance={event_accumulator.SCALARS: 0},
    )
    _absorb_print = ea.Reload()
    # make sure the scalars are in the event accumulator tags
    assert all(
        s in ea.Tags()["scalars"] for s in scalars
    ), "some scalars were not found in the event accumulator"
    return {k: pd.DataFrame(ea.Scalars(k)) for k in scalars}

logdir = "logs"
envs = ["PongNoFrameskip-v4", "BreakoutNoFrameskip-v4", "SpaceInvadersNoFrameskip-v4", "QbertNoFrameskip-v4"]

# get all subdirs
subdirs = [f.path for f in os.scandir(logdir) if f.is_dir()]
all_returns = []
for subdir in subdirs:
    # get the first sub sub dir
    subsubdir = [f.path for f in os.scandir(subdir) if f.is_dir()][0]

    # parse the tensorboard
    returns = {}
    for env in envs:
        try:
            res = parse_tensorboard(subsubdir, [f"rollout/return_{env.replace('NoFrameskip-v4', '')}"])
            step = res[f"rollout/return_{env.replace('NoFrameskip-v4', '')}"]["step"]
            value = res[f"rollout/return_{env.replace('NoFrameskip-v4', '')}"]["value"]
            returns[env] = {"step": step, "value": value}
        except:
            continue
    all_returns.append(returns)


# plot the returns
fig, axs = plt.subplots(2, 2, figsize=(10, 10))
for i, env in enumerate(envs):
    ax = axs[i//2, i%2]
    for returns in all_returns:
        if len(returns) > 1:
            color = 'blue'
            label = 'MT-DQN'
        else:
            color = 'red'
            label = 'DQN'
        if env in returns:
            res = returns[env]
            xs = res["step"]
            ys = res['value']
            print(env, len(xs), len(ys))
            xs, ys = smooth(xs, ys)
            print(env, len(xs), len(ys))

            ax.plot(xs, ys, color=color, label=label)

    ax.set_title(env)
    ax.set_xlabel("steps")
    ax.set_ylabel("returns")
    if i == 0:
        ax.legend()
plt.tight_layout()
plt.savefig("returns.png")
plt.clf()

# now plot the transfer case
envs = {"BoxingNoFrameskip-v4": ("logs/2024_04_19_15_01_01/DQN_1", "logs/2024_04_19_17_20_34/DQN_1"),
        "AirRaidNoFrameskip-v4": ("logs/2024_04_22_09_12_22/DQN_1", "logs/2024_04_22_11_19_29/DQN_1"),
        "ChopperCommandNoFrameskip-v4": ("logs/2024_04_22_13_25_29/DQN_1", "logs/2024_04_22_15_49_46/DQN_1"),
        "TennisNoFrameskip-v4": ("logs/2024_04_22_17_59_14/DQN_1", "logs/2024_04_22_20_08_02/DQN_1")
        }
fig, axs = plt.subplots(2, 2, figsize=(10, 10))
index = 0

# parse the tensorboard
for env, (mt_dqn_dir, dqn_dir) in envs.items():
    try:
        env_name = env.replace("NoFrameskip-v4", "")
        mt_dqn_returns = parse_tensorboard(mt_dqn_dir, [f"rollout/return_{env_name}"])
        dqn_returns = parse_tensorboard(dqn_dir, [f"rollout/return_{env_name}"])

        ax = axs[index//2, index%2]
        index += 1
        # plot the returns
        for returns, label, color in [(mt_dqn_returns, "MT-DQN", "blue"), (dqn_returns, "DQN", "red")]:
            step = returns[f"rollout/return_{env_name}"]["step"]
            value = returns[f"rollout/return_{env_name}"]["value"]
            # xs, ys = smooth(step, value)
            xs, ys = step, value
            ax.plot(xs, ys, label=label, color=color)

        ax.set_title(f"{env_name}")
        ax.set_xlabel("Steps")
        ax.set_ylabel("Returns")
        if index == 1:
            ax.legend()
    except:
        continue
plt.tight_layout()
plt.savefig("transfer_returns.png")