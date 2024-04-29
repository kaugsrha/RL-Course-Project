# RL-Course-Project

This repo implements our course project for Spring, 2024, Theory and Practice of
Reinforcement Learning. This project tests a multi-task version of DQN
where a single feature extractor is shared between multiple Atari environments. 
The goal was to investigate if there would be positive or negative transfer by taking this approach.
Positive transfer would imply there are shared visual features between environments.
Each environment has a separate matrix C mapping the features to value functions.

To get started, install dependencies with

    pip install numpy torch stable_baselines3[all] gymnasium[atari,accept-rom-license] tqdm rich tensorboard matplotlib opencv-python-headless

Then, you may run the experiments with

    ./run_experiment.sh # linux
 or
 
    ./run_experiment.bat # windows

To plot, first modify the directory locations in plot.py and then run 

    python plot.py

Images will be written to the working directory. Alternatively, you may view the results via

    tensorboard --logdir logs

Source code is available in /src/. It is an extensively modified version
of DQN from stable baselines 3. It runs all environments in parallel in a vectorized
gymnasium environment. 