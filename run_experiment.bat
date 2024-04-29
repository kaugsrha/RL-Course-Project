@echo off

set environments="PongNoFrameskip-v4,BreakoutNoFrameskip-v4,SpaceInvadersNoFrameskip-v4,QbertNoFrameskip-v4"

python train.py --envs %environments%
python train.py --envs "PongNoFrameskip-v4"
python train.py --envs "BreakoutNoFrameskip-v4"
python train.py --envs "SpaceInvadersNoFrameskip-v4"
python train.py --envs "QbertNoFrameskip-v4"


python transfer.py --env "BoxingNoFrameskip-v4" --load_dir "logs/2024_04_18_15_43_42/model.zip"
python transfer.py --env "BoxingNoFrameskip-v4"

python transfer.py --env "AirRaidNoFrameskip-v4" --load_dir "logs/2024_04_18_15_43_42/model.zip"
python transfer.py --env "AirRaidNoFrameskip-v4"

python transfer.py --env "ChopperCommandNoFrameskip-v4" --load_dir "logs/2024_04_18_15_43_42/model.zip"
python transfer.py --env "ChopperCommandNoFrameskip-v4"

python transfer.py --env "TennisNoFrameskip-v4" --load_dir "logs/2024_04_18_15_43_42/model.zip"
python transfer.py --env "TennisNoFrameskip-v4"