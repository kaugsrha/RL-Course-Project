#python train.py --envs "PongNoFrameskip-v4,BreakoutNoFrameskip-v4,SpaceInvadersNoFrameskip-v4,QbertNoFrameskip-v4"
#python train.py --envs "PongNoFrameskip-v4"
#python train.py --envs "BreakoutNoFrameskip-v4"
#python train.py --envs "SpaceInvadersNoFrameskip-v4"
#python train.py --envs "QbertNoFrameskip-v4"


python train.py --head_type nn --envs "PongNoFrameskip-v4,BreakoutNoFrameskip-v4,SpaceInvadersNoFrameskip-v4,QbertNoFrameskip-v4"
python train.py --head_type nn --envs "PongNoFrameskip-v4"
python train.py --head_type nn --envs "BreakoutNoFrameskip-v4"
python train.py --head_type nn --envs "SpaceInvadersNoFrameskip-v4"
python train.py --head_type nn --envs "QbertNoFrameskip-v4"