import gymnasium
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.vec_env import DummyVecEnv
import copy

# Create a class which initializes a given list of atari environments and allows us to reset by choosing a new environment and a new episode
class MegaAtariEnv(DummyVecEnv):
    def __init__(self, env_name_list, *args, **kwargs):
        self.env_name_list = env_name_list

        # create envs
        env_functions = []
        for env_str in env_name_list:
            # print("Creating env ", env_str)
            fcn = lambda env_str=env_str: AtariWrapper(gymnasium.make(env_str, *args, **kwargs))
            env_functions.append(copy.deepcopy(fcn))

        # wrap into sb3 stuff
        env = DummyVecEnv(env_functions)
        self.env = VecFrameStack(env, n_stack=4)

        # get largest action space
        self.action_space = gymnasium.spaces.Discrete(max([self.envs[i].action_space.n for i in range(len(self.env_name_list))]))
        self.all_action_spaces = gymnasium.spaces.MultiDiscrete([self.envs[i].action_space.n for i in range(len(self.env_name_list))])
        self.env_strs = env_name_list


    def step(self, actions):
        self.step_async(actions)
        return self.step_wait()

    def reset(self):
        return self.env.reset()
    
    def step_async(self, actions):
        return self.env.step_async(actions)
    
    def step_wait(self):
        return self.env.step_wait()


    def __getattr__(self, name):
            # If the attribute is not found in OuterClass, try to find it in the inner object
            if hasattr(self.env, name):
                return getattr(self.env, name)
            else:
                raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")



if __name__ == "__main__":

    import cv2

    env = MegaAtariEnv(["PongNoFrameskip-v4","BreakoutNoFrameskip-v4", "SpaceInvadersNoFrameskip-v4", "QbertNoFrameskip-v4"], render_mode="rgb_array")
    
    # create renderer
    width, height = 160 * 2, 210 * 2
    out = cv2.VideoWriter('MegaEnvTest.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (width, height))

    env.reset()
    step_number = 0
    for i in range(500):
        action = env.all_action_spaces.sample()
        o, r, d, i = env.step(action)
        print(o.shape)
        img = env.render()
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        out.write(img)
        # print(action.shape, o.shape, r.shape, d.shape, i)
    out.release()
