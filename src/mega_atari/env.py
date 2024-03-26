import gymnasium

# Create a class which initializes a given list of atari environments and allows us to reset by choosing a new environment and a new episode
class MegaAtariEnv(gymnasium.Env):
    def __init__(self, env_name_list, current_env_num=None, *args, **kwargs):
        self.env_name_list = env_name_list

        # crreate envs
        self.envs = {}
        for env in env_name_list:
            self.envs[env] = gymnasium.make(env, *args, **kwargs)

        # keep track of current
        self.current_index = current_env_num if current_env_num is not None else len(env_name_list) - 1

        # verify obs space the same
        self.observation_space_0 = self.envs[self.env_name_list[0]].observation_space
        for env in self.env_name_list:
            assert self.envs[env].observation_space == self.observation_space_0

        # get largest action space
        self.max_action_space = max([self.envs[env_index].action_space.n for env_index in self.env_name_list])


    def reset(self, change_env=True, new_env_num=None):
        if change_env:
            if new_env_num is None:
                self.current_index = (self.current_index + 1) % len(self.env_name_list)
            else:
                self.current_index = new_env_num
        # self.current_env = self.env_name_list[self.current_env_num]
        # old_env = self.env.reset()
        # if change_env:
        #     self.env = super().__init__(self.current_env)
        #     self.observation_space_dict[self.current_env] = self.env.observation_space
        #     self.action_space_dict[self.current_env] = self.env.action_space
        obs, info = self.envs[self.env_name_list[self.current_index]].reset()
        info["env_name"] = self.env_name_list[self.current_index]
        return obs, info

    def step(self, action):
        next_state, reward, terminated, truncated, info = self.envs[self.env_name_list[self.current_index]].step(action)
        info["env_name"] = self.env_name_list[self.current_index]
        return next_state, reward, terminated, truncated, info

    def render(self):
        # self.env.render()
        return self.envs[self.env_name_list[self.current_index]].render()
    def close(self):
        for k,v in self.envs.items():
            v.close()

    # override attribute to be a method
    @property
    def action_space(self):
        return self.envs[self.env_name_list[self.current_index]].action_space
    @property
    def observation_space(self):
        return self.envs[self.env_name_list[self.current_index]].observation_space



    def get_env_name(self):
        return self.env_name_list[self.current_index]

    def get_env_num(self):
        return self.current_index

    def get_env_name_list(self):
        return self.env_name_list

if __name__ == "__main__":


    import matplotlib.pyplot as plt


    env = MegaAtariEnv(["Pong-v4", "Breakout-v4"], render_mode="rgb_array")


    for env_index in env.env_name_list:
        env.reset()
        for timestep in range(100):
            env.step(env.action_space.sample())
            img = env.render()
            plt.imshow(img)
            plt.show(block=False)
            plt.pause(0.05)