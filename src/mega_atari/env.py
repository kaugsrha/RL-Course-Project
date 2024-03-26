import gymnasium

# Create a class which initializes a given list of atari environments and allows us to reset by choosing a new environment and a new episode
class MegaAtariEnv(gymnasium.Env):
    def __init__(self, env_name_list, current_env_num=0):
        self.env_name_list = env_name_list
        self.current_env = env_name_list[current_env_num]
        self.current_env_num = current_env_num
        self.env = super().__init__(self.current_env)
        self.env.get_info()["env_name"] = self.current_env
        self.observation_space_dict = {self.current_env: self.env.observation_space}
        self.action_space_dict = {self.current_env: self.env.action_space}

    def mega_env_reset(self, change_env=True, new_env_num=None):
        if change_env:
            if new_env_num is None:
                self.current_env_num = (self.env_num + 1) % len(self.env_name_list)
            else:
                self.current_env_num = new_env_num
        self.current_env = self.env_name_list[self.current_env_num]
        old_env = self.env.reset()
        if change_env:
            self.env = super().__init__(self.current_env)
            self.observation_space_dict[self.current_env] = self.env.observation_space
            self.action_space_dict[self.current_env] = self.env.action_space
        return old_env

    def step(self, action):
        return self.env.step(action)

    def render(self):
        self.env.render()

    def close(self):
        self.env.close()

    def get_action_space(self):
        return self.env.action_space

    def get_observation_space(self):
        return self.env.observation_space

    def get_env_name(self):
        return self.env_name

    def get_env_num(self):
        return self.env_num

    def get_env_name_list(self):
        return self.env_name_list