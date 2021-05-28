from malib.aienv.env import BaseEnv


class BaseWrapper(BaseEnv):
    def __init__(self, env):
        self.env = env
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

    def step(self, action):
        return self.env.step(action)

    def reset(self):
        return self.env.reset()

    def render(self):
        return self.env.render()


class ObservationWrapper(BaseWrapper):
    def reset(self):
        obs = self.env.reset()
        return self.observation(obs)

    def step(self, action):
        obs, r, done, info = self.env.step(action)
        return self.observation(obs), r, done, info

    def observation(self, obs):
        return obs


class RewardWrapper(BaseWrapper):
    def reset(self):
        obs = self.env.reset()
        return self.observation(obs)

    def step(self, action):
        obs, r, done, info = self.env.step(action)
        return obs, self.reward(r), done, info

    def reward(self, obs):
        return obs


class ActionWrapper(BaseWrapper):
    def step(self, action):
        action = self.action(action)
        o, r, d, i = self.env.step(action)
        return o, r, d, i

    def action(self, action):
        return action
