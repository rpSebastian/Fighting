import gym

from malib.aienv.env import BaseEnv


class CartPole_V1(BaseEnv):
    def __init__(self):
        self.env = gym.make("CartPole-v1")
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space

    def step(self, action):
        return self.env.step(action)

    def reset(self):
        return self.env.reset()

    def render(self):
        return self.env.render()
