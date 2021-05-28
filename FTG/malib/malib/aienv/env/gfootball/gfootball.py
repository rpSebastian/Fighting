import cv2 as cv
import gfootball.env as gfootball_env

from malib.aienv.env import BaseEnv


class GFootBall(BaseEnv):
    def __init__(self, env_params: dict, render=False):
        super(GFootBall).__init__()
        self.env = gfootball_env.create_environment(**env_params, render=False)
        # self.env=gfootball_env.create_environment(env_name =env_name,render=False)
        self.cv_render = render
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

    def step(self, action):
        o, r, d, info = self.env.step(action)
        # if self.cv_render:
        #     screen = self.env.render(mode="rgb_array")
        #     cv.imshow("game", screen)
        #     cv.waitKey(1)
        return o, r, d, info

    def reset(self):
        return self.env.reset()

    def render(self):
        # self.cv_render = True
        # return self.env.render(mode="rgb_array")
        return self.env.render()
