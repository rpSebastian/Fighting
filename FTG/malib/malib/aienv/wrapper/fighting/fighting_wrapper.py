from malib.aienv.wrapper import ActionWrapper, BaseWrapper, ObservationWrapper


class ObsWrapper(ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.game_roles = ["p0"]

    def observation(self, obs):
        # return 'hello example observation wrapper'
        obss = {}
        for role_name in self.game_roles:
            obss[role_name] = obs
        return obss


class AWrapper(ActionWrapper):
    def action(self, action):
        return action["p0"]


class FrameLimitWrapper(BaseWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.frame_num = 0
        self.max_frame = 3000

    def reset(self):
        self.frame_num = 0
        return self.env.reset()

    def step(self, action):
        s, r, d, i = self.env.step(action)
        # i["episode_reward"] = self.frame_num
        # if d:
        #     win_result = {}
        #     if self.frame_num > 195:
        #         win_result["p0"] = 1
        #     elif self.frame_num > 80:
        #         win_result["p0"] = 0
        #     else:
        #         win_result["p0"] = -1

        #     i["win_result"] = win_result

        # i["other"] = None
        # if self.frame_num >= self.max_frame:
        #     d = True
        # self.frame_num += 1
        return s, r, d, i
