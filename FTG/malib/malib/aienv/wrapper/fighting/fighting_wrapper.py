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
        self.episode_reward = 0
        self.step_num = 0
        self.last_i = None

    def reset(self):
        self.episode_reward = 0
        self.step_num = 0
        self.last_i = None
        return self.env.reset()

    def step(self, action):
        s, r, d, i = self.env.step(action)
        if i is None:
            i = self.last_i
        # Java 异常终止， d=True， i=None
        self.last_i = i
        self.step_num += 1
        # if self.step_num % 10 == 0:
            # d = True
        own_hp = i[0]
        opp_hp = i[1]
        self.episode_reward += r
        info = {}
        info["own_hp"] = own_hp
        info["opp_hp"] = opp_hp
        info["episode_reward"] = self.episode_reward
        info["hp_diff"] = own_hp - opp_hp
        if d:
            win_result = {}
            if own_hp > opp_hp:
                win_result["p0"] = 1
            else:
                win_result["p0"] = 0
            info["win_result"] = win_result

        info["other"] = None
        return s, r, d, info
