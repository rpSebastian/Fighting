import rlcard

from malib.aienv.env import BaseEnv


class LeducHoldem(BaseEnv):
    def __init__(self):
        self.env = rlcard.make("mahjong", config={"seed": 0})
        self.action_space = None
        self.observation_space = 36

    def step(self, actionn):
        """return s,r,d,i"""
        state, player_id = self.env.step(actionn)
        r = self.env.get_payoffs()  # list of players
        d = self.env.is_over()
        i = {}
        i["next_player"] = player_id
        return state, r, d, i

    def reset(self):
        return self.env.reset()
