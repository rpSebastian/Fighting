from malib.aienv.env import BaseEnv
from malib.aienv.wrapper import BaseWrapper, ObservationWrapper, ActionWrapper
from smac.env import StarCraft2Env


class SmacEnv(BaseEnv):
    def __init__(self, env_params: dict, render=False):
        super(SmacEnv, self).__init__()
        self.env = StarCraft2Env(**env_params)
        self.env_info = self.env.get_env_info()
        self.n_actions = self.env_info["n_actions"]
        self.n_agents = self.env_info["n_agents"]
        self.episode_reward = 0

    def get_env_info(self):
        return self.env.get_env_info()

    def get_obs(self):
        return self.env.get_obs()

    def get_state(self):
        return self.env.get_state()

    def step(self, action):
        r, d, i = self.env.step(action)
        self.raw_info = dict(i)
        return r, d, i

    def reset(self):
        obs, _ = self.env.reset()
        self.episode_reward = 0
        return obs


class SmacWrapper(BaseWrapper):
    def __init__(self, env):
        super(SmacWrapper, self).__init__(env)

    def step(self, action):
        reward, done, info = self.env.step(action)
        self.env.episode_reward += reward
        obs = self.env.get_obs()

        if done:
            win_result = {}
            if "battle_won" in info:
                if info["battle_won"]:
                    win_result["p0"] = 1
                else:
                    win_result["p0"] = -1
                info["win_result"] = win_result
                info["episode_reward"] = self.env.episode_reward
            else:
                info["error"] = True
                info_len = len(info)
                print(
                    "------------- PROTOCOL ERROR ---------- info len:{}".format(info)
                )
        return obs, reward, done, info


class SmacObsWrapper(ObservationWrapper):
    def __init__(self, env):
        super(SmacObsWrapper, self).__init__(env)

    def observation(self, obs):
        result = {}
        observation = {}

        avail_actions = {}
        for i in range(self.n_agents):
            agent_name = "a" + str(i)
            observation[agent_name] = obs[i]
            avail_action = self.env.get_avail_agent_actions(i)
            avail_actions[agent_name] = avail_action
        result["obs"] = observation
        result["avail_actions"] = avail_actions

        res = {}
        res["p0"] = result

        return res

    def reset(self):
        self.env.reset()
        obs = self.env.get_obs()
        return self.observation(obs)


class SmacActionWrapper(ActionWrapper):
    def action(self, action):
        return action["p0"]
