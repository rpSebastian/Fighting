import sys

from absl import flags
from pysc2 import maps
from pysc2.env import sc2_env

from malib.aienv.env import BaseEnv

FLAGS = flags.FLAGS
FLAGS(sys.argv)


class SC2_ENV(BaseEnv):
    def __init__(self, env_params: dict) -> None:
        super(SC2_ENV).__init__()

        # print('sc2 env config:',env_params)
        self.players = []
        home_player_config = env_params.pop("home_player_config")
        home_player_config["race"] = sc2_env.Race[home_player_config["race"]]
        self.players.append(sc2_env.Agent(**home_player_config))
        map_inst = maps.get(env_params["map_name"])
        if map_inst.players == 1:
            if "away_players_config" in env_params:
                print(
                    "WARRING!! there is no away player in this map, which should be remove from config dict"
                )
                env_params.pop("away_players_config")
        else:
            away_players_config = env_params.pop("away_players_config")
            for away_type, away_config in away_players_config.items():
                away_config["race"] = sc2_env.Race[away_config["race"]]
                if away_type == "bot":
                    self.players.append(sc2_env.Bot(**away_config))
                elif away_type == "agent":
                    self.players.append(sc2_env.Agent(**away_config))
                else:
                    raise Exception(
                        "away player must be bot or agent,can not be {}".format(
                            away_type
                        )
                    )
        env_params["players"] = self.players
        agent_interface_format = sc2_env.parse_agent_interface_format(
            **env_params.pop("agent_interface_format")
        )
        env_params["agent_interface_format"] = agent_interface_format
        self.env = sc2_env.SC2Env(**env_params)

        # self.observation_space=self.env.observation_space
        # self.action_space=self.env.action_space

        # self.env=sc2_env(**env_params)

    def step(self, action):
        o, r, d, i = self.env.step(action)
        return o, r, d, i

    def reset(self):
        return self.env.reset()
