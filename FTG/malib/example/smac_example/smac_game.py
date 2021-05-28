import ray
import sys
import time

sys.path.insert(0, "../malib")
# sys.path.insert(0, "example")
import example
from example.smac_example.agent.scplayer import SCPlayer
from example.smac_example.env.smac_env import SmacEnv
from example.smac_example.env.smac_env import (
    SmacEnv,
    SmacWrapper,
    SmacObsWrapper,
    SmacActionWrapper,
)

from malib.utils import regist, register_init
from malib.utils import default_league_config, default_learner_config
from malib.game import Game
from malib.action import ProbSampleAction, RandomAction
from malib.feature import DictFeature

if torch.cuda.is_available():
    ray.init(address="auto")
else:
    ray.init()

register = register_init()
regist("smacenv", SmacEnv)
regist("smacwrapper", SmacWrapper)
regist("smacobswrapper", SmacObsWrapper)
regist("smacactionwrapper", SmacActionWrapper)
regist("prob_action", ProbSampleAction)
regist("random_action", RandomAction)
regist("dict_feature", DictFeature)
regist("scplayer", SCPlayer)


l_config = default_learner_config()
env_config = dict(
    config_name="env_config",
    smac_env=dict(
        raw_env="smacenv",
        wrapper=["smacwrapper", "smacobswrapper", "smacactionwrapper"],
        env_params=dict(map_name="8m"),
    ),
)
learner_config = dict(
    config_name="learner_config",
    game_number=2,
    env_name="smac_env",
    player_id="p0",
)
player_config = dict(
    config_name="player_config",
    players=["p0"],
    p0=dict(
        # player_name="agent:MAPlayer",
        player_name="scplayer",
        agents=["a0", "a1", "a2", "a3", "a4", "a5", "a6", "a7"],
        action_config="prob_action",
        feature_config="dict_feature",
    ),
    agent_config=dict(
        a0=dict(agent="agent:BaseAgent", model_id="m0"),
        a1=dict(agent="agent:BaseAgent", model_id="m0"),
        a2=dict(agent="agent:BaseAgent", model_id="m0"),
        a3=dict(agent="agent:BaseAgent", model_id="m0"),
        a4=dict(agent="agent:BaseAgent", model_id="m0"),
        a5=dict(agent="agent:BaseAgent", model_id="m0"),
        a6=dict(agent="agent:BaseAgent", model_id="m0"),
        a7=dict(agent="agent:BaseAgent", model_id="m0"),
    ),
    model_config=dict(
        m0=dict(
            model_name="model:MLP",
            model_in_shape=(80),
            model_out_shape=(14),
        ),
    ),
)

l_config.update(learner_config)
l_config.update(env_config)
l_config.update(player_config)

game = Game(l_config, training=False, register_handle=register)
env_info = game.env.get_env_info()
n_actions = env_info["n_actions"]
n_agents = env_info["n_agents"]
print(n_agents)
print(n_actions)
game_result_info = game.run_episode()
print(game_result_info)
time.sleep(1000)