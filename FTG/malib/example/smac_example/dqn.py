import ray
import sys
import time
import torch
import numpy as np

sys.path.insert(0, "../malib")
from malib.utils import Logger, default_league_config, default_learner_config
from malib.utils import regist, register_init
from malib.learn import League, Learner
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
from malib.action import ProbSampleAction, RandomAction, GreedyAction
from malib.feature import DictFeature

if torch.cuda.is_available():
    ray.init(address="auto")
else:
    ray.init()

register_handle = register_init()
regist("smacenv", SmacEnv)
regist("smacwrapper", SmacWrapper)
regist("smacobswrapper", SmacObsWrapper)
regist("smacactionwrapper", SmacActionWrapper)
regist("prob_action", ProbSampleAction)
regist("random_action", RandomAction)
regist("greedy_action", GreedyAction)
regist("dict_feature", DictFeature)
regist("scplayer", SCPlayer)

env_config = dict(
    config_name="env_config",
    smac_env=dict(
        raw_env="smacenv",
        wrapper=["smacwrapper", "smacobswrapper", "smacactionwrapper"],
        env_params=dict(map_name="8m"),
    ),
)
data_config = dict(
    config_name="data_config",
    data_to_save=dict(
        player_data=["feature", "obs", "model_out", "action"],
        other_data=["game_data", "reward"],
    ),
    train_data_num=256,
    tra_len=1,
    batch_size=128,
    data_async=False,
    data_capacity=2000,
    data_sample_mode="USWR",
)
eval_config = dict(
    config_name="eval_config",
    eval_game_number=5,
    total_episode_number=10,
    ray_mode="sync",
    eval_mode="env",  # env: 单个player在env中测试， dynamic：挑选对手，opponent_id:指定对手
    env_name="smac_env",
    players=["p0"],
    evaluator_num=2,
)
if torch.cuda.is_available():
    game_number = 5
else:
    game_number = 2

learner_config = dict(
    config_name="learner_config",
    game_number=game_number,
    env_name="smac_env",
    player_id="p0",
    local_data=True,
    learn_model_id=["m0"],
    ray_mode="sync",  # sync: ray.get, async: ray.wait
)


trainer_config = dict(
    config_name="trainer_config",
    use_gpu=torch.cuda.is_available(),
    gpu_num=1,
    trainer_mode="local",
    m0=dict(
        trainer_number=1,
        trainer_name="trainer:DQNTrainer",
        lr=0.0001,
        target_model_update_iter=30,
        EPSILON=0.9,
        GAMMA=0.9,
        # training_procedure= train_on_batch,
    ),
)
player_config = dict(
    config_name="player_config",
    players=["p0"],
    p0=dict(
        # player_name="agent:MAPlayer",
        player_name="scplayer",
        agents=["a0", "a1", "a2", "a3", "a4", "a5", "a6", "a7"],
        action_config=dict(
            action_name="greedy_action",
            epsilon=1.0,
            episode_count=10000000,
        ),
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
            model_params=dict(
                in_dim=(80),
                out_dim=(14),
                hidden_dim=512,
            ),
        ),
    ),
)

league_config_dict = dict(
    config_name="league_config",
    eval_players=["p0"],
    eval_auto=True,
    auto_save=False,
    standings_mode=[
        "reward",
        "winrate",
    ],  # reward:compute the reward,score：compute the score， winrate:compute the winrate
    env_name="cartpole_v0",
    workdir="logs/league",
)

config = default_learner_config()
config.update(learner_config)
config.update(player_config)
config.update(data_config)
config.update(env_config)
config.update(trainer_config)
config.update(eval_config)

league_config = default_league_config()
league_config.update(eval_config)
league_config.update(league_config_dict)
league_config.update(player_config)
league_config.update(env_config)

config.save()
league_config.save()


logger = Logger(config.log_config)
logger.config_info([config, league_config])


class MyLearner(Learner):
    def __init__(self, config, register_handle):
        super(MyLearner, self).__init__(config=config, register_handle=register_handle)
        self.build_games()
        self.build_trainers()
        self.init_games_weights()
        # self.start_data_thread()

    def learning_procedure(self, learner=None):
        t0 = time.time()
        data = self.ask_for_data(min_episode_count=3)
        t1 = time.time()
        result = self.learn_on_data(data)
        t2 = time.time()

        self.sync_weights(result)
        t3 = time.time()
        game_last_info = self.get_game_info()
        game_reward = [g_data["info"]["episode_reward"] for g_data in game_last_info]
        mean_reward1 = np.mean(game_reward)
        self.logger.add_scalar("p0/reward", mean_reward1, self.learn_step_number)
        t4 = time.time()

        logger.info(
            [
                "learner step number:{},train reward:{}".format(
                    self.learn_step_number, mean_reward1
                )
            ]
        )

        return result


if __name__ == "__main__":
    print('start ----------------------------------------------')
    league_cls = League.as_remote().remote
    league = league_cls(league_config, register_handle=register_handle)

    learner = MyLearner(config, register_handle=register_handle)
    for i in range(5000):
        learner.step()

        if i % 3 == 0:
            p = learner.get_training_player()
            league.add_player.remote(p)
    time.sleep(100000)
