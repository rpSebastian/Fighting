import random
import sys
import time

import numpy as np
import ray
import torch

sys.path.insert(0, "../malib")

from malib.utils import Logger, default_league_config, default_learner_config
from malib.utils import regist, register_init
from malib.learn import League, Learner

from malib.feature import TensorFeature
from malib.action import ProbSampleAction

# ray.init()
ray.init(address="auto")

register = register_init()
# time.sleep(4)


regist("action_func1", ProbSampleAction)

regist("feature_func1", TensorFeature)


def train_on_batch(self, data):
    """define the training procedure in trainer

    Args:
        data (Data): datas for training

    Returns:
        result (dict): training result
    """
    pass


def td_on_episode(storage, episode_data):
    data_return = {}

    for a_name, a_d in episode_data.items():
        data = {}
        # data_name = "reward"
        data_name = "td_reward"
        reward = list(a_d["reward"])
        for j in range(len(reward) - 2, -1, -1):
            reward[j] = reward[j] + 0.9 * reward[j + 1]
        data[data_name] = reward
        data_return[a_name] = data
    return data_return


regist("td_on_episode", td_on_episode)

data_config = dict(
    config_name="data_config",
    data_to_save=dict(
        player_data=["feature", "obs", "model_out", "action"],
        other_data=["game_data", "reward"],
        preprocessing_data=["td_reward"],
    ),
    train_data_num=64,
    tra_len=1,
    batch_size=64,
    data_async=False,
    data_capacity=64,
    data_sample_mode="FIFO",
    process_on_episode="td_on_episode",
)
eval_config = dict(
    config_name="eval_config",
    eval_game_number=5,
    total_episode_number=100,
    ray_mode="sync",
    eval_mode="env",  # env: 单个player在env中测试， dynamic：挑选对手，opponent_id:指定对手
    env_name="cartpole_v0",
    players=["p0"],
    evaluator_num=5,
)
if torch.cuda.is_available():
    game_number = 20
else:
    game_number = 2

learner_config = dict(
    config_name="learner_config",
    game_number=game_number,
    env_name="cartpole_v0",
    player_id="p0",
    local_data=False,
    learn_agent_id=["a0"],
    learn_model_id=["m0"],
    ray_mode="sync",  # sync: ray.get, async: ray.wait
)

trainer_config = dict(
    config_name="trainer_config",
    use_gpu=torch.cuda.is_available(),
    gpu_num=1,
    trainer_mode="remote",
    m0=dict(
        trainer_number=1,
        trainer_name="trainer:PPOTrainer",
        lr=0.00068,
        gamma=0.98,
        lmbda=0.95,
        es_clip=0.1,
        epoch=3,
        # training_procedure= train_on_batch,
    ),
)

player_config = dict(
    config_name="player_config",
    players=["p0"],
    p0=dict(
        player_name="agent:SAPlayer",
        agents=["a0"],
        action_config="action_func1",
        feature_config="feature_func1",
    ),
    agent_config=dict(
        a0=dict(agent="agent:BaseAgent", model_id="m0"),
    ),
    model_config=dict(
        m0=dict(
            model_name="model:AC", model_params=dict(in_dim=(4), out_dim=([(2), (1)]))
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

config.update(trainer_config)
config.update(eval_config)
# example: how to save config to python file
# config.save(name="test")
# example: how to load config from python file
# config.load_config("logs/test/config.py")

league_config = default_league_config()
league_config.update(eval_config)
league_config.update(league_config_dict)
league_config.update(player_config)

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
        data = self.ask_for_data(min_episode_count=0)
        t1 = time.time()
        self.clear_data()
        result = self.learn_on_data(data)
        t2 = time.time()

        self.sync_weights(result)
        t3 = time.time()
        game_last_info = self.get_game_info()
        game_reward = [g_data["info"]["episode_reward"] for g_data in game_last_info]
        mean_reward1 = np.mean(game_reward)
        self.logger.add_scalar("p0/reward", mean_reward1, self.learn_step_number)
        t4 = time.time()
        # print(t1 - t0, t2 - t1, t3 - t2, t4 - t3)

        logger.info(
            [
                "learner step number:{},train reward:{}".format(
                    self.learn_step_number, mean_reward1
                )
            ]
        )

        return result


if __name__ == "__main__":
    league_cls = League.as_remote().remote
    league = league_cls(league_config, register_handle=register)

    learner = MyLearner(config, register_handle=register)
    for i in range(5000):
        learner.step()

        if i % 3 == 0:
            p = learner.get_training_player()
            league.add_player.remote(p)
            # p0 = league.get_player(player_id="p0")
            # print(p0.player_name)
            # print(p0.player_long_name)
            # res = league.get_standings()
            # # learner.sync_weights(player_w_dict=p0.get_player_weights(),player_id=p0.player_id)
            # # learner.init_games_weights(player_w_dict=p0.get_player_weights(),player_id=p0.player_id)
            # print(res)
            # print("==========================================")
            # eval_state = league.get_eval_state()
            # print(eval_state)
        # import os

        # os._exit(0)

    time.sleep(100000)
