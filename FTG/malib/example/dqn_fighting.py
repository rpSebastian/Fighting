import random
import sys
import time
import pickle

import numpy as np
import ray
import torch

sys.path.insert(0, "../malib")

from malib.learn import Learner, League
from malib.utils import Logger, default_learner_config, default_league_config
from malib.utils import regist, register_init
from malib.action import GreedyAction
from malib.feature import TensorFeature

ray.init()
# ray.init(address="auto")

register_handle = register_init()
regist("greedy_action", GreedyAction)
regist("tensor_feature", TensorFeature)
data_config = dict(
    config_name="data_config",
    data_to_save=dict(
        player_data=["feature", "obs", "model_out", "action"],
        other_data=["game_data", "reward"],
    ),
    train_data_num=10240,
    tra_len=3,
    batch_size=1024,
    data_async=False,
    data_capacity=200000,
    data_sample_mode="USWR",
)
eval_config = dict(
    config_name="eval_config",
    eval_game_number=10,
    total_episode_number=100,
    ray_mode="sync",
    eval_mode="env",  # env: 单个player在env中测试， dynamic：挑选对手，opponent_id:指定对手
    env_name="fighting",
    players=["p0"],
    evaluator_num=1,
)
if torch.cuda.is_available():
    game_number = 10
else:
    game_number = 1

learner_config = dict(
    config_name="learner_config",
    game_number=game_number,
    env_name="fighting",
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
        lr=0.001,
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
        player_name="agent:SAPlayer",
        agents=["a0"],
        action_config=dict(
            action_name="greedy_action",
            epsilon=1.0,
            episode_count=20000,
        ),
        feature_config="tensor_feature",
    ),
    agent_config=dict(
        a0=dict(
            agent="agent:BaseAgent",
            model_id="m0",
        ),
    ),
    model_config=dict(
        m0=dict(model_name="model:MLP", model_params=dict(in_dim=(144), out_dim=(40))),
    ),
)
league_config_dict = dict(
    config_name="league_config",
    eval_players=["p0"],
    eval_auto=True,
    auto_save=True,
    standings_mode=[
        "reward",
        "winrate",
        "hpdiff"
    ],  # reward:compute the reward,score：compute the score， winrate:compute the winrate
    env_name="fighting",
    workdir="logs/league",
)

config = default_learner_config()
config.update(learner_config)
config.update(player_config)
config.update(data_config)

config.update(trainer_config)
config.update(eval_config)

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
        self.start_data_thread()
        # self.init_games_weights_from_file()

    def init_games_weights_from_file(self):
        weights_path = "logs/league/p0_0_2021-05-30-20-44-26.pth"
        with open(weights_path, "rb") as f:
            info = pickle.load(f)
            weights = info["m0"]
        result = {}
        result["train_result"] = {}
        result["train_result"]["weights"] = weights
        result["player_id"] = "p0"
        result["model_id"] = "m0"
        self.sync_weights(result)

    def learning_procedure(self, learner=None):
        t0 = time.time()
        data = self.ask_for_data(min_episode_count=0)
        t1 = time.time()
        result = self.learn_on_data(data)
        t2 = time.time()
        self.sync_weights(result)
        t3 = time.time()
        game_last_info = self.get_game_info()
        
        game_reward = [g_data["info"]["episode_reward"] for g_data in game_last_info]
        mean_reward1 = np.mean(game_reward)
        game_hp_diff = [g_data["info"]["hp_diff"] for g_data in game_last_info]
        mean_hp_diff = np.mean(game_hp_diff)
        game_win_result = [g_data["info"]["win_result"]["p0"] for g_data in game_last_info]
        win_rate = np.mean(game_win_result)


        self.logger.add_scalar("p0/reward", mean_reward1, self.learn_step_number)
        self.logger.add_scalar("p0/hp_diff", mean_hp_diff, self.learn_step_number)
        self.logger.add_scalar("p0/win_rate", win_rate, self.learn_step_number)
        t4 = time.time()
        logger.info("{} {} {} ".format(t1 - t0, t2 - t1, t3 - t2))
        g0 = game_last_info[0]["info"]
        logger.info("step_num: {} own hp: {} opp hp: {}".format(g0["step_num"], g0["own_hp"], g0["opp_hp"]))
        logger.info(
            [
                "learner step number:{},train reward:{}, hp diff:{}, win rate: {}, epsilon: {}".format(
                    self.learn_step_number, mean_reward1, mean_hp_diff, win_rate, self.get_epsilon()
                )
            ]
        )

        return result


if __name__ == "__main__":
    league_cls = League.as_remote().remote
    league = league_cls(league_config, register_handle=register_handle)

    learner = MyLearner(config, register_handle=register_handle)
    # p = learner.get_training_player()
    # league.add_player.remote(p)
    for i in range(50000000):
        learner.step()
        if i % 30 == 0:
            p = learner.get_training_player()
            league.add_player.remote(p)
    time.sleep(100000)
