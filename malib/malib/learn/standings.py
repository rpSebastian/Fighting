import collections

import numpy as np
import ray

from malib.utils import Logger


class Standings(object):
    def __init__(self, config):
        self.workdir = config.workdir
        self.log_name = "standings.txt"
        self.logger = Logger(config.log_config)
        self.logger.info("standings init ...")
        # self.log=open(self.log_name,'w')
        self.standings_mode = config.standings_mode
        if "score" in self.standings_mode:
            self.score = collections.defaultdict(lambda: 0)
        if "reward" in self.standings_mode:
            self.reward = collections.defaultdict(lambda: 0)
        if "winrate" in self.standings_mode:
            self.WR = collections.defaultdict(
                lambda: collections.defaultdict(lambda: 0)
            )
            self.win = collections.defaultdict(
                lambda: collections.defaultdict(lambda: 0)
            )
            self.loss = collections.defaultdict(
                lambda: collections.defaultdict(lambda: 0)
            )
            self.draw = collections.defaultdict(
                lambda: collections.defaultdict(lambda: 0)
            )

    def update(self, result, player_info, opponent_info):
        total_game_num = len(result)
        player_id = player_info["player_id"]
        player_name = player_info["player_name"]

        assert type(opponent_info) is not list, "还不支持多个对手的情形"
        if "reward" in self.standings_mode:
            r_list = [i["episode_reward"] for i in result]
            mean_r = np.mean(r_list)
            self.reward[player_info["player_name"]] = mean_r
            tb_name = player_info["player_id"] + "/eval_reward"
            self.logger.add_scalar(tb_name, mean_r, player_info["learn_step_number"])

        if "winrate" in self.standings_mode:
            if opponent_info is None:
                player_win_count = 0
                player_draw_count = 0
                player_loss_count = 0
                for res in result:
                    win_result = res["win_result"]
                    if_win = win_result[player_id]
                    if if_win == 1:
                        player_win_count += 1
                    elif if_win == 0:
                        player_draw_count += 1
                    else:
                        player_loss_count += 1
                self.win[player_name] = player_win_count
                self.draw[player_name] = player_draw_count
                self.loss[player_name] = player_loss_count
                player_win_rate = player_win_count / len(result)
                tb_name = player_id + "/eval_win_rate"
                self.logger.add_scalar(
                    tb_name, player_win_rate, player_info["learn_step_number"]
                )

    def get_standings(self):
        result = {}
        if "reward" in self.standings_mode:
            result["reward"] = self.reward
        if "winrate" in self.standings_mode:
            res = {}
            res["win"] = self.win
            res["loss"] = self.loss
            res["draw"] = self.draw
            result["winrate"] = res
        return result

    def get_WR(self, p1_id, p2_id):
        return self.WR[p1_id][p2_id]

    def get_score(self, p1_id):
        return self.score[p1_id]

    def flush(self, info):
        pass

    @classmethod
    def as_remote(
        cls,
        num_cpus: int = 1,
        num_gpus: int = 0,
        memory: int = None,
        object_store_memory: int = None,
        resources: dict = None,
    ) -> type:
        """[summary]

        Args:
            num_cpus (int, optional): [description]. Defaults to 1.
            num_gpus (int, optional): [description]. Defaults to 0.
            memory (int, optional): [description]. Defaults to None.
            object_store_memory (int, optional): [description]. Defaults to None.
            resources (dict, optional): [description]. Defaults to None.

        Returns:
            type: [description]
        """
        return ray.remote(
            num_cpus=num_cpus,
            num_gpus=num_gpus,
            memory=memory,
            object_store_memory=object_store_memory,
            resources=resources,
        )(cls)
