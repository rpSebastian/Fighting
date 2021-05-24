import collections
import datetime
import os

import ray

from malib.utils import Logger

from .evaluator import VecEvaluator
from .match import Match
from .player_pool import PlayerPool
from .standings import Standings


class League:
    """league manages all players

    .. the function of league is::

    1. manage all players, save player to local files, select player for evaluator
    2. evaluate players
    3. maintain the standings of all players
    """

    def __init__(self, config, register_handle=None):
        self.workdir = config.workdir
        if not os.path.exists(self.workdir):
            os.makedirs(self.workdir)
        self.config = config
        self.register_handle = register_handle
        standings_cls = Standings.as_remote().remote
        self.standings = standings_cls(self.config)
        self.eval_auto = config.eval_auto
        self.vec_evaluator = self.make_vec_evaluator()
        self.vec_evaluator.start.remote()
        self.match = Match(self.config)
        self.player_pool = PlayerPool(self.config)
        self.eval_mode = config.eval_config.eval_mode

        self.player_index = collections.defaultdict(lambda: 0)
        self.logger = Logger(config.log_config)

    def add_player(self, player, eval=None):
        eval_player = None
        if eval:
            eval_player = eval
        else:
            eval_player = self.eval_auto

        time_now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        player_id = player.player_id
        player.use_ref_weights()
        player_name = "{}_{}".format(player_id, self.player_index[player_id])
        player_long_name = "{}_{}".format(player_name, time_now)
        player.update_player_name(player_name, player_long_name)
        self.player_index[player_id] += 1
        self.player_pool.add_player(player)
        if eval_player:
            self.eval_player(player)

    def eval_player(self, player):
        """evaluate the player and update player infos on standings

        opponent can be BasePlayer object when the game has two players
        or a list of BasePlayer objects when the game has more than two players

        Args:
            player (BasePlayer): player to eval
        """
        opponent = None
        opponent_info = None
        if self.eval_mode == "env":
            pass
        elif self.eval_mode == "dynamic":
            opponent = self.select_opponent(player)
        else:
            opponent = self.get_player(self.eval_mode)

        if opponent is not None:
            if type(opponent) is list:
                opponent_info = [i.info for i in opponent]
            else:
                opponent_info = opponent.info

        self.vec_evaluator.eval_player.remote(player, opponent)

    def get_eval_state(self):
        eval_state_info = self.vec_evaluator.state_info.remote()
        return ray.get(eval_state_info)

    def get_player(self, player_name=None, player_id=None, strategy="random"):
        if player_name:
            return self.player_pool.get_player_by_name(player_name)
        elif player_id:
            if strategy == "random":
                assert player_id is not None, "player_id must be specified"
                return self.player_pool.get_player_random(player_id)
        else:
            raise ValueError(
                "must set a params of get_player between player_name and player_id"
            )

    def select_opponent(self, player):
        pass

    def update_standings(self, eval_result, player_info, opponent_info):
        self.standings.update.remote(eval_result, player_info, opponent_info)

    def make_vec_evaluator(self):
        eval_cls = VecEvaluator.as_remote().remote
        remote_evaluator = eval_cls(
            config=self.config,
            standings=self.standings,
            register_handle=self.register_handle,
        )
        return remote_evaluator

    def get_standings(self):
        return ray.get(self.standings.get_standings.remote())

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
