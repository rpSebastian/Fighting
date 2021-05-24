import os
import pickle
import random
from copy import deepcopy


class PlayerPool(object):
    def __init__(self, config):
        self.workdir = config.workdir
        self.player_dict = {}
        self.auto_save = config.auto_save

    def add_player(self, player):
        self.player_dict[player.player_name] = player
        if self.auto_save:
            self.save_player(player)

    def save_player(self, player):
        file_name = player.player_long_name + ".pth"
        file_path = os.path.join(self.workdir, file_name)
        with open(file_path, "wb") as f:
            pickle.dump(player.get_player_weights(), f)

    def get_player_by_name(self, player_name):
        # TODO: if use deepcopy?
        return self.player_dict[player_name]

    def get_player_random(self, player_id):
        player_name_li = list(self.player_dict.keys())
        player_name_list = [i for i in player_name_li if i.split("_")[0] == player_id]
        player_name = random.choice(player_name_list)
        player = self.player_dict[player_name]
        return player
