from .register import make, register

"""
   _|_|_|  _|      _|  _|      _|  
 _|          _|  _|    _|_|  _|_|  
 _|  _|_|      _|      _|  _|  _|  
 _|    _|      _|      _|      _|  
   _|_|_|      _|      _|      _|  
"""
register(
    id="cartpole_v0",
    params={
        "raw_env": "gym:CartPole_V0",
        "wrapper": [
            "gym:ObsWrapper",
            "gym:AWrapper",
            "gym:FrameLimitWrapper",
        ],
    },
)
register(
    id="cartpole_v1",
    params={
        "raw_env": "gym:CartPole_V1",
        "wrapper": [
            "gym:ObsWrapper",
            "gym:AWrapper",
            "gym:FrameLimitWrapper",
        ],
    },
)
"""
 __   ___  __   __  ___  __                 
/ _` |__  /  \ /  \  |  |__)  /\  |    |    
\__> |    \__/ \__/  |  |__) /~~\ |___ |___ 
"""
register(
    id="gfootball_11v11_easy_stochastic_example",
    params={
        "raw_env": "gfootball:GFootBall",
        "env_params": {"env_name": "11_vs_11_easy_stochastic"},
    },
)
"""
   _____ _              _____            __ _     _____ _____ 
  / ____| |            / ____|          / _| |   |_   _|_   _|
 | (___ | |_ __ _ _ __| |     _ __ __ _| |_| |_    | |   | |  
  \___ \| __/ _` | '__| |    | '__/ _` |  _| __|   | |   | |  
  ____) | || (_| | |  | |____| | | (_| | | | |_   _| |_ _| |_ 
 |_____/ \__\__,_|_|   \_____|_|  \__,_|_|  \__| |_____|_____|
"""

sc2_base_confg = {
    "map_name": "DefeatRoaches",
    "battle_net_map": False,
    "home_player_config": {
        "race": "zerg",  # ['random', 'protoss', 'terran', 'zerg']
        "name": None,  # name in replay
    },
    "away_players_config": {
        "bot": {"race": "zerg", "difficulty": "easy", "build": None},
    },
    "agent_interface_format": {
        "feature_screen": "84",
        "feature_minimap": "64",
        "rgb_screen": None,
        "rgb_minimap": None,
        "action_space": None,
        "use_feature_units": False,
        "use_raw_units": False,
    },
    "step_mul": 8,
    "game_steps_per_episode": None,
    "disable_fog": False,
    "visualize": True,
}
register(
    id="SC2_DefeatRoaches",
    params={"raw_env": "pysc2:SC2_ENV", "env_params": sc2_base_confg},
)

"""
 ____  __    ___    __    ____  ____  
(  _ \(  )  / __)  /__\  (  _ \(  _ \ 
 )   / )(__( (__  /(__)\  )   / )(_) )
(_)\_)(____)\___)(__)(__)(_)\_)(____/ 
"""
register(id="leduc_holdem", params={"raw_env": "rlcard:LeducHoldem"})
register(id="mahjong", params={"raw_env": "rlcard:MahJong"})
