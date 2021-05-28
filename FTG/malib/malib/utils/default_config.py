import copy
import datetime

from .config import Config

data_config = dict(
    config_name="data_config",
    data_to_save=dict(
        player_data=["feature", "obs", "model_out", "action"],
        other_data=[],
        preprocessing_data=[],
    ),
    data_players=["p0"],
    train_data_num=256,
    tra_len=1,
    batch_size=128,
    data_async=False,
    data_capacity=2000,
    data_sample_mode="USWR",
    process_on_episode=False,
)
learner_config = dict(
    config_name="learner_config",
    game_number=5,
    local_data=True,
    env_name="cartpole_v0",
    learn_player_id="p0",
    ray_get_later=False,
    players=["p0"],
    learn_agent_id=["a0"],
    ray_mode="sync",  # sync: ray.get, async: ray.wait
)
model_config = dict(
    config_name="model_config",
    m0=dict(
        model_name="model:MLP",
        model_in_shape=(4),
        model_out_shape=(2),
    ),
)
player_config = dict(
    config_name="player_config",
    players=[],
)

trainer_config = dict(
    config_name="trainer_config",
    use_gpu=False,
    gpu_num=1,
    trainer_mode="local",
    m0=dict(
        trainer_number=1,
        trainer_name="learner:DQNTrainer",
        lr=0.001,
        target_model_update_iter=30,
        EPSILON=0.9,
        GAMMA=0.9,
        training_procedure=None,
    ),
)

eval_config = dict(
    config_name="eval_config",
    eval_game_num=10,
    total_episode_number=20,
    ray_mode="sync",
    env_name=None,
    players=None,
    evaluator_num=1,
)
league_config = dict(
    config_name="league_config",
    eval_players=["p0"],
    eval_auto=True,
    auto_save=True,
    standings_mode=["score"],
    logger=None,
)
log_config = dict(
    config_name="log_config",
    log_name="default",
    time_now=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    log_dir="logs/test",
    tb_dir="logs/tmp",
)

env_config = dict(config_name="env_config")


def default_learner_config():

    config = Config()
    config.update(learner_config)
    config.update(data_config)
    config.update(trainer_config)
    config.update(eval_config)
    config.update(log_config)
    config.update(env_config)
    config.update(player_config)

    return config


def default_league_config():
    config = Config()
    config.update(eval_config)
    config.update(league_config)
    config.update(log_config)
    config.update(env_config)
    config.update(player_config)

    return config
