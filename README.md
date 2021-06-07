# Fighting

## install 

sudo apt-get install openjdk-8-jre xvfb

pip install gym py4j port_for opencv-python

cd FTG/malib && pip install -e .

## Command

所有命令需要在 FTG 文件夹下运行

### Test env

xvfb-run -s "-screen 0 600x400x24" python malib/example/test_env.py

### Train DQN

xvfb-run -s "-screen 0 600x400x24" python malib/example/dqn_fighting.py

tensorboard --logdir ./logs --reload_multifile True

### 参数控制方式

* N-step Learning。  修改 data_config 中的 tra_len=1 / 3
* Double DQN。   修改 trainer_config 中的 double=True / False
* NoisyNet。  修改 player_config 中的 noisy=True / False, player_config 中的 epsilon_enable=False / True
* Loss。  修改 trainer_config 中的 trainer_config=MSELoss / smooth_l1_loss
* Dueling DQN. 修改 player_config 中的 dueling=True / False
* Categorical DQN. 修改 trainer_config 中的 trainer_name=CategoricalDQNTrainer / DQNTrainer, model_config中的模型为 Categorical / MLP
    * v_min, v_max, atom_size 控制值分布的精度。