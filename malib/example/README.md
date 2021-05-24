

## 算法

| 环境        | 运行命令                                | 实验结果                                                           | 参数设置                                                                                                   |
| ----------- | --------------------------------------- | ------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------- |
| Cartpole V0 | ```python example/dqn_cartpole_v0.py``` | <img src="results/dqn_cartpole_v0_result.png" style="zoom:50%;" /> | bs:128<br />batch_num:256<br />LR:0.001<br />data_capacity:2000<br />game_num:5<br />target_update_iter:30 |
| Cartpole V0 | ```python example/ppo_cartpole_v0.py``` | <img src="results/ppo_cartpole_V0.png" style="zoom:50%;" />        | bs:64<br />batch_num:64<br />LR:0.00068<br />data_capacity:64<br />game_num:20                             |



