# MALIB

![black style](https://github.com/CASIA-CRISE-GML/malib/actions/workflows/.github/workflows/autoblack.yml/badge.svg)
![package](https://github.com/CASIA-CRISE-GML/malib/actions/workflows/.github/workflows/python-package.yml/badge.svg)


## 介绍

MALIB是一个通用博弈对抗学习平台，从环境封装、算法实现、分布式计算、应用集成、平台管理等方面提供统一的开发接口。
<img width="700" height="auto" src="./docs/images/platform.png">

**MALIB的主要特性和设计目标为：**

-   **标准的模块化抽象，提供统一的API**。环境、模型、智能体、玩家、学习器、评估器等模块均抽象成统一的标准的接口，用一套统一的接口实现单智能体、多智能体、单个玩家、多个玩家、分层/不分层决策、中心化/去中心化决策等各类不同的强化学习和博弈对抗算法。
-   **易用、高效、稳定的分布式学习平台**。针对强化学习算法和博弈算法对算力的需求，提供对分布式计算的完整支持。使用者不需要关心分布式的实现和资源管理，只需要定义算法。平台充分利用硬件资源提供高效、稳定的计算。
-   **提供丰富的可用的算法资源，可快速在新环境或对新算法进行验证**。平台内置最常用的学习环境，进行 统一的封装和个性化的定制。内置经典强化学习和博弈学习算法。提供多种算法样例，实现方便的算法定制、环境定制和算法验证。
-   **在硬件上对集群和计算资源进行统一管理和调度**。使用K8s和Docker构建计算集群硬件管理和资源调度系统。



## 安装

[安装指导](./Installation.md)

## 快速入门和详细介绍

见[项目文档](https://github.com/CASIA-CRISE-GML/MALib-Tutorial).

## TODO
 #### 博弈对抗环境

 博弈对抗环境统一使用[OpenAI Gym接口](https://gym.openai.com/docs/)进行封装，使用环境修饰类对环境进行二次定制。

- [x] [OpenAI Gym](https://github.com/openai/gym)
- [x] [StarCraft II - PySC2](https://github.com/deepmind/pysc2)
- [x] [Google Research Football](https://github.com/google-research/football)
- [x] [rlcard(德扑、麻将等)](https://github.com/datamllab/rlcard)
- [x] [SMAC - StarCraft Multi-Agent Challenge](https://github.com/oxwhirl/smac)
- [ ] [ViZDoom](https://github.com/mwydmuch/ViZDoom)
- [ ] 飞行器对抗

## 实验和结果

[示例代码和实验结果](example/README.md)



