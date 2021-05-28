# Installation

-  ray
-  torch
-  gym

MALib是一个可以安装的 python 包， 如果一切顺利，可以使用如下命令进行安装。

```bash
        python setup.py install
        # or
        pip install . 

        # remove
        pip uninstall malib
```

## Environment Installation

### Google Football

Better install `gfootball` from their website, `https://github.com/google-research/football`

#### Linux

```bash

# 安装 pygame 依赖项
sudo apt install git cmake build-essential libgl1-mesa-dev \
        libsdl2-dev libsdl2-image-dev libsdl2-ttf-dev libsdl2-gfx-dev \
        libboost-all-dev libdirectfb-dev libst-dev mesa-utils \
        xvfb x11vnc libsdl-sge-dev libsdl-image1.2-dev libsdl-mixer1.2-dev \
        libsdl-ttf2.0-dev libsmpeg-dev libsdl1.2-dev libportmidi-dev \
        libswscale-dev libavformat-dev libavcodec-dev libfreetype6-dev

# 安装 gfootball
pip install gfootball
```

## Development

### 设置开发环境

多人合作开发一个项目应该有统一的开发工具、运行环境、代码风格、命名规范。推荐使用 **VSCode** 作为开发环境。
下面主要是设置代码风格检查，它会自动在 commit 时候帮助你检查甚至是修改代码，以满足本项目共同遵守的规范。

- isort      import 排序与格式化
- pre-commit 自动检查代码格式
- black      格式化代码
- flake8     Python 风格检查


```bash
        # 以 develop 模式进行安装是在 `PYTHONPATH` 中创建了一个软链接，你在该目录下所做的更改会直接生效，不用重新安装
        pip install -r requirment-dev.txt
        
        python setup.py develop
        # or
        pip install --editable . 
        
        # 安装 pre-commit 在commit时候自动进行代码检查
        pre-commit install
```

`.pre-commit-config.yaml` 中定义了在 commit 前自动检查代码风格的操作，会自动检查：

 - import 顺序、长度等；会修改代码
 - black 风格化；会修改代码
 - flake8 Python 风格检查；不会修改代码

 目前这个 `.pre-commit-config.yaml` 文件比较严格，如果有任何的风格欠缺就会禁止 commit 操作。
 当你的 commit 遇到问题时候：
 
 - 可以运行 `pre-commit run --all-files` 或多次 `git commit -m xxx` 将代码风格修改成同一风格。
 - 如果还有报错，可以修改代码，满足 `pre-commit` 条件。例如， `if d == True:` 这句话会触发`flake8 E712` 该写成 `if d:`。
 - 最差情况是在 commit 时候加上， `--no-verify` 跳过检查， `git commit --no-verify -am "messages"`

### Git 流程

[Git Workflow](https://github.com/CASIA-CRISE-GML/MALib-Tutorial/blob/main/git_workflow.md)
[如何 Git](https://github.com/CASIA-CRISE-GML/MALib-Tutorial/blob/main/how-to-git.pdf)
[如何 Pull Request](https://github.com/CASIA-CRISE-GML/MALib-Tutorial/blob/main/how-to-pull-request.pdf)
