from pathlib import Path
java_path = str(list(Path(__file__).absolute().parents)[6] / "FTG")
import sys
sys.path.insert(0, java_path)
from fightingice_env import FightingiceEnv
# from pathlib import Path
from malib.aienv.env import BaseEnv


class Fighting(BaseEnv):
    def __init__(self):
        self.env = FightingiceEnv(java_env_path=java_path)
        self.env_args = ["--fastmode", "--disable-window", "--grey-bg", "--inverted-player", "1", "--mute"]
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space

    def step(self, action):
        return self.env.step(action)

    def reset(self):
        return self.env.reset(env_args=self.env_args)

    def render(self):
        return self.env.render()

