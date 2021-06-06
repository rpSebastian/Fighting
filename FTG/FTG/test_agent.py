import torch
import torch.nn as nn
import pickle
from fightingice_env import FightingiceEnv

class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=256, device: str = "cpu") -> None:
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim
        self.device = device
        self.linear1 = nn.Sequential(
            nn.Linear(self.in_dim, self.hidden_dim),
            nn.ReLU(),
        )
        self.linear1_2 = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
        )

        self.linear2 = nn.Sequential(nn.Linear(self.hidden_dim, self.out_dim))

    def forward(self, x):
        x = self.linear1(x)
        x = self.linear1_2(x)
        x = self.linear2(x)
        return x

    def get_weights(self):
        return self.state_dict()

    def set_weights(self, weights):
        self.load_state_dict(weights)


class Agent():
    def __init__(self, weights_path, in_dim=144, out_dim=40):
        self.model = MLP(in_dim, out_dim)
        with open(weights_path, "rb") as f:
            info = pickle.load(f)
            weights = info["m0"]
        self.model.set_weights(weights)

    def action(self, obs):
        feature = torch.tensor(obs).float()
        model_out = self.model(feature)
        action_index = torch.argmax(model_out)
        action = action_index.item()
        return action

def run_episode(env, env_args, agent):
    o = env.reset(env_args=env_args)
    step_num = 0
    while True:
        step_num += 1
        a = agent.action(o)
        print(a)
        o, r, d, i = env.step(a)
        if d:
            break
        own_hp = i[0]
        opp_hp = i[1]
        print("step: {} own hp: {}, opp_hp: {}".format(step_num, own_hp, opp_hp))
    if i is None:
        # Java terminates unexpectedly
        return run_episode(env, env_args, agent)
    own_hp = i[0]
    opp_hp = i[1]
    hp_diff = own_hp - opp_hp
    if own_hp > opp_hp:
        win = 1
    else:
        win = 0
    print("round result: own hp {} vs opp hp {}, you {}".format(own_hp, opp_hp, 'win' if win else 'lose'))
    return hp_diff, win

def test():
    # weights_path = "logs/league/p0_389_2021-05-29-23-45-58.pth"
    # weights_path = "logs/league/p0_83_2021-05-30-10-06-39.pth"
    # weights_path = "logs/league/p0_36_2021-05-30-17-06-14.pth"
    # weights_path = "logs/league/p0_41_2021-05-30-17-45-03.pth"
    # weights_path = "logs/league/p0_66_2021-06-04-08-03-08.pth"
    # weights_path = "logs/league/p0_116_2021-06-04-14-24-29.pth"
    weights_path = "logs/league/p0_131_2021-06-04-16-15-22.pth"
    agent = Agent(weights_path)
    env = FightingiceEnv(port=4232)
    env_args = ["--fastmode", "--disable-window", "--grey-bg", "--inverted-player", "1", "--mute"]
    env_args2 = ["--fastmode", "--grey-bg", "--inverted-player", "1", "--mute"]
    for i in range(100):
        run_episode(env, env_args2, agent)

test()
