from malib import makeenv
import random
env = makeenv("fighting")

while True:
    obs = env.reset()
    reward, done, info = 0, False, None
    ep_len = 0
    while not done:
        act = random.randint(0, 10)
        # TODO: or you can design with your RL algorithm to choose action [act] according to  state [obs]
        new_obs, reward, done, info = env.step(dict(p0=act))
        print("current hp: own {} vs opp {}, reward {}, episode reward {:.2f}".format(info["own_hp"], info["opp_hp"], reward, info["episode_reward"]))
        ep_len += 1
        if not done:
            # TODO: (main part) learn with data (obs, act, reward, new_obs)
            # suggested discount factor value: gamma in [0.9, 0.95]
            pass
        elif info is not None:
            print("round result: own hp {} vs opp hp {}, win_result {}, episode_reward {:.2f}, hp diff {}".format(info["own_hp"], info["opp_hp"],
                    info["win_result"]["p0"], info["episode_reward"], info["hp_diff"]))
        else:
            # java terminates unexpectedly
            pass