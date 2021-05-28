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
        ep_len += 1
        print(ep_len, reward)
        if not done:
            # TODO: (main part) learn with data (obs, act, reward, new_obs)
            # suggested discount factor value: gamma in [0.9, 0.95]
            pass
        elif info is not None:
            print("round result: own hp {} vs opp hp {}, you {}".format(info[0], info[1], 'win' if info[0]>info[1] else 'lose'))
        else:
            # java terminates unexpectedly
            pass