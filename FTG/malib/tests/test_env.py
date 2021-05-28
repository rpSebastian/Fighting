import sys
import time

import pytest

import malib


def test_envs():
    """registry stores all the supported environments"""
    from malib.aienv.register import registry

    # print(registry.envs)
    assert len(registry.envs) > 0, "There is no supported environment"


def test_gfootball():
    """gfootball"""
    # skip MacOS and windows
    if sys.platform.startswith("darwin") or sys.platform.startswith("win"):
        return
    try:
        import gfootball
    except ImportError:
        return

    env1 = malib.makeenv("gfootball_11v11_easy_stochastic")
    # env1.render()
    t0 = time.time()
    for _ in range(10):
        env1.reset()
        done = False
        while not done:
            action = env1.action_space.sample()
            observation, reward, done, info = env1.step(action)
            # print(action)
            # print(observation)
            # print(reward)
        t1 = time.time()
        print(t1 - t0)
        t0 = t1
