class BaseEnv(object):
    """
    basic env class, all envs should inherit this class.

    the flowing methods must be implemented:
    reset()
    step()

    """

    def __init__(self):

        self.action_space = None
        self.observation_space = None

    def __getattr__(self, name):
        return getattr(self.env, name)

    def step(self, action):
        """run single step of environment.

        accepts an action and returns a tuple

        Args:
            action ([type]): [description]

        Returns:
            Tuple: (obs,reward,done,info)
        """
        raise NotImplementedError

    def reset(self):
        """set the environment to ready for a new episode

        Returns:
            observation ( object): the initial observation
        """
        raise NotImplementedError
