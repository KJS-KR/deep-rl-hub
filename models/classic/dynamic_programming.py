import numpy as np


class PolicyIteration:
    def __init__(self, env, discount_factor=0.99, theta=1e-10):
        self.env = env
        self.discount_factor = discount_factor
        self.theta = theta
        self.policy = None
        self.value_function = None
        self.state_space = env.observation_space.n
        self.action_space = env.action_space.n
        self.initialize_policy_and_value_function()

    def initialize_policy_and_value_function(self):
        self.policy = np.ones((self.state_space, self.action_space)) / self.action_space
        self.value_function = np.zeros(self.state_space)
