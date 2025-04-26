from abc import ABC, abstractmethod


class BaseAgent(ABC):
    def __init__(self, env, discount_factor=0.99):
        self.env = env
        self.discount_factor = discount_factor

    @abstractmethod
    def train(self):
        """Train the agent on the environment."""
        pass

    @abstractmethod
    def get_policy(self):
        pass

    @abstractmethod
    def get_value_function(self):
        pass
