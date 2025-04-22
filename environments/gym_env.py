import gym
import numpy as np


class GymEnv:
    """
    Wrapper for OpenAI Gym environments.
    """

    def __init__(self, env_name: str, render_mode: str = None):
        """
        Initialize the Gym environment.

        Args:
            env_name (str): Name of the Gym environment to load.
            render_mode (str): Rendering mode, e.g., "human" or None.
        """
        try:
            self.env = gym.make(env_name, render_mode=render_mode)
        except gym.error.Error as e:
            raise ValueError(f"Invalid environment name '{env_name}'.\n{e}")

        self.env_name = env_name
        self.render_mode = render_mode

        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

        # Reset and store initial state
        self.state, _ = (
            self.env.reset()
            if hasattr(self.env.reset(), "__iter__")
            else (self.env.reset(), {})
        )

    def step(self, action):
        """
        Take an action in the environment.
        Returns (next_state, reward, done, info)
        """
        result = self.env.step(action)
        if len(result) == 5:
            next_state, reward, terminated, truncated, info = result
            done = terminated or truncated
        else:
            next_state, reward, done, info = result
        return next_state, reward, done, info

    def reset(self, seed: int = None):
        """
        Reset the environment and return the initial state.
        """
        result = self.env.reset(seed=seed)
        self.state, _ = result if isinstance(result, tuple) else (result, {})
        return self.state

    def render(self):
        """
        Render the current state of the environment.
        """
        self.env.render()

    def close(self):
        """
        Close the environment and release resources.
        """
        self.env.close()

    def sample_action(self):
        """
        Sample a random action from the action space.
        """
        return self.action_space.sample()

    def info(self):
        """
        Return environment info summary.
        """
        return {
            "env_name": self.env_name,
            "observation_space": self.observation_space,
            "action_space": self.action_space,
            "render_mode": self.render_mode,
        }

    def __str__(self):
        return f"<GymEnv {self.env_name}, obs: {self.observation_space}, act: {self.action_space}>"


if __name__ == "__main__":
    # 환경 3개 중 하나 선택 가능
    env_names = ["CartPole-v1", "MountainCar-v0", "LunarLander-v2"]
    env = GymEnv(env_name=env_names[2], render_mode="human")

    print(env.info())

    state = env.reset()
    done = False
    while not done:
        action = env.sample_action()
        state, reward, done, info = env.step(action)
        env.render()

    env.close()
