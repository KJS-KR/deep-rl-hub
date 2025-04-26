# import gym

# # env = gym.make("Humanoid-v4", render_mode="human")
# # obs = env.reset()
# # for _ in range(1000):
# #     action = env.action_space.sample()
# #     obs, reward, terminated, truncated, info = env.step(action)
# #     env.render()


# from stable_baselines3 import PPO
# import gym

# env = gym.make("HalfCheetah-v4", render_mode="human")
# env.reset()

# model = PPO("MlpPolicy", env, verbose=1)
# model.learn(total_timesteps=100_000)


import gym


class MujocoEnv(gym.Env):
    """
    A wrapper for Mujoco environments in OpenAI Gym.
    """

    def __init__(self, env_name: str, render_mode: str = None):
        """
        Initialize the Mujoco environment.

        Args:
            env_name (str): Name of the Mujoco environment to load.
            render_mode (str, optional): Rendering mode, e.g., "human" or None.
        """

        super(MujocoEnv, self).__init__()

        try:
            self.env = gym.make(env_name, render_mode=render_mode)
        except gym.error.Error as e:
            raise ValueError(f"Invalid environment name '{env_name}'.\n{e}")

        self.env_name = env_name
        self.render_mode = render_mode

        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        self.state, _ = (
            self.env.reset()
            if hasattr(self.env.reset(), "__iter__")
            else (self.env.reset(), {})
        )

    def reset(self, **kwargs):
        """
        Reset the environment and return the initial state.

        Args:
            **kwargs: Additional keyword arguments for the reset method.

        Returns:
            The initial state of the environment.
        """
        result = self.env.reset(**kwargs)
        self.state, _ = result if isinstance(result, tuple) else (result, {})
        return self.state

    def step(self, action):
        """
        Take an action in the environment.

        Args:
            action: The action to take.

        Returns:
            A tuple (next_state, reward, done, info) where:
                - next_state: The next state of the environment.
                - reward: The reward received after taking the action.
                - done: A boolean indicating if the episode has ended.
                - info: Additional information from the environment.
        """
        result = self.env.step(action)
        if len(result) == 5:
            next_state, reward, terminated, truncated, info = result
            done = terminated or truncated
        else:
            next_state, reward, done, info = result
        return next_state, reward, done, info

    def render(self, mode="human"):
        """
        Render the current state of the environment.

        Args:
            mode (str): The rendering mode. Default is "human".
        """
        return self.env.render(mode=mode)

    def close(self):
        """
        Close the environment and release resources.
        """
        self.env.close()

    def sample_action(self):
        """
        Sample a random action from the action space.

        Returns:
            A random action from the action space.
        """
        return self.action_space.sample()

    def info(self):
        """
        Return environment info summary.

        Returns:
            A dictionary containing environment information.
        """
        return {
            "env_name": self.env_name,
            "observation_space": self.observation_space,
            "action_space": self.action_space,
            "render_mode": self.render_mode,
        }

    def __str__(self):
        """
        Return a string representation of the environment.
        """
        return f"MujocoEnv(env_name={self.env_name}, render_mode={self.render_mode})"
