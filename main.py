from agents.classic.dynamic_programming import PolicyIteration

from environments.gym_env import GymEnv

from utils.log import setup_logging
from utils.tb import TensorboardLogger

import argparse
import os


def main():
    # 로깅 셋업
    logger = setup_logging(log_dir="./logs", log_file="policy_iteration.log")
    tb_logger = TensorboardLogger(log_dir="./logs", experiment_name="policy_iter")

    env = GymEnv("Taxi-v3")
    agent = PolicyIteration(
        env,
        logger=logger,
        tb_logger=tb_logger,
        discount_factor=0.99,
        theta=1e-6,
    )
    agent.train()


if __name__ == "__main__":
    main()
