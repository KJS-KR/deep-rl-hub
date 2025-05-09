from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import os


class TensorboardLogger:
    def __init__(self, log_dir="./logs", experiment_name="default"):
        """
        Tensorboard SummaryWriter 래퍼
        :param log_dir : 로그 디렉토리
        :param experiment_name : 하위 폴더명 (e.g, PPO_Run_1)
        """

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        full_log_dir = os.path.join(log_dir, f"{experiment_name}_{timestamp}")
        self.writer = SummaryWriter(log_dir=full_log_dir)

    def log_scalar(self, tag, value, step):
        """
        스칼라 값 기록 (e.g., reward, loss)
        """
        self.writer.add_scalar(tag, value, step)

    def log_histogram(self, tag, values, step):
        """
        히스토그램 기록 (e.g., 정책 확률 분포 등)
        """
        self.writer.add_histogram(tag, values, step)

    def close(self):
        """
        Writer 종료
        """
        self.writer.close()


if __name__ == "__main__":
    tb_logger = TensorboardLogger(experiment_name="policy_iter")

    tb_logger.log_scalar("reward/episode", reward, episode)
    tb_logger.close()
