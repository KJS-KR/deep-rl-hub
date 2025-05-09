from abc import ABC, abstractmethod
import torch


class BaseAgent(ABC):
    def __init__(
        self,
        env,
        logger,
        tb_logger,
        discount_factor=0.99,
    ):
        """
        :param env: OpenAI Gym과 호환되는 환경 인스턴스
        :param logger: 로깅을 위한 로거 인스턴스
        :param tb_logger: Tensorboard 로깅을 위한 로거 인스턴스
        :param discount_factor: 미래 보상에 대한 할인 계수 (γ)
        """
        self.env = env

        self.logger = logger
        self.tb_logger = tb_logger

        self.discount_factor = discount_factor

    @abstractmethod
    def train(self):
        """환경에서 에이전트를 학습시키는 핵심 루프"""
        pass

    @abstractmethod
    def get_policy(self):
        """학습된 정책 반환"""
        pass

    @abstractmethod
    def get_value_function(self):
        """상태 가치 함수 반환환"""
        pass


class BaseDPAgent(BaseAgent):
    def __init__(
        self,
        env,
        logger,
        tb_logger,
        discount_factor=0.99,
        theta=1e-6,
    ):
        """
        Base class for Dynamic Programming agents (e.g., Policy Iteration, Value Iteration)

        :param env: Gym 환경, 반드시 env.P를 포함해야 함
        :param discount_factor: 할인율 γ
        :param theta: 수렴 임계값
        """
        # env.P 유효성 검사
        if not hasattr(env, "P"):
            raise ValueError(
                f"[BaseDPAgent] The environment `{getattr(env, 'spec', 'Unknown')}` does not provide `env.P`, "
                f"which is required for model-based dynamic programming algorithms.\n"
                f"✔ 사용 가능한 예시: 'FrozenLake-v1', 'Taxi-v3', 'CliffWalking-v0'"
            )

        super().__init__(env, logger, tb_logger, discount_factor)
        self.state_n = env.observation_space.n
        self.action_n = env.action_space.n
        self.theta = theta

        # DP에서 사용되는 value function과 policy 초기화화
        self.value_function = None
        self.policy = None
        self.init_value_func()
        self.init_policy()

    def init_value_func(self):
        """가치 함수 초기화 (0으로 시작)"""
        self.value_function = torch.zeros(self.state_n)

    def init_policy(self):
        """무작위 균등 정책 초기화화"""
        self.policy = torch.ones((self.state_n, self.action_n)) / self.action_n


class BaseTabularAgent(BaseAgent):
    def __init__(
        self,
        env,
        logger,
        tb_logger,
        discount_factor=0.99,
        learning_rate=0.1,
    ):
        """
        :param learning_rate : Q-값 업데이트에 사용되는 학습률률
        """
        super().__init__(env, logger, tb_logger, discount_factor)
        self.learning_rate = learning_rate
        self.state_n = env.observation_space_n
        self.action_n = env.action_space_n
        self.q_table = None
        self.init_table()

    def init_table(self):
        """Q-테이블을 0으로 초기화화"""
        self.q_table = torch.zeros((self.state_n, self.action_n))
