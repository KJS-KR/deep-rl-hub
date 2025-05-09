from agents.base.base_agent import BaseDPAgent
import numpy as np
import torch


class PolicyIteration(BaseDPAgent):
    def __init__(
        self,
        env,
        logger,
        tb_logger,
        discount_factor=0.99,
        theta=1e-6,
    ):
        super().__init__(
            env,
            logger,
            tb_logger,
            discount_factor,
            theta,
        )  # 여기에서 자동 검증됨

    def policy_evaluation(self):
        """
        현재 정책에 대해 value function을 반복적으로 평가
        수렴 조건은 각 상태의 가치 변화량이 0 이하일 때때
        """
        step = 0
        self.logger.info("Starting policy evaluation...")

        while True:
            delta = 0
            for s in range(self.state_n):
                v = 0
                for a in range(self.state_n):
                    action_prob = self.policy[s, a]
                    for prob, next_state, reward, done in self.env.P[s][a]:
                        v += (
                            action_prob  # pi(a|s)
                            * prob  # P(s'|s, a)
                            * (
                                reward
                                + self.discount_factor * self.value_function[next_state]
                            )
                        )
                delta = max(delta, torch.abs(v - self.value_function[s]))
                self.value_function[s] = v
            self.tb_logger.add_scalar(
                "PolicyEvaluation/value_function", v, self.env.step_count
            )
            self.tb_logger.add_scalar(
                "PolicyEvaluation/delta", delta, self.env.step_count
            )

            step += 1
            if delta < self.theta:
                self.logger.info(
                    f"Policy Evaluation converged after {step} steps with delta {delta:.4f}."
                )
                break

    def policy_improvement(self):
        """
        각 상태에 대해 가치가 가장 높은 행동을 선택하여 정책을 개선
        정책이 더 이상 변하지 않으면 종료
        """
        policy_stable = True
        for s in range(self.state_n):
            old_action = torch.argmax(self.policy[s]).item()
            action_values = torch.zeros(self.action_n)
            for a in range(self.action_n):
                for prob, next_state, reward, done in self.env.P[s][a]:
                    action_values[a] += prob * (
                        reward + self.discount_factor * self.value_function[next_state]
                    )
            best_action = torch.argmax(action_values).item()

            # 기존 정책과 비교해 변했는지 확인
            if old_action != best_action:
                policy_stable = False
                self.logger.info(
                    f"Policy improved at state {s}: {old_action} -> {best_action}."
                )

            # 정책 갱신 (one-hot)
            self.policy[s] = torch.nn.functional.one_hot(
                torch.tensor(best_action), self.action_n
            ).float()

        self.logger.info(
            f"Policy improvement completed. Policy stable: {policy_stable}."
        )
        return policy_stable

    def train(self):
        """정책 평가와 정책 개선을 반복하여 최적 정책 학습"""
        iteration = 0
        while True:
            self.logger.info(f"Training Iteration {iteration} started.")
            self.policy_evaluation()
            if self.policy_improvement():
                self.logger.info(f"Training Iteration {iteration}: Policy stabilized.")
                break
            iteration += 1

        self.logger.info("Training completed.")

    def get_policy(self):
        return self.policy

    def get_value_function(self):
        return self.value_function
