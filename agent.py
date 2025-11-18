# agent.py
import random
from typing import Any, Dict


class Agent:
    """
    강화학습 에이전트 뼈대.
    - select_action: 관측을 받아서 행동을 선택
    - store_transition: 리플레이 버퍼에 저장
    - train_step: 한 번 학습 스텝 수행

    TODO: 여기에 DQN/DDQN/SAC 등 실제 네트워크/optimizer를 넣으면 된다.
    """

    def __init__(self):
        # TODO: Q-network, optimizer, replay buffer 등 초기화
        pass

    def select_action(self, obs: Dict[str, Any]) -> Dict[str, float]:
        """
        관측을 보고 행동을 선택.
        지금은 예시로 랜덤 속도 명령을 반환한다.
        """
        # vx, vy: -1.0 ~ 1.0 m/s 사이 랜덤
        vx = random.uniform(-0.5, 0.5)
        vy = random.uniform(-0.5, 0.5)
        vz = 0.0  # 고도는 유지한다고 가정
        yaw_deg = 0.0

        return {"vx": vx, "vy": vy, "vz": vz, "yaw_deg": yaw_deg}

    def store_transition(
        self,
        obs: Dict[str, Any],
        action: Dict[str, float],
        reward: float,
        next_obs: Dict[str, Any],
        done: bool,
    ):
        """
        (s, a, r, s', done) 저장
        TODO: 리플레이 버퍼에 추가
        """
        pass

    def train_step(self):
        """
        리플레이 버퍼에서 샘플을 뽑아서 한 번 학습.
        TODO: DQN/DDQN 업데이트 로직 구현
        """
        pass

    def save(self, path: str):
        """
        학습된 파라미터 저장.
        TODO: torch.save(...) 등으로 구현
        """
        print(f"[AGENT] (TODO) save model to {path}")

    def load(self, path: str):
        """
        파라미터 로드.
        """
        print(f"[AGENT] (TODO) load model from {path}")
