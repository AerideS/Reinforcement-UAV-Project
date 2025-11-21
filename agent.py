# agent.py
import random
from typing import Any, Dict, List
from collections import deque, namedtuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


Transition = namedtuple(
    "Transition", ("state", "action", "reward", "next_state", "done")
)


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def push(
        self,
        state_vec: np.ndarray,
        action_idx: int,
        reward: float,
        next_state_vec: np.ndarray,
        done: bool,
    ):
        self.buffer.append(
            Transition(state_vec, action_idx, reward, next_state_vec, done)
        )

    def sample(self, batch_size: int):
        transitions = random.sample(self.buffer, batch_size)
        batch = Transition(*zip(*transitions))
        return batch

    def __len__(self):
        return len(self.buffer)


class DQN(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
        )

    def forward(self, x):
        return self.net(x)


class Agent:
    """
    간단 DQN 에이전트.
    - 관측(obs dict)을 받아서 디스크리트 액션 선택
    - 장애물 충돌을 피하면서 목표에 도달하는 정책을 학습하도록 설계
    """

    def __init__(self):
        # ---- 하이퍼파라미터 ----
        self.gamma = 0.99
        self.lr = 1e-3
        self.batch_size = 64
        self.buffer_capacity = 50_000
        self.target_update_interval = 500  # step마다 타겟넷 동기화
        self.eps_start = 1.0
        self.eps_end = 0.1
        self.eps_decay = 10_000  # step 수 기준 선형 감소

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # 디스크리트 액션 정의 (NED 속도 명령 템플릿)
        v = 3  # m/s
        self.action_templates: List[Dict[str, float]] = [
            {"vx": v, "vy": 0.0, "vz": 0.0, "yaw_deg": 0.0},   # 0: 앞으로 (north+)
            {"vx": 0.0, "vy": v, "vz": 0.0, "yaw_deg": 0.0},   # 1: 오른쪽 (east+)
            {"vx": 0.0, "vy": -v, "vz": 0.0, "yaw_deg": 0.0},  # 2: 왼쪽 (east-)
            {"vx": -v, "vy": 0.0, "vz": 0.0, "yaw_deg": 0.0},  # 3: 뒤로 (north-)
            {"vx": 0.0, "vy": 0.0, "vz": 0.0, "yaw_deg": 0.0}, # 4: 제자리
        ]
        self.n_actions = len(self.action_templates)

        # 네트워크는 첫 관측을 보고 input_dim 결정 (lazy init)
        self.policy_net: DQN | None = None
        self.target_net: DQN | None = None
        self.optimizer: optim.Optimizer | None = None

        # obs dict 의 key 순서를 기억해서 feature 순서를 고정
        self.obs_keys: List[str] | None = None

        # 리플레이 버퍼
        self.memory = ReplayBuffer(self.buffer_capacity)

        # epsilon-greedy 및 target 업데이트를 위한 step 카운터
        self.total_steps = 0
        self.epsilon = self.eps_start

    # --------------------------
    # 내부 유틸
    # --------------------------
    def _ensure_network_initialized(self, obs: Dict[str, Any]):
        if self.policy_net is not None:
            return

        # obs dict 의 key 목록을 정렬해서 feature 순서 고정
        self.obs_keys = sorted(obs.keys())
        input_dim = len(self.obs_keys)

        self.policy_net = DQN(input_dim, self.n_actions).to(self.device)
        self.target_net = DQN(input_dim, self.n_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)

        print(f"[AGENT] Network initialized. input_dim={input_dim}, n_actions={self.n_actions}")
        print(f"[AGENT] obs_keys = {self.obs_keys}")

    def _obs_to_vec(self, obs: Dict[str, Any]) -> np.ndarray:
        assert self.obs_keys is not None
        return np.array(
            [float(obs[k]) for k in self.obs_keys],
            dtype=np.float32,
        )

    def _obs_to_tensor(self, obs: Dict[str, Any]) -> torch.Tensor:
        vec = self._obs_to_vec(obs)
        t = torch.from_numpy(vec).unsqueeze(0).to(self.device)  # (1, input_dim)
        return t

    def _update_epsilon(self):
        # total_steps 가 증가함에 따라 epsilon 선형 감소
        frac = min(1.0, self.total_steps / float(self.eps_decay))
        self.epsilon = self.eps_start + frac * (self.eps_end - self.eps_start)

    # --------------------------
    # 외부 인터페이스
    # --------------------------
    def select_action(self, obs: Dict[str, Any]) -> Dict[str, float]:
        """
        관측을 보고 행동을 선택.
        - epsilon-greedy 정책
        """
        # 네트워크 최초 초기화
        self._ensure_network_initialized(obs)

        self.total_steps += 1
        self._update_epsilon()

        # epsilon-greedy
        if random.random() < self.epsilon:
            action_idx = random.randrange(self.n_actions)
        else:
            assert self.policy_net is not None
            self.policy_net.eval()
            with torch.no_grad():
                state = self._obs_to_tensor(obs)
                q_values = self.policy_net(state)  # (1, n_actions)
                action_idx = int(torch.argmax(q_values, dim=1).item())

        # 액션 템플릿 + 내부용 인덱스(_idx) 붙여서 반환
        action = dict(self.action_templates[action_idx])
        action["_idx"] = action_idx
        return action

    def store_transition(
        self,
        obs: Dict[str, Any],
        action: Dict[str, float],
        reward: float,
        next_obs: Dict[str, Any],
        done: bool,
    ):
        """
        (s, a, r, s', done) 저장 → 리플레이 버퍼에 추가
        """
        if self.policy_net is None:
            # 아직 네트워크 초기화 전이면 아무 것도 안 함
            return

        action_idx = int(action.get("_idx", 0))
        state_vec = self._obs_to_vec(obs)
        next_state_vec = self._obs_to_vec(next_obs)

        self.memory.push(state_vec, action_idx, reward, next_state_vec, done)

    def train_step(self):
        """
        리플레이 버퍼에서 샘플을 뽑아서 한 번 학습.
        DQN 업데이트 수행.
        """
        if self.policy_net is None or self.target_net is None or self.optimizer is None:
            return

        if len(self.memory) < self.batch_size:
            return

        batch = self.memory.sample(self.batch_size)

        state_batch = torch.from_numpy(
            np.stack(batch.state)
        ).to(self.device)  # (B, input_dim)
        action_batch = torch.tensor(
            batch.action, dtype=torch.int64, device=self.device
        ).unsqueeze(1)  # (B, 1)
        reward_batch = torch.tensor(
            batch.reward, dtype=torch.float32, device=self.device
        ).unsqueeze(1)  # (B, 1)
        next_state_batch = torch.from_numpy(
            np.stack(batch.next_state)
        ).to(self.device)  # (B, input_dim)
        done_batch = torch.tensor(
            batch.done, dtype=torch.float32, device=self.device
        ).unsqueeze(1)  # (B, 1)

        # Q(s,a)
        self.policy_net.train()
        q_values = self.policy_net(state_batch).gather(1, action_batch)  # (B,1)

        # target = r + γ * (1 - done) * max_a' Q_target(s', a')
        with torch.no_grad():
            next_q_values = self.target_net(next_state_batch).max(1, keepdim=True)[0]
            targets = reward_batch + self.gamma * (1.0 - done_batch) * next_q_values

        loss = nn.functional.mse_loss(q_values, targets)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

        # 일정 스텝마다 타깃 네트워크 동기화
        if self.total_steps % self.target_update_interval == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
            print(f"[AGENT] Target network updated at step {self.total_steps}")

    def save(self, path: str):
        """
        학습된 파라미터 저장.
        """
        if self.policy_net is None or self.obs_keys is None:
            print("[AGENT] save: network not initialized yet.")
            return

        state = {
            "state_dict": self.policy_net.state_dict(),
            "obs_keys": self.obs_keys,
        }
        torch.save(state, path)
        print(f"[AGENT] Model saved to {path}")

    def load(self, path: str):
        """
        파라미터 로드.
        (주의: obs_keys 가 동일한 환경에서만 사용해야 함)
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.obs_keys = checkpoint["obs_keys"]
        input_dim = len(self.obs_keys)

        self.policy_net = DQN(input_dim, self.n_actions).to(self.device)
        self.target_net = DQN(input_dim, self.n_actions).to(self.device)
        self.policy_net.load_state_dict(checkpoint["state_dict"])
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)

        print(f"[AGENT] Model loaded from {path}")
        print(f"[AGENT] obs_keys = {self.obs_keys}")
