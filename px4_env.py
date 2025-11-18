# px4_env.py
import asyncio
import subprocess
import signal
import os
import math
from typing import Any, Dict, Tuple, Optional

from mavsdk import System
from mavsdk.offboard import VelocityNedYaw, OffboardError

from config import CFG
from gz_helpers.collision_watch import watch_contacts
from mavsdk_helpers.connection import ensure_armed_and_takeoff


class Px4Env:
    """
    PX4 + Gazebo 기반 강화학습 환경.

    사용 예시 (train_rl.py 같은 데서):

        env = Px4Env()
        obs = await env.reset()
        done = False
        while not done:
            action = agent.select_action(obs)
            obs, reward, done, info = await env.step(action)
        await env.close_episode()
        await env.close()   # 전체 종료 시
    """

    def __init__(
        self,
        sitl_root: str = "/home/user/바탕화면/PX4-Autopilot",
        world_name: str = "rf_arena",
        model_name: str = "x500_0",
        max_steps: int = 1000,
    ):
        self.cfg = CFG()

        self.sitl_root = sitl_root
        self.world_name = world_name
        self.model_name = model_name

        self.max_steps = max_steps
        self.step_count = 0

        # PX4+Gazebo (SITL) 프로세스
        self.sitl_proc: Optional[subprocess.Popen] = None

        # MAVSDK 드론 핸들
        self.drone: Optional[System] = None

        # 충돌 감시
        self.collision_event: asyncio.Event = asyncio.Event()
        self.collision_task: Optional[asyncio.Task] = None

        # Offboard 상태
        self._offboard_running: bool = False

    # -------------------------------------------------
    # 0. SITL 프로세스 그룹 관리
    # -------------------------------------------------
    def _start_sitl(self):
        """
        PX4_GZ_WORLD, PX4_GZ_GUI 환경변수를 사용해
        make px4_sitl gz_x500 을 별도 프로세스 그룹으로 실행.
        """
        if self.sitl_proc is not None and self.sitl_proc.poll() is None:
            # 이미 떠 있으면 그대로 사용
            return

        print("[ENV] Starting PX4+Gazebo (SITL)...")
        cmd = f"PX4_GZ_GUI=0 PX4_GZ_WORLD={self.world_name} make px4_sitl gz_x500"

        # 새 세션(=새 프로세스 그룹)으로 실행
        self.sitl_proc = subprocess.Popen(
            ["bash", "-lc", cmd],
            cwd=self.sitl_root,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,  # <-- 프로세스 그룹 리더가 된다
        )
        print(f"[ENV] SITL started (pid={self.sitl_proc.pid})")

    def _stop_sitl(self):
        """
        우리가 띄운 PX4+Gazebo 프로세스 그룹 전체를 종료.
        SIGINT → (필요시) SIGKILL 순서로 보냄.
        """
        if self.sitl_proc is None:
            return

        if self.sitl_proc.poll() is not None:
            print("[ENV] SITL already exited.")
            self.sitl_proc = None
            return

        pid = self.sitl_proc.pid
        print(f"[ENV] Stopping PX4+Gazebo (SITL pid={pid})...")

        try:
            pgid = os.getpgid(pid)
        except ProcessLookupError:
            print("[ENV] Process group not found; SITL probably already gone.")
            self.sitl_proc = None
            return

        try:
            # 우선 SIGINT로 정상 종료 시도
            os.killpg(pgid, signal.SIGINT)
            try:
                self.sitl_proc.wait(timeout=5.0)
                print("[ENV] SITL group terminated with SIGINT.")
            except subprocess.TimeoutExpired:
                print("[ENV] SITL group did not exit, sending SIGKILL...")
                os.killpg(pgid, signal.SIGKILL)
                self.sitl_proc.wait(timeout=3.0)
                print("[ENV] SITL group killed with SIGKILL.")
        except ProcessLookupError:
            print("[ENV] SITL process group already gone.")
        finally:
            self.sitl_proc = None

    # -------------------------------------------------
    # 1. reset : 에피소드 시작
    # -------------------------------------------------
    async def reset(self) -> Dict[str, Any]:
        """
        에피소드 시작:
          1) SITL (PX4+Gazebo) 시작
          2) PX4 연결
          3) 이륙
          4) Offboard 시작
          5) 충돌 watcher 시작
          6) 초기 obs 반환
        """
        # 1) SITL 실행
        self._start_sitl()
        print("[ENV] Waiting SITL boot...")
        await asyncio.sleep(8.0)  # PX4+Gazebo 부팅 시간 (필요하면 조절)

        # 2) PX4 연결
        self.drone = System()
        print("[ENV] Connecting to PX4...")
        await self.drone.connect(system_address=self.cfg.system_address)

        async for cs in self.drone.core.connection_state():
            if cs.is_connected:
                print("[ENV] PX4 connected.")
                break

        # 3) Health & Home 확인 + 이륙
        await ensure_armed_and_takeoff(self.drone, self.cfg.takeoff_alt)

        # 4) Offboard 시작
        await self._start_offboard()

        # 5) 충돌 이벤트 초기화 + watcher 시작
        self.collision_event.clear()
        self.collision_task = asyncio.create_task(
            watch_contacts(self.collision_event)
        )

        self.step_count = 0

        # 6) 초기 관측(obs)
        obs = await self._get_obs()
        print("[ENV] Episode reset complete.")
        return obs

    # -------------------------------------------------
    # 2. step : action 적용
    # -------------------------------------------------
    async def step(
        self, action: Dict[str, float]
    ) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """
        action:
            예) {"vx": 0.5, "vy": 0.0, "vz": 0.0, "yaw_deg": 0.0}
            - NED 기준 속도 명령이라고 가정.

        반환:
            obs, reward, done, info
        """
        self.step_count += 1

        # 1) action → Offboard 속도 명령으로 적용
        await self._apply_action(action)

        # 2) dt 만큼 시간 경과
        await asyncio.sleep(0.02)  # 50 Hz 정도

        # 3) 관측
        obs = await self._get_obs()

        # 4) 보상 계산
        reward = self._compute_reward(obs, action)

        # 5) 종료 조건
        done = False
        info: Dict[str, Any] = {}

        if self.collision_event.is_set():
            print("[ENV] Collision detected → done=True")
            done = True
            info["reason"] = "collision"

        if self.step_count >= self.max_steps:
            print("[ENV] Max steps reached → done=True")
            done = True
            info["reason"] = "timeout"

        return obs, reward, done, info

    # -------------------------------------------------
    # 3. 에피소드 종료/정리
    # -------------------------------------------------
    async def close_episode(self):
        """
        에피소드 종료 시 정리:
          - 충돌 watcher 취소
          - Offboard stop
          - (선택) 착륙
          - SITL 종료 (여기에서 에피소드마다 완전히 내림)
        """
        print("[ENV] Closing episode...")

        # 충돌 watcher 정리
        if self.collision_task:
            self.collision_task.cancel()
            try:
                await self.collision_task
            except asyncio.CancelledError:
                pass
            self.collision_task = None

        # Offboard 정지
        await self._stop_offboard()

        # 착륙/Disarm (원하면 유지)
        if self.drone is not None:
            try:
                await self.drone.action.land()
                await asyncio.sleep(3.0)
            except Exception:
                pass

        self.drone = None

        # 에피소드마다 SITL까지 완전히 종료하고 싶으면 여기서 stop
        self._stop_sitl()

        print("[ENV] Episode closed.")

    async def close(self):
        """
        전체 학습 끝나서 환경 완전히 닫을 때.
        여러 번 호출되어도 안전하게 동작하도록 함.
        """
        try:
            await self.close_episode()
        except Exception:
            # 이미 에피소드가 닫힌 상태일 수도 있음 → 무시
            pass
        self._stop_sitl()
        print("[ENV] Env fully closed.")

    # -------------------------------------------------
    # 4. Offboard 시작/정지
    # -------------------------------------------------
    async def _start_offboard(self):
        """
        PX4 Offboard 모드 시작.
        (PX4 요구사항: 먼저 set_velocity_ned를 몇 번 보내고 start)
        """
        if self.drone is None:
            raise RuntimeError("Drone is not connected")

        print("[ENV] Starting Offboard...")
        zero = VelocityNedYaw(0.0, 0.0, 0.0, 0.0)

        # 초기 0 속도 명령 몇 번
        for _ in range(5):
            await self.drone.offboard.set_velocity_ned(zero)
            await asyncio.sleep(0.05)

        try:
            await self.drone.offboard.start()
            self._offboard_running = True
            print("[ENV] Offboard started.")
        except OffboardError as e:
            print(f"[ENV] Offboard start failed: {e._result.result}")
            self._offboard_running = False
            raise

    async def _stop_offboard(self):
        """
        Offboard 모드 정지.
        """
        if not self._offboard_running or self.drone is None:
            return
        print("[ENV] Stopping Offboard...")
        try:
            await self.drone.offboard.stop()
        except Exception:
            pass
        self._offboard_running = False

    # -------------------------------------------------
    # 5. action → 속도 명령 적용
    # -------------------------------------------------
    async def _apply_action(self, action: Dict[str, float]):
        """
        action dict를 NED 속도 명령으로 매핑해서 offboard에 전달.
        """
        if self.drone is None:
            return

        vx = float(action.get("vx", 0.0))
        vy = float(action.get("vy", 0.0))
        vz = float(action.get("vz", 0.0))
        yaw_deg = float(action.get("yaw_deg", 0.0))

        cmd = VelocityNedYaw(vx, vy, vz, yaw_deg)
        try:
            await self.drone.offboard.set_velocity_ned(cmd)
        except OffboardError as e:
            print(f"[ENV] set_velocity_ned failed: {e._result.result}")

    # -------------------------------------------------
    # 6. 관측 / 보상 (TODO: 네 연구에 맞게 구현)
    # -------------------------------------------------
    async def _get_obs(self) -> Dict[str, Any]:
        """
        PX4 telemetry에서 관측 벡터 구성.
        TODO: 네 목적에 맞게 실제 값들(position, velocity, attitude 등)을 넣으면 됨.
        """
        if self.drone is None:
            return {}

        # 예시: 실제 구현 시에는 telemetry.position_velocity_ned(),
        # attitude_euler(), distance sensor 등에서 값 읽어서 obs로 구성.
        # 지금은 템플릿이라 dummy 값만 리턴.
        return {
            "dummy": 0.0
        }

    def _compute_reward(self, obs: Dict[str, Any], action: Dict[str, float]) -> float:
        """
        보상 함수.
        TODO: 목표 거리, 속도 패널티, 충돌 패널티 등 네가 원하는 방식으로 구현.
        """
        # 템플릿: 일단 0 리워드
        return 0.0
