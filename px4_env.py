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


def kill_stale_sitl():
    """
    혹시 이전 실행에서 남은 px4/gz 관련 프로세스가 있으면 정리.
    - train_rl.py 를 새로 돌리기 직전
    - _stop_sitl() 의 fallback 용도로 사용
    """
    patterns = [
        "make px4_sitl gz_x500",
        "PX4_SIM_MODEL=gz_x500",          # px4 실행 환경
        "bin/px4",                        # px4 binary
        "gz sim --verbose=1 -r -s",       # 네가 쓰는 gz sim 옵션
    ]
    for pat in patterns:
        try:
            subprocess.run(["pkill", "-f", pat], check=False)
        except Exception:
            pass


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

        # ---- 목표 위치 (NED 기준, 홈 포지션 기준) ----
        # 예시: 북쪽(+)으로 10m, 동쪽 0m, 고도는 이륙 고도 근처 유지
        self.goal_north = 40.0
        self.goal_east = 40.0
        self.goal_down = -self.cfg.takeoff_alt  # NED 기준: down 이 + 라서 음수

        # 이전 스텝에서의 목표까지 거리 (진행도 보상용)
        self.prev_dist_to_goal: Optional[float] = None

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
        # 혹시 이전 실행에서 남은 SITL 이 있으면 한 번 정리
        kill_stale_sitl()

        if self.sitl_proc is not None and self.sitl_proc.poll() is None:
            # 이미 떠 있으면 그대로 사용
            print("[ENV] SITL already running, reuse it.")
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
        이후 혹시 남았을 수 있는 프로세스는 kill_stale_sitl()로 정리.
        """
        if self.sitl_proc is None:
            # 그래도 혹시 모를 잔여 프로세스 정리
            kill_stale_sitl()
            return

        if self.sitl_proc.poll() is not None:
            print("[ENV] SITL already exited.")
            self.sitl_proc = None
            kill_stale_sitl()
            return

        pid = self.sitl_proc.pid
        print(f"[ENV] Stopping PX4+Gazebo (SITL pid={pid})...")

        try:
            pgid = os.getpgid(pid)
        except ProcessLookupError:
            print("[ENV] Process group not found; SITL probably already gone.")
            self.sitl_proc = None
            kill_stale_sitl()
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
            # 마지막으로 한번 더 정리 (혹시 그룹 밖으로 분리된 gz sim 등)
            kill_stale_sitl()

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

        # 이륙 직후 내부 상태가 안정되도록 잠깐 대기
        await asyncio.sleep(1.5)

        # 4) Offboard 시작 (재시도 로직 포함)
        await self._start_offboard()

        # 5) 충돌 이벤트 초기화 + watcher 시작
        self.collision_event.clear()
        self.collision_task = asyncio.create_task(
            watch_contacts(self.collision_event)
        )

        self.step_count = 0

        # 6) 초기 관측(obs)
        obs = await self._get_obs()
        # 초기 목표 거리 저장 (진행도 보상용)
        self.prev_dist_to_goal = obs.get("dist_to_goal", None)

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
        dt = 1.0 / self.cfg.step_hz
        await asyncio.sleep(dt)

        # 3) 관측
        obs = await self._get_obs()

        # 4) 보상 계산
        reward = self._compute_reward(obs, action)

        # 5) 종료 조건
        done = False
        info: Dict[str, Any] = {}

        # (1) 충돌
        if self.collision_event.is_set():
            print("[ENV] Collision detected → done=True")
            done = True
            info["reason"] = "collision"

        # (2) 목표 도달
        if obs.get("dist_to_goal", 1e9) < 1.0 and not done:
            print("[ENV] Goal reached → done=True")
            done = True
            info["reason"] = "goal"

        # (3) 타임아웃
        if self.step_count >= self.max_steps and not done:
            print("[ENV] Max steps reached → done=True")
            done = True
            info["reason"] = "timeout"

        # 다음 스텝에서 사용할 이전 거리 갱신
        self.prev_dist_to_goal = obs.get("dist_to_goal", self.prev_dist_to_goal)

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

        # 에피소드마다 SITL까지 완전히 종료
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
        (PX4 요구사항: 먼저 set_velocity_ned를 여러 번 보내고 start)

        가끔 NO_SETPOINT_SET 뜨는 타이밍 이슈가 있어서
        몇 번 재시도하도록 구현.
        """
        if self.drone is None:
            raise RuntimeError("Drone is not connected")

        print("[ENV] Starting Offboard...")
        zero = VelocityNedYaw(0.0, 0.0, 0.0, 0.0)

        dt = 1.0 / self.cfg.offboard_hz
        max_attempts = 3

        for attempt in range(1, max_attempts + 1):
            print(f"[ENV] Offboard start attempt {attempt}/{max_attempts}")

            # 1) dummy setpoint 여러 번 쏘기
            for i in range(10):
                try:
                    await self.drone.offboard.set_velocity_ned(zero)
                except OffboardError as e:
                    # 여기서도 가끔 에러 날 수 있으니 로그만 찍고 계속 진행
                    print(f"[ENV] set_velocity_ned failed (pre-start, i={i}): {e._result.result}")
                await asyncio.sleep(dt)

            # 2) start 시도
            try:
                await self.drone.offboard.start()
                self._offboard_running = True
                print("[ENV] Offboard started.")
                return  # 성공하면 함수 종료
            except OffboardError as e:
                print(f"[ENV] Offboard start failed (attempt {attempt}): {e._result.result}")
                self._offboard_running = False

                # 마지막 시도가 아니면 잠깐 쉬고 재시도
                if attempt < max_attempts:
                    await asyncio.sleep(1.0)
                    continue
                else:
                    # 마지막 시도까지 실패하면 예외 그대로 올리기
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
    # 6. 관측 / 보상
    # -------------------------------------------------
    async def _get_obs(self) -> Dict[str, Any]:
        """
        PX4 telemetry에서 관측 벡터 구성.
        - position_velocity_ned() 를 이용해 NED 위치/속도 읽어서 obs 생성.
        """
        if self.drone is None:
            return {}

        # 최신 값 한 번만 읽기
        async for pvned in self.drone.telemetry.position_velocity_ned():
            pos = pvned.position
            vel = pvned.velocity
            break

        north = float(pos.north_m)
        east = float(pos.east_m)
        down = float(pos.down_m)

        vn = float(vel.north_m_s)
        ve = float(vel.east_m_s)
        vd = float(vel.down_m_s)

        dist = self._goal_distance(north, east)

        obs = {
            "north": north,
            "east": east,
            "down": down,
            "vel_north": vn,
            "vel_east": ve,
            "vel_down": vd,
            "goal_north": self.goal_north,
            "goal_east": self.goal_east,
            "goal_down": self.goal_down,
            "dist_to_goal": dist,
        }
        return obs

    def _goal_distance(self, north: float, east: float) -> float:
        """
        수평면에서 목표까지 거리 [m]
        """
        return math.sqrt(
            (north - self.goal_north) ** 2 + (east - self.goal_east) ** 2
        )

    def _compute_reward(self, obs: Dict[str, Any], action: Dict[str, float]) -> float:
        """
        보상 함수.
        - 목표에 가까워질수록 +보상
        - step마다 약간의 시간 패널티
        - 속도 제곱 패널티 (너무 큰 속도 방지)
        - 목표 근접 시 큰 보상
        - 충돌 시 큰 패널티
        """
        dist = float(obs.get("dist_to_goal", 0.0))

        if self.prev_dist_to_goal is None:
            self.prev_dist_to_goal = dist

        # 1) 목표에 가까워진 정도 (이전 거리 - 현재 거리)
        progress = self.prev_dist_to_goal - dist
        reward = 2.0 * progress  # 가까워지면 양수, 멀어지면 음수

        # 2) 시간 패널티
        reward -= 0.01

        # 3) 속도 패널티
        vx = float(action.get("vx", 0.0))
        vy = float(action.get("vy", 0.0))
        vz = float(action.get("vz", 0.0))
        speed_sq = vx * vx + vy * vy + vz * vz
        reward -= 0.001 * speed_sq

        # 4) 목표 도달 보상 (1m 이내)
        if dist < 1.0:
            reward += 100.0

        # 5) 충돌 패널티
        if self.collision_event.is_set():
            reward -= 50.0

        return reward
