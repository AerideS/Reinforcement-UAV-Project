#!/usr/bin/env python3
"""
gz_collision_watch_cli_v2.py

드론(x500_0)이 Gazebo에서 어떤 물체와든 충돌(contact)이 발생하면:

  1. PX4 offboard 정지 및 POSITION 모드로 전환.
  2. Gazebo에서 x500_0을 시작 위치로 텔레포트 (위치, 속도, 각속도 모두 초기화).
  3. 0 속도로 offboard 재시작.

- PX4 SITL: udpin://:14540
- World   : rf_arena
- Model   : x500_0
"""

import asyncio
import subprocess
import sys

from mavsdk import System
from mavsdk.offboard import VelocityNedYaw, OffboardError
from mavsdk.telemetry import FlightMode

# ==============================
# 환경 설정 (네 현재 환경에 맞게 세팅)
# ==============================
WORLD = "rf_arena"
MODEL = "x500_0"  # PX4 SITL 드론 모델 이름 (gz GUI 좌측 트리에서 확인)
# Gazebo contact 메시지에서 보통 "rf_arena::x500_0::base_link::collision" 이런 식으로 나옴
MODEL_COLLISION_PREFIX = f"{WORLD}::{MODEL}::"

# 월드 전체 충돌 토픽 (모든 충돌 이벤트가 여기로 나옴)
CONTACT_TOPIC = f"/world/{WORLD}/physics/contacts"

# 시작점 (ENU 좌표)
START_X = 0.0
START_Y = 0.0
START_Z = 3.0

SYSTEM_ADDRESS = "udpin://:14540"

# x500_0 메인 링크 이름 추정 (일반적으로 base_link)
# set_link_twist 서비스에서 이 이름으로 링크 속도를 0으로 초기화
MODEL_MAIN_LINK = f"{MODEL}::base_link"


# --------------------------------------------------------
#  Gazebo service helper
# --------------------------------------------------------
def call_gz_service(service_name: str, req_type: str, rep_type: str, req_data: str):
    """
    일반적인 gz service 호출 헬퍼
    """
    cmd = [
        "gz", "service",
        "-s", service_name,
        "--reqtype", req_type,
        "--reptype", rep_type,
        "--timeout", "300",
        "--req", req_data,
    ]
    print(f"[GZ] call: {' '.join(cmd)}")
    try:
        out = subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT)
        print("[GZ] response:", out.strip())
        return out.strip()
    except subprocess.CalledProcessError as e:
        print(f"[GZ][ERR] service {service_name} failed:")
        print(e.output)
        return None


def teleport_to_start():
    """
    Gazebo에서 MODEL을 시작 위치로 텔레포트하고,
    메인 링크 속도/각속도를 0으로 초기화.
    """
    print("\n[RESET] Teleporting model and resetting velocity...")

    # 1) Pose 설정
    pose_req = (
        f'name: "{MODEL}", '
        f'position: {{ x: {START_X}, y: {START_Y}, z: {START_Z} }}, '
        'orientation: { w: 1.0 }'
    )
    call_gz_service(
        f"/world/{WORLD}/set_pose",
        "gz.msgs.Pose",
        "gz.msgs.Boolean",
        pose_req,
    )

    # 2) Twist 설정 (linear, angular 모두 0)
    twist_req = (
        f'entity_name: "{MODEL_MAIN_LINK}", '
        'linear: { x: 0.0, y: 0.0, z: 0.0 }, '
        'angular: { x: 0.0, y: 0.0, z: 0.0 }'
    )
    call_gz_service(
        f"/world/{WORLD}/set_link_twist",
        "gz.msgs.Twist",
        "gz.msgs.Boolean",
        twist_req,
    )

    print("[RESET] Teleport & twist reset complete.\n")


# --------------------------------------------------------
#  PX4 reset sequence
# --------------------------------------------------------
async def reset_px4_and_teleport(drone: System):
    """
    1) offboard stop
    2) POSITION 모드
    3) Gazebo 텔포 + 속도 초기화
    4) 잠시 대기
    5) 0 속도로 offboard 재시작
    """
    print("\n[PX4-RESET] Begin PX4 reset sequence")

    # 1) Offboard stop
    try:
        await drone.offboard.stop()
        print("[PX4-RESET] offboard.stop() OK")
    except OffboardError as e:
        print(f"[PX4-RESET][WARN] offboard.stop failed: {e}")
    except Exception as e:
        print(f"[PX4-RESET][WARN] offboard.stop exception: {e}")

    # 2) POSITION 모드
    try:
        await drone.action.set_flight_mode(FlightMode.POSITION)
        print("[PX4-RESET] FlightMode.POSITION set")
    except Exception as e:
        print(f"[PX4-RESET][WARN] set_flight_mode failed: {e}")

    # 3) Gazebo 텔레포트 (동기 → 스레드로 넘겨 비동기처럼)
    await asyncio.to_thread(teleport_to_start)

    # 4) 약간 대기 (물리엔진 & PX4 안정화)
    await asyncio.sleep(2.0)

    # 5) 0 속도로 offboard 재시작
    try:
        await drone.offboard.set_velocity_ned(VelocityNedYaw(0.0, 0.0, 0.0, 0.0))
        try:
            await drone.offboard.start()
            print("[PX4-RESET] offboard.start() with zero velocity OK")
        except OffboardError as e:
            print(f"[PX4-RESET][WARN] offboard.start failed: {e}")
    except Exception as e:
        print(f"[PX4-RESET][WARN] set_velocity_ned(0) failed: {e}")

    print("[PX4-RESET] Done\n")


# --------------------------------------------------------
#  Contact topic watcher (async)
# --------------------------------------------------------
async def watch_contacts(drone: System):
    """
    gz topic -e -t CONTACT_TOPIC 을 subprocess로 실행해서
    stdout 라인을 비동기로 읽으면서,
    contact 메시지에 'rf_arena::x500_0::' 가 나오면 reset 시퀀스 수행.
    """
    cmd = ["gz", "topic", "-e", "-t", CONTACT_TOPIC]
    print("[INFO] Running:", " ".join(cmd))
    print("[INFO] contact 토픽 모니터링 시작. 드론 관련 충돌 시 PX4 reset + 텔레포트.\n")

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )

    try:
        while True:
            line_bytes = await proc.stdout.readline()
            if not line_bytes:
                # 프로세스 종료
                break

            line = line_bytes.decode("utf-8", errors="ignore").strip()
            if not line:
                continue

            # 디버깅용 전체 출력 (원하면 주석 풀기)
            # print("[CONTACT]", line)

            # contact 메시지에 'rf_arena::x500_0::' 가 포함되어 있으면
            if MODEL_COLLISION_PREFIX in line:
                print("\n[HIT] 드론 모델이 충돌에 관여한 contact 감지!")
                print("[CONTACT]", line)
                await reset_px4_and_teleport(drone)
                print("[INFO] 모니터링 계속...\n")

                # 너무 많은 이벤트가 연속으로 쏟아지는 것 방지
                await asyncio.sleep(0.5)

    except asyncio.CancelledError:
        pass
    except KeyboardInterrupt:
        print("\n[INFO] user interrupted, stopping watcher…")
    finally:
        if proc.returncode is None:
            proc.terminate()


# --------------------------------------------------------
#  main
# --------------------------------------------------------
async def main():
    drone = System()
    print(f"[PX4] Connecting to {SYSTEM_ADDRESS} ...")
    await drone.connect(system_address=SYSTEM_ADDRESS)

    # MAVSDK 연결 완료 대기
    async for cs in drone.core.connection_state():
        if cs.is_connected:
            print("[PX4] Connected to PX4")
            break

    # 연결만 확인하고, 나머지 제어(arm, takeoff, offboard start 등)는
    # 별도 스크립트/프로세스에서 하고 있다고 가정.
    # 여기서는 contact만 감시.
    await watch_contacts(drone)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n[QUIT] User interrupted main program.")
        sys.exit(0)
