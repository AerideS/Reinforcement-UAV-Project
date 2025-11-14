# gz_collision_watch_cli.py
"""
충돌(contact) 토픽 모니터링 + x500_0 관련 충돌 발생 시:
  1) PX4 쪽 offboard 정지 / POSITION 모드 / 속도 0 / offboard 재시작 시도
  2) Gazebo에서 x500_0을 시작 위치로 텔레포트

※ RL 메인 루프와 동시에 MAVSDK에 붙어도 (멀티 GCS 구조라) 기본적으로는 동작함.
  다만 offboard start/stop을 여기서도 건드리므로,
  최종적으로는 이 reset 로직을 RL 프로세스 안으로 옮기는 걸 추천.
"""

import asyncio
import subprocess
import sys

from mavsdk import System
from mavsdk.offboard import VelocityNedYaw, OffboardError
from mavsdk.telemetry import FlightMode

# ---- 설정 ----
WORLD = "rf_arena"          # world 이름
MODEL = "x500_0"            # PX4 SITL 드론 모델 이름 (GUI 왼쪽 트리에서 확인)
CONTACT_TOPIC = (
    "/world/rf_arena/model/obs_a/link/link/sensor/contact_sensor/contact"
)
# 필요하면 obs_b, obs_c 토픽도 따로 늘리거나,
# /world/rf_arena/physics/contacts 같은 걸 쓰도록 확장 가능.

# 시작점 (ENU 좌표)
START_X = 0.0
START_Y = 0.0
START_Z = 3.0   # 약간 띄워서 스폰 (takeoff_alt 비슷하게)

SYSTEM_ADDRESS = "udpin://:14540"  # PX4 SITL 기본 주소


# --------------------------------------------------------
#  Gazebo: x500_0 텔레포트
# --------------------------------------------------------
def teleport_to_start():
    """gz service /world/<WORLD>/set_pose 로 MODEL을 시작점으로 텔레포트"""
    req = (
        f'name: "{MODEL}", '
        f'position: {{ x: {START_X}, y: {START_Y}, z: {START_Z} }}, '
        'orientation: { w: 1.0 }'
    )
    cmd = [
        "gz", "service",
        "-s", f"/world/{WORLD}/set_pose",
        "--reqtype", "gz.msgs.Pose",
        "--reptype", "gz.msgs.Boolean",
        "--timeout", "300",
        "--req", req,
    ]
    print("\n[RESET] Teleporting model to start pose...")
    print("       ", " ".join(cmd))
    try:
        out = subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT)
        print("[RESET] service response:", out.strip())
    except subprocess.CalledProcessError as e:
        print("[RESET][ERR] service failed:")
        print(e.output)


# --------------------------------------------------------
#  PX4: 속도 죽이고, POSITION 모드 → 텔포 → 0 속도 offboard 재시작
# --------------------------------------------------------
async def reset_px4_and_teleport(drone: System):
    """
    2번 방식 구현:
    1) offboard stop 시도
    2) flight mode = POSITION 시도
    3) Gazebo 텔레포트
    4) 잠깐 대기
    5) 0 속도로 offboard 재시작 시도
    """
    print("\n[PX4-RESET] Begin PX4 reset sequence")

    # 1) offboard stop
    try:
        await drone.offboard.stop()
        print("[PX4-RESET] offboard.stop() OK")
    except OffboardError as e:
        print(f"[PX4-RESET][WARN] offboard.stop failed: {e}")
    except Exception as e:
        print(f"[PX4-RESET][WARN] offboard.stop exception: {e}")

    # 2) POSITION 모드로 변경
    try:
        await drone.action.set_flight_mode(FlightMode.POSITION)
        print("[PX4-RESET] FlightMode.POSITION set")
    except Exception as e:
        print(f"[PX4-RESET][WARN] set_flight_mode failed: {e}")

    # 3) Gazebo 텔레포트 (동기 함수이므로 별도 thread에서 실행)
    await asyncio.to_thread(teleport_to_start)

    # 4) 잠깐 대기해서 PX4가 새 위치를 인지하고
    #    내부 제어가 속도를 줄일 시간을 줌
    await asyncio.sleep(1.0)

    # 5) 0 속도 setpoint + offboard 재시작
    try:
        await drone.offboard.set_velocity_ned(VelocityNedYaw(0.0, 0.0, 0.0, 0.0))
        try:
            await drone.offboard.start()
            print("[PX4-RESET] offboard.start() with zero velocity")
        except OffboardError as e:
            print(f"[PX4-RESET][WARN] offboard.start failed: {e}")
    except Exception as e:
        print(f"[PX4-RESET][WARN] set_velocity_ned(0) failed: {e}")

    print("[PX4-RESET] Done\n")


# --------------------------------------------------------
#  contact 토픽 모니터링 (비동기)
# --------------------------------------------------------
async def watch_contacts(drone: System):
    """
    gz topic -e -t CONTACT_TOPIC 을 subprocess로 실행해서
    stdout 라인을 비동기로 읽으면서, x500_0 관련 contact 발생 시 reset 호출
    """
    cmd = ["gz", "topic", "-e", "-t", CONTACT_TOPIC]
    print("[INFO] Running:", " ".join(cmd))
    print("[INFO] contact 토픽을 그대로 출력하면서, x500_0이 등장하면 PX4 reset + 텔레포트 합니다.\n")

    # ⚠ 여기서 text=True 를 쓰지 않고, 바이너리 스트림을 받고 decode 한다
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )

    try:
        while True:
            line_bytes = await proc.stdout.readline()
            if not line_bytes:
                break

            line = line_bytes.decode("utf-8", errors="ignore").strip()
            if not line:
                continue

            # 전체 라인 출력
            print("[CONTACT]", line)

            # x500_0 관련 충돌이면 reset 루틴 호출
            if f"{MODEL}::" in line:
                print(f"\n[HIT] {MODEL} 이(가) 관련된 contact 메시지 감지!")
                await reset_px4_and_teleport(drone)
                print("[INFO] 이후 메시지는 계속 모니터링합니다.\n")

    except asyncio.CancelledError:
        pass
    except KeyboardInterrupt:
        print("\n[INFO] user interrupted, stopping…")
    finally:
        proc.terminate()


# --------------------------------------------------------
#  메인: MAVSDK 연결 + contact watcher 실행
# --------------------------------------------------------
async def main():
    # MAVSDK System 준비
    drone = System()
    print(f"[PX4] Connecting to {SYSTEM_ADDRESS} ...")
    await drone.connect(system_address=SYSTEM_ADDRESS)

    # 연결 완료 대기
    async for cs in drone.core.connection_state():
        if cs.is_connected:
            print("[PX4] Connected to PX4")
            break

    # arm / takeoff 는 여기서 안 함.
    # RL 메인 코드가 이미 제어하고 있다는 가정.
    await watch_contacts(drone)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n[QUIT]")
        sys.exit(0)
