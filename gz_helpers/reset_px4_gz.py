# gz_helpers/reset_px4_gz.py
import asyncio
import subprocess
import math

from mavsdk import System
from mavsdk.offboard import OffboardError

WORLD = "rf_arena"
MODEL = "x500_0"  # 실제 모델명 확인 필요 (x500일 수도 있음)

RESET_POSE = {
    "x": 0.0,
    "y": 0.0,
    "z": 1.0,
    "yaw_deg": 0.0
}


# -----------------------------------------------------
# Gazebo 모델 Pose + 속도 초기화
# -----------------------------------------------------
async def reset_model_state():
    print(f"[RESET] Resetting MODEL state ({MODEL}) in Gazebo")

    yaw_rad = math.radians(RESET_POSE["yaw_deg"])
    qz = math.sin(yaw_rad / 2)
    qw = math.cos(yaw_rad / 2)

    # Poz + Orientation
    pose_req = (
        f'name: "{MODEL}" '
        f'position: {{ x: {RESET_POSE["x"]}, y: {RESET_POSE["y"]}, z: {RESET_POSE["z"]} }} '
        f'orientation: {{ x: 0.0, y: 0.0, z: {qz}, w: {qw} }}'
    )

    cmd_pose = [
        "gz", "service",
        "-s", f"/world/{WORLD}/set_pose",
        "--reqtype", "gz.msgs.Pose",
        "--reptype", "gz.msgs.Boolean",
        "--timeout", "300",
        "--req", pose_req,
    ]

    try:
        subprocess.run(cmd_pose, check=True)
        print("[RESET] Pose reset OK")
    except Exception as e:
        print(f"[RESET] pose reset FAILED: {e}")

    # ➤ 속도/각속도 초기화 (WorldControl 메시지 사용)
    cmd_zero_vel = [
        "gz", "service",
        "-s", f"/world/{WORLD}/control",
        "--reqtype", "gz.msgs.WorldControl",
        "--reptype", "gz.msgs.Boolean",
        "--timeout", "3000",
        "--req",
        f'model {{"name": "{MODEL}" velocity: {{x: 0 y: 0 z: 0}} angular_velocity: {{x: 0 y: 0 z: 0}} }}'
    ]

    try:
        subprocess.run(cmd_zero_vel, check=True)
        print("[RESET] Velocity zeroed OK")
    except Exception as e:
        print(f"[RESET] Velocity reset FAILED: {e}")


# -----------------------------------------------------
# PX4 Health 확인
# -----------------------------------------------------
async def wait_px4_ready(drone: System, timeout: float = 8.0):
    print("[RESET] Waiting PX4 health OK")

    start = asyncio.get_event_loop().time()
    async for health in drone.telemetry.health():
        if health.is_local_position_ok and health.is_home_position_ok:
            print("[RESET] PX4 ready")
            return True
        if asyncio.get_event_loop().time() - start > timeout:
            print("[RESET] Health TIMEOUT")
            return False
        await asyncio.sleep(0.1)


# -----------------------------------------------------
# 최종 Reset Episode Logic
# -----------------------------------------------------
async def reset_episode(drone: System, takeoff_alt: float):
    print("\n[RESET] === Episode Reset Begin ===")

    # 1) Offboard 종료
    try:
        await drone.offboard.stop()
        print("[RESET] Offboard stopped")
    except Exception:
        print("[RESET] Offboard already inactive")

    # 2) Disarm
    try:
        await drone.action.disarm()
        print("[RESET] Disarmed")
    except Exception:
        print("[RESET] Disarm warning - but continuing")

    # 3) Gazebo에서만 모델 State Reset
    await reset_model_state()
    await asyncio.sleep(0.5)

    # 4) PX4 상태 확인
    ready = await wait_px4_ready(drone)
    if not ready:
        print("[RESET] ❌ Critical reset failure (Health not recovered)")
        return False

    # 5) Arm + Takeoff (재시도 포함)
    try:
        await drone.action.arm()
        await drone.action.takeoff()
        print("[RESET] ✅ Takeoff OK")
    except Exception as e:
        print(f"[RESET] ❌ Takeoff failed: {e}")
        return False

    await asyncio.sleep(2.0)
    print("[RESET] === Episode Reset Complete ===\n")
    return True
