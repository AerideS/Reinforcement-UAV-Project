#!/usr/bin/env python3
"""
manual_control_simple.py
- WASD / ZX / F 로 PX4 SITL 드론을 수동 제어
- Gazebo 충돌 감지는 일단 제외 (나중에 별도 스크립트로)
"""

import asyncio
import sys
import termios
import tty

from mavsdk import System
from mavsdk.offboard import VelocityNedYaw, OffboardError

CONTROL_SPEED = 1.5   # m/s
UPDATE_RATE = 20      # Hz → 0.05s


def getch():
    """터미널에서 키 하나 읽기 (blocking)"""
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
    return ch


async def manual_control():
    drone = System()
    await drone.connect(system_address="udpin://:14540")

    print("[INFO] Waiting for vehicle...")
    async for cs in drone.core.connection_state():
        if cs.is_connected:
            print("[OK] Connected")
            break

    print("[INFO] Waiting for local position...")
    async for h in drone.telemetry.health():
        if h.is_local_position_ok:
            print("[OK] Local position ready")
            break

    # Arm & takeoff
    print("[INFO] Arming")
    await drone.action.arm()
    await drone.action.set_takeoff_altitude(2.0)
    await drone.action.takeoff()
    await asyncio.sleep(3.0)

    # Start offboard
    print("[INFO] Starting offboard...")
    await drone.offboard.set_velocity_ned(VelocityNedYaw(0, 0, 0, 0))
    try:
        await drone.offboard.start()
    except OffboardError as e:
        print("[ERR] Offboard failed:", e)
        return

    print("\n=== Manual Control Started ===")
    print("W/S: forward/back   A/D: left/right")
    print("Z/X: down/up        F: stop        Q: quit")
    print("=========================================================\n")

    vx = 0.0  # east (x)
    vy = 0.0  # north (y)
    vz = 0.0  # down-positive (NED)
    yaw = 0.0

    loop_dt = 1.0 / UPDATE_RATE

    try:
        while True:
            print("Input: ", end="", flush=True)
            key = getch()
            print(key)

            if key in ('q', 'Q'):
                print("[INFO] Quit requested")
                break

            # movement mapping
            if key in ('w', 'W'):
                vy = CONTROL_SPEED    # north+
            elif key in ('s', 'S'):
                vy = -CONTROL_SPEED   # north-
            elif key in ('a', 'A'):
                vx = -CONTROL_SPEED   # east-
            elif key in ('d', 'D'):
                vx = CONTROL_SPEED    # east+
            elif key in ('f', 'F'):   # stop
                vx = 0.0; vy = 0.0; vz = 0.0
            elif key in ('z', 'Z'):
                vz = 1.0    # DOWN (NED +)
            elif key in ('x', 'X'):
                vz = -1.0   # UP   (NED -)

            # send velocity: (north, east, down, yaw_deg)
            try:
                await drone.offboard.set_velocity_ned(VelocityNedYaw(vy, vx, vz, yaw))
            except Exception as e:
                print("[WARN] set_velocity_ned failed:", e)

            await asyncio.sleep(loop_dt)

    finally:
        print("[INFO] Landing...")
        try:
            await drone.action.land()
        except Exception as e:
            print("[WARN] land failed:", e)

        await asyncio.sleep(3.0)

        try:
            await drone.offboard.stop()
        except Exception:
            pass

        print("[DONE] Manual control finished.")


if __name__ == "__main__":
    try:
        asyncio.run(manual_control())
    except KeyboardInterrupt:
        print("\n[QUIT]")
