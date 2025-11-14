import asyncio, time
from mavsdk import System
from mavsdk.telemetry import FlightMode

# 호환 타입 임포트
try:
    from mavsdk.offboard import VelocityNedYawRate as _VelType
except ImportError:
    from mavsdk.offboard import VelocityNedYaw as _VelType

async def iter_one(stream): # 텔레메트리 스트림
    async for v in stream:
        return v
    return None

async def wait_until_ready(drone: System, timeout_s: float = 60.0):
    t0 = time.time()
    while True:
        h = await iter_one(drone.telemetry.health())
        if h and h.is_local_position_ok and h.is_global_position_ok and h.is_home_position_ok:
            print("[OK] PX4 Health+Home ready")
            break
        if time.time() - t0 > timeout_s:
            print("[WARN] Timeout waiting for PX4 readiness")
            break
        await asyncio.sleep(0.5)

async def ensure_armed_and_takeoff(drone: System, target_alt: float):
    await wait_until_ready(drone)

    # POSITION 모드로 전환 (오프모드 진입 전 안전)
    try:
        await drone.action.set_flight_mode(FlightMode.POSITION)
        print("[Mode] POSITION")
    except Exception as e:
        print("[WARN] set mode failed:", e)
    
    # 오프보드 dummy setpoint 1회 (PX4 요구)
    try:
        await drone.offboard.set_velocity_ned(_VelType(0, 0, 0, 0))
        print("[Init] sent dummy offboard setpoint")
    except Exception as e:
        print("[WARN] dummy setpoint failed:", e)
    
    # Arm 재시도
    for i in range(5):
        try:
            await drone.action.arm()
            print("[OK] Armed")
            break
        except Exception as e:
            print(f"[arm] retry {i+1}/5:", e)
            await asyncio.sleep(1.0)
    else:
        raise RuntimeError("Arming failed after retries")

    # 이륙
    try:
        await drone.action.set_takeoff_altitude(target_alt)
    except Exception:
        pass
    await drone.action.takeoff()

    # 고도 도달 간단 대기
    t0 = time.time()
    while True:
        pos = await iter_one(drone.telemetry.position())
        if pos and pos.relative_altitude_m >= target_alt * 0.9:
            print(f"[OK] Reached ~{target_alt} m")
            break
        if time.time() - t0 > 30:
            print("[WARN] takeoff altitude wait timeout")
            break
        await asyncio.sleep(0.2)