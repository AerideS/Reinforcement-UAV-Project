# main.py
import asyncio
import sys
from mavsdk import System

from config import CFG
from mavsdk_helpers.connection import ensure_armed_and_takeoff
from mavsdk_helpers.streamer import OffboardStreamer
from gz_helpers.collision_watch import watch_contacts
# reset_episode는 안 쓰고, 에피소드 끝나면 그냥 종료하는 구조로

MAX_STEPS = 2000  # 에피소드당 최대 스텝(필요에 맞게 조절)


async def run_episode() -> int:
    cfg = CFG()

    # 1. PX4 연결
    drone = System()
    print("[PX4] Connecting...")
    await drone.connect(system_address=cfg.system_address)

    async for cs in drone.core.connection_state():
        if cs.is_connected:
            print("[PX4] Connected.")
            break

    # 2. 이륙
    await ensure_armed_and_takeoff(drone, cfg.takeoff_alt)

    # 3. Offboard streamer 시작
    streamer = OffboardStreamer(drone)
    streamer_task = asyncio.create_task(streamer.run(cfg.offboard_hz))

    # 4. 충돌 감시 (Event 기반)
    collision_event = asyncio.Event()
    watcher_task = asyncio.create_task(watch_contacts(collision_event))

    print("\n====== EPISODE START ======\n")
    step = 0
    exit_code = 0

    try:
        while step < MAX_STEPS:
            step += 1

            # TODO: RL action / step
            await asyncio.sleep(0.02)

            if collision_event.is_set():
                print("[MAIN] Collision detected → EPISODE DONE")
                # 여기서 굳이 reset 안 하고, 그냥 에피소드 종료
                exit_code = 0  # 보통 '정상적으로 끝난 episode'로 기록
                break

        else:
            # MAX_STEPS 초과로 종료된 경우
            print("[MAIN] Max steps reached → EPISODE DONE")
            exit_code = 0  # 또는 1로 해서 타임리밋을 실패로 볼 수도 있음

    except Exception as e:
        print(f"[MAIN] Fatal error in episode: {e}")
        exit_code = 1

    finally:
        # 정리
        print("[MAIN] Stopping tasks...")
        watcher_task.cancel()
        streamer_task.cancel()
        await asyncio.gather(watcher_task, streamer_task, return_exceptions=True)

        # 착륙/Disarm은 선택 사항 (어차피 프로세스가 끝나면 SITL도 죽을 예정이면 생략 가능)
        try:
            await drone.action.land()
        except Exception:
            pass

        print("[MAIN] Episode finished. Exit code:", exit_code)

    return exit_code


if __name__ == "__main__":
    code = asyncio.run(run_episode())
    sys.exit(code)
