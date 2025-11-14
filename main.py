import asyncio, time, math
from typing import Tuple
import torch

from config import CFG
from mavsdk import System
from mavsdk_helpers.connection import ensure_armed_and_takeoff
from mavsdk_helpers.streamer import OffboardStreamer
#from mavsdk_helpers.env_px4 import PX4Env
#from agents.ddqn_agent import DDQN, Replay

from gz_helpers.collision_watch import watch_contacts

async def train():
    cfg = CFG()

    # MAVSDK 연결 및 이륙
    drone = System()
    await drone.connect(system_address=cfg.system_address)
    await ensure_armed_and_takeoff(drone, cfg.takeoff_alt)

    # Offboard 스트리머 시작
    streamer = OffboardStreamer(drone)
    _stream_task = asyncio.create_task(streamer.run(cfg.offboard_hz))

    # 충돌 이벤트 큐
    collision_queue: asyncio.Queue = asyncio.Queue()

    # 충돌 감시 테스크 시작 (같은 이벤트 루프에서 병렬 실행)
    _watch_task = asyncio.create_task(watch_contacts(collision_queue))

    # 일단 watcher가 끝날 때까지 대기
    await _watch_task

if __name__ == "__main__":
    try:
        asyncio.run(train())
    except KeyboardInterrupt:
        print("Interrupted.")