# gz_helpers/collision_watch.py
import asyncio
import time

WORLD = "rf_arena"
MODEL = "x500_0"  # 드론 모델 이름 (gz world 안 모델명과 동일해야 함)

# rf_arena.sdf 에 정의된 모든 contact 센서 토픽
CONTACT_TOPICS = [
    # --- ground ---
    "/world/rf_arena/model/ground_plane/link/link/sensor/ground_contact_sensor/contact",

    # --- walls ---
    "/world/rf_arena/model/wall_north/link/link/sensor/contact_sensor/contact",
    "/world/rf_arena/model/wall_south/link/link/sensor/contact_sensor/contact",
    "/world/rf_arena/model/wall_east/link/link/sensor/contact_sensor/contact",
    "/world/rf_arena/model/wall_west/link/link/sensor/contact_sensor/contact",

    # --- original obstacles ---
    "/world/rf_arena/model/obs_a/link/link/sensor/contact_sensor/contact",
    "/world/rf_arena/model/obs_b/link/link/sensor/contact_sensor/contact",
    "/world/rf_arena/model/obs_c/link/link/sensor/contact_sensor/contact",

    # --- additional obstacles ---
    "/world/rf_arena/model/obs_d/link/link/sensor/contact_sensor/contact",
    "/world/rf_arena/model/obs_e/link/link/sensor/contact_sensor/contact",
    "/world/rf_arena/model/obs_f/link/link/sensor/contact_sensor/contact",
    "/world/rf_arena/model/obs_g/link/link/sensor/contact_sensor/contact",
    "/world/rf_arena/model/obs_h/link/link/sensor/contact_sensor/contact",
]

GRACE_SECONDS = 3.0       # 시작 후 이 시간 동안은 충돌 무시 (이륙 중 ground contact 방지)
COOLDOWN_SECONDS = 2.0    # 한 번 이벤트 발생 후 최소 간격


async def _watch_one_topic(topic: str, collision_event: asyncio.Event):
    """
    단일 contact 센서 토픽을 감시하는 코루틴.
    topic: ground / walls / obs_* 센서 토픽 중 하나
    """
    cmd = ["gz", "topic", "-e", "-t", topic]
    print(f"[COLLISION] Watching topic: {topic}")
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )

    start_time = time.monotonic()

    inside_contact = False
    brace_depth = 0
    hit_model = False  # 이 contact 블록 안에 MODEL(x500_0)이 들어왔는지 여부

    last_emit_time = 0.0

    try:
        while True:
            line_bytes = await proc.stdout.readline()
            if not line_bytes:
                break

            line = line_bytes.decode("utf-8", errors="ignore").strip()
            if not line:
                continue

            # contact 블록 시작
            if line.startswith("contact {"):
                inside_contact = True
                brace_depth = 1
                hit_model = False
                continue

            if inside_contact:
                brace_depth += line.count("{")
                brace_depth -= line.count("}")

                # collision1:, collision2:, name: 중 하나에 모델 이름이 들어오면 hit
                if (
                    "collision1:" in line
                    or "collision2:" in line
                    or "name:" in line
                ):
                    if f"{MODEL}::" in line:
                        hit_model = True

                # contact 블록 끝
                if brace_depth <= 0:
                    inside_contact = False

                    if hit_model:
                        now = time.monotonic()
                        elapsed = now - start_time

                        if elapsed <= GRACE_SECONDS:
                            print(f"[COLLISION] ignored (grace) on {topic}")
                            continue

                        if now - last_emit_time < COOLDOWN_SECONDS:
                            print(f"[COLLISION] suppressed (cooldown) on {topic}")
                            continue

                        last_emit_time = now
                        print(f"[COLLISION] HIT detected on {topic} → event.set()")
                        collision_event.set()

    except asyncio.CancelledError:
        pass
    finally:
        if proc.returncode is None:
            proc.terminate()
            await proc.wait()
        print(f"[COLLISION] watcher for {topic} exit.")


async def watch_contacts(collision_event: asyncio.Event):
    """
    ground / walls / obs_* 센서 contact 토픽을 동시에 감시.
    어느 쪽에서든 x500_0과 contact가 잡히면 collision_event.set()
    """
    tasks = [
        asyncio.create_task(_watch_one_topic(topic, collision_event))
        for topic in CONTACT_TOPICS
    ]

    try:
        await asyncio.gather(*tasks)
    except asyncio.CancelledError:
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        print("[COLLISION] all watchers cancelled.")
