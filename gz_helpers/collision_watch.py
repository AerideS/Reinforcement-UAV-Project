# gz_helpers/collision_watch.py
import asyncio
import time

WORLD = "rf_arena"
MODEL = "x500_0"  # 드론 모델 이름 (필요하면 x500 등으로 수정)

# 실제로 존재하는 contact 센서 토픽 3개
CONTACT_TOPICS = [
    "/world/rf_arena/model/obs_a/link/link/sensor/contact_sensor/contact",
    "/world/rf_arena/model/obs_b/link/link/sensor/contact_sensor/contact",
    "/world/rf_arena/model/obs_c/link/link/sensor/contact_sensor/contact",
]

GRACE_SECONDS = 3.0       # 시작 후 이 시간 동안은 충돌 무시
COOLDOWN_SECONDS = 2.0    # 한 번 이벤트 발생 후 최소 간격


async def _watch_one_topic(topic: str, collision_event: asyncio.Event):
    """
    단일 contact 센서 토픽을 감시하는 코루틴.
    topic: obs_a / obs_b / obs_c 센서 토픽 중 하나
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

            # 디버깅 필요하면 잠깐 켜보기
            # print(f"[RAW {topic}] {line}")

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
    obs_a / obs_b / obs_c 센서 contact 토픽을 동시에 감시.
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
