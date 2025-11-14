"""
Gazebo contact 토픽 모니터링 전용 모듈

기능:
   - gz topic -e -t <CONTACT_TOPIC>를 subprocess로 실행
   - stdout 라인을 읽으면서 MODEL 관련 contact 발생 시
     asyncio.Queue 로 "collision" 이벤트를 전달

* PX4 reset은 여기서 하지 않는다.
"""
import asyncio

# 설정
WORLD = "rf_arena"
MODEL = "x500_0"

# obs_a에 달린 contact sensor 토픽 (예시)
CONTACT_TOPIC = (
    "/world/rf_arena/model/obs_a/link/link/sensor/contact_sensor/contact"
)

# 필요하면 obs_b, obs_c 토픽을 추가해서 여러 개를 동시에 모니터링하도록 확장 가능

async def watch_contacts(event_queue: asyncio.Queue):
    """
    gz topic -e -t CONTACT_TOPIC을 subprocess로 실행해서
    stdout 라인을 비동기로 읽으면서, x500_0 관련 contact 발생 시
    event_queue에 collision 이벤트를 put
    
    event 예시(dict):
    {
        "type": "collision",
        "model": MODEL,
        "raw": <원본 라인 문자열>
    }
    """
    cmd = ["gz", "topic", "-e", "-t", CONTACT_TOPIC]
    print("[COLLISION-WATCH] Running:", " ".join(cmd))
    print("[COLLISION-WATCH] contact 토픽을 모니터링합니다. "
          "MODEL 관련 contact 발생 시 event_queue로 'collision' 이벤트를 보냅니다.\n")
    
    proc = await asyncio.create_subprocess_exec(  # subprocess.Popen()의 async 버전
        *cmd,  # gz topic -e -t ... 외부 명령어를 새 프로세스로 실행
        stdout=asyncio.subprocess.PIPE,          # 새로 띄운 프로세스의 표준 출력(stdout)을 파이프로 연결
        stderr=asyncio.subprocess.STDOUT,        # 에러 출력(stderr)도 stdout과 합쳐서 한 스트림으로 모음
    )

    try:
        while True:
            # IPC(PIPE)로 데이터 읽음
            line_bytes = await proc.stdout.readline()
            if not line_bytes:
                # 프로세스 종료
                print("[COLLISION-WATCH] gz topic process ended")
                break
            
            line = line_bytes.decode("utf-8", errors="ignore").strip()
            if not line:
                continue
            
            # 전체 라인 출력 (디버깅용)
            print("[CONTACT]", line)

            # x500_0 관련 충돌이면 이벤트 전달
            if f"{MODEL}::" in line:
                print(f"\n[HIT] {MODEL} 이(가) 관련된 contact 메시지 감지! "
                      "→ collision 이벤트 전송\n")
                event = {
                    "type": "collision",
                    "model": MODEL,
                    "raw": line,
                }
                await event_queue.put(event)
    
    except asyncio.CancelledError:
        print("[COLLISION-WATCH] Cancelled.")
    except KeyboardInterrupt:
        print("\n[COLLISION-WATCH] KeyboardInterrupt, stopping... ")
    finally:
        if proc.returncode is None:  # 아직 안 죽었으면
            proc.terminate()
            await proc.wait()
        print("[COLLISION-WATCH] exit.")
