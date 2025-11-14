import asyncio, math

# MAVSDK Offboard 타입 호환 레이어
try:
    # 신형/일부 빌드: yaw-rate 지원
    from mavsdk.offboard import VelocityNedYawRate as _VelType
    _USE_YAW_RATE = True
except ImportError:  # 구형: yaw(heading)만 지원
    from mavsdk.offboard import VelocityNedYaw as _VelType
    _USE_YAW_RATE = False

from mavsdk.offboard import OffboardError
from utils import enu_to_ned

class OffboardStreamer:
    def __init__(self, drone):
        self.drone = drone
        self.vx = self.vy = self.vz = 0.0
        self.yaw_rate = 0.0 # rad/s
        self._run = False
        self._yaw_deg = 0.0 # heading 유지 (구형 API용)

    def set(self, vx, vy, vz, yaw_rate):
        self.vx, slef.vy, self.vz, self.yaw_rate = float(vx), float(vy), float(vz), float(yaw_rate)
    
    async def run(self, offboard_hz: float):
        """Offbooard 루프 시작. offboard_hz 주기로 NED 속도 명령 송신."""
        self._run = True
        period = 1.0 / float(offboard_hz)

        # 초기 0 setpoint (PX4 요구사항)
        vx_n, vy_n, vz_n = enu_to_ned(0.0, 0.0, 0.0)
        if _USE_YAW_RATE:
            await self.drone.offboard.set_velocity_ned(_VelType(vx_n, vy_n, vz_n, 0.0))
        else:
            await self.drone.offboard.set_velocity_ned(_VelType(vx_n, vy_n, vz_n, self._yaw_deg))

        try:
            await self.drone.offboard.start()
            print("[Offboard] started")
        except OffboardError as e:
            print("[Offboard] start failed:", e)
            raise
        
        try:
            while self._run:
                vx_n, vy_n, vz_n = enu_to_ned(self.vx, self.vy, self.vz)

                if _USE_YAW_RATE:
                    # yaw_rate: rad/s -> deg/s
                    await self.drone.offboard.set_velocity_ned(
                        _VelType(vx_n, vy_n,  vz_n, math.degrees(self.yaw_rate))
                    )
                else:
                    # 구형 API: yaw(heading)만 있음 -> yaw_rate 적분
                    self._yaw_deg = (self._yaw_deg + math.degrees(self.yaw_rate) * period) % 360.0
                    await self.drone.offboard.set_velocity_ned(
                        _VelType(vx_n, vy_n, vz_n, self.yaw_deg)
                    )
                await asyncio.sleep(period)
        finally:
            try:
                await self.drone.offboard.stop()
                print("[Offboared] stopped")
            except Exception:
                pass
    def stop(self):
        self._run = False