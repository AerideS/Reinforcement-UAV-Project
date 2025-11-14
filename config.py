from typing import Tuple, Optional
from dataclasses import dataclass

@dataclass
class CFG:
    # Connection
    system_address: str = "udpin://:14540" # MAVSDK auto-backend

    # Rates
    offboard_hz: float = 30.0 # Offboard setpoint stream rate
    step_hz: float = 10.0 # RL step rate

    # Flight
    takeoff_alt: float = 1.0 # auto takeoff altitude [m]