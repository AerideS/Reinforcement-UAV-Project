import math

def enu_to_ned(vx: float, vy: float, vz:float):
    # ENU -> NED: [x, y, z]_NED = [y, x, -z]_ENU
    return (vy, vx, -vz)