import asyncio
import time

       
async def moveMountInElevationSteps(mount, target_el, azimuth=0, step_size=0.25, time_sleep=1):
    """Move the mount from the current elevation angle to the target elevation angle 
    in steps to avoid any issues whe M1M3 and/or M2 are running with the LUT using the 
    Mount instead of the inclinometer.
    
    This function will actually calculate the number of steps using the ceiling
    in order to make sure that we move carefully. 
    
    Parameters
    ----------
    mtmount : Remote 
        Mount CSC remote. 
    target_el : float
        Target elevation angle in degrees
    azimuth : float
        Azimuth angle in degres (default)
    step_size : float
        Step elevation size in degrees (default: 0.25)
    time_sleep : float
        Sleep time between movements (default: 1)
        
    Returns
    -------
    azimuth : float
        Current azimuth
    elevation : float
        Current elevation
    """
    current_el = mtmount.tel_elevation.get().actualPosition
    n_steps = int(np.ceil(np.abs(current_el - target_el) / step_size))

    for el in np.linspace(current_el, target_el, n_steps):
        print(f"Moving elevation to {el:.2f} deg")
        await mtmount.cmd_moveToTarget.set_start(azimuth=azimuth, elevation=el)
        time.sleep(time_sleep)
        
    return azimuth, el