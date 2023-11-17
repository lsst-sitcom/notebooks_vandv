import asyncio
import time
import numpy as np
from scipy.signal import find_peaks
from astropy.time import Time


async def moveMountInElevationSteps(
    mount, target_el, azimuth=0, step_size=0.25, time_sleep=1
):
    """Move the mount from the current elevation angle to the target 
    elevation angle in steps to avoid any issues whe M1M3 and/or M2 are 
    running with the LUT using the mount instead of the inclinometer.

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
    current_el = mount.tel_elevation.get().actualPosition
    n_steps = int(np.ceil(np.abs(current_el - target_el) / step_size))

    for el in np.linspace(current_el, target_el, n_steps):
        print(f"Moving elevation to {el:.2f} deg")
        await mount.cmd_moveToTarget.set_start(azimuth=azimuth, elevation=el)
        time.sleep(time_sleep)

    if current_el == target_el:
        el = target_el

    return azimuth, el

def get_slew_pairs(starts, stops, return_unmatched=False):
    """
    Given vectors of start times and stop times take the longer vector
    and iterate over it. If that is `starts`, for each start time select all stop
    times that are > than the start time and < than the next start time.
    If multiple `stops` are detected select the minimum one. Also,
    the unmatched start and stop times can be returned with `return_unmatched`.
    Parameters
    ----------
    starts : float
        slew start times
    stops : float
        slew stop times
    return_unmatched : bool
        Whether to return stops or starts that cannot be associated
    
    Returns
    -------
    starts : float
        slew start times that match the returned stop times
    stops : float
        slew stop times that match the returned start times
    """
    new_starts = []
    new_stops = []
    unmatched_stops = []
    unmatched_starts = []
    
    if (len(starts) ==0) | len(stops) == 0:
        print("No slews identified")
        return [], []

    if len(stops) <= len(starts):
        for i in range(len(starts)):
            if i == len(starts) - 1:
                stop_sel = stops > starts[i]
            else:
                stop_sel = (stops > starts[i]) & (stops < starts[i + 1])
            if stop_sel.sum() == 1:
                new_stops.append(stops[stop_sel][0])
                new_starts.append(starts[i])
            if stop_sel.sum() > 1:
                new_stops.append(np.min(stops[stop_sel]))
                new_starts.append(starts[i])
                for j in stops[stop_sel]:
                    if j != np.min(stops[stop_sel]):
                        unmatched_stops.append(j)
            if stop_sel.sum() == 0:
                unmatched_starts.append(starts[i])

    if len(stops) > len(starts):
        for i in range(len(stops)):
            if i == 0:
                start_sel = (starts < stops[0]) & (starts > 0)
            else:
                start_sel = (starts < stops[i]) & (starts > stops[i - 1])
            if start_sel.sum() == 1:
                new_stops.append(stops[i])
                new_starts.append(starts[start_sel][0])
            if start_sel.sum() > 1:
                new_stops.append(stops[i])
                new_starts.append(np.max(starts[start_sel]))
                for j in starts[start_sel]:
                    if j != np.max(starts[start_sel]):
                        unmatched_starts.append(j)
            if start_sel.sum() == 0:
                unmatched_stops.append(stops[i])
    if (len(unmatched_starts) > 1) | (len(unmatched_stops) > 1):
        print("unmatched stops or starts found")
    if return_unmatched:
        return (
            np.array(new_starts),
            np.array(new_stops),
            unmatched_starts,
            unmatched_stops,
        )
    else:
        return np.array(new_starts), np.array(new_stops)


def get_starts_command_track_target(
    command_track_target_times, command_track_target_positions, shift=0.1
):
    """takes times and position (azimuth or elevation) 'lsst.sal.MTMount.command_trackTarget'
    and uses them to identify timestamps where the telesope has jumped more than 0.1 deg
    Parameters
    ----------
    command_track_target_times : float
        timestamps from command_track_target data_frame
    command_track_target_positions : float
        positions from command_track_target data_frame, should be azimuth or elevation
    shift : float
        shift threshold in degrees to identify slew starts
    
    Returns
    -------
    slew_start_times : float
        identified slew start times
    """
    lags = command_track_target_positions[1:] - command_track_target_positions[:-1]
    slew_indexes = np.where(abs(lags) > 0.1)[0] + 1  # lags is 1 shorter than positions
    slew_start_times = command_track_target_times[slewIndexes]
    return slew_start_times


def get_slews_edge_detection_telemetry(
    times, velocities, kernel_size=100, height=0.005, vel_thresh=0.05
):
    """
    given timestamps and velocity telemetry from mtMount azmuith or elevation.
    First, we smooth the velocity measurements and convert them to speed.
    Then, an edge detection kernel convolved with the speed data, `starts` are 
    identified by maxima and `stops` by minima values of the convolved data.
    
    Parameters
    ----------
    times : float
        timestamps from mtMount dataframe
    velocities: float
        actualVelocity measurements
    kernel size: int
        size of smoothing and edge detection kernel
    height: float
        minimum height of edge detection peaks (if spurious slews are 
        identified this should be raised)
    vel_thresh: float
        the minimum max velocity of a slew to flag
    
    Returns
    -------
    starts : float
        slew start times that match the returned stop times
    stops : float
        slew stop times that match the returned start times
    
    
    """
    smooth_kernel = np.ones(kernel_size) / kernel_size
    smoothed_speed = abs(np.convolve(velocities, smooth_kernel, mode="same"))

    edge_kernel = (
        np.concatenate(
            [1 * np.ones(int(kernel_size / 2)), -1 * np.ones(int(kernel_size / 2))]
        )
        / kernel_size
    )
    edge = np.convolve(smoothed_speed, edge_kernel, mode="same")

    starts = times[find_peaks(edge, height=height)[0]]
    stops = times[find_peaks(-1 * edge, height=height)[0]]
    maxv = []
    if (len(starts) ==0) | len(stops) == 0:
        print("No slews identified")
        return [], []
    starts, stops = get_slew_pairs(starts, stops)   
    
    for i, st in enumerate(starts):
        sel_vel = (times >= starts[i]) & (times <= stops[i])
        maxv.append(np.max(np.abs(smoothed_speed[sel_vel])))
    sel_slew = np.array(maxv) > vel_thresh

    starts = starts[sel_slew]
    stops = stops[sel_slew]

    for i, st in enumerate(starts):
        # adjust times to correspond with where the smoothed velocity has reached 0
        sel_starts = (times < st) & (smoothed_speed < 0.01)
        starts[i] = times[sel_starts][np.argmin(np.abs(times[sel_starts] - starts[i]))]

        sel_stops = (times > stops[i]) & (smoothed_speed < 0.01)
        stops[i] = times[sel_stops][np.argmin(np.abs(times[sel_stops] - stops[i]))]

    return starts, stops


def get_slews_command_track_target_and_telemetry(
    command_track_target_data_frame,
    mt_mount_data_frame,
    drive="azmiuth",
    kernel_size=100,
    height=0.005,
    vel_thresh=0.05,
):
    """
    given command track target data identify slew `starts` as large shifts in 
    the azimuth. Then, identify `stops` from the timestamps and velocity 
    telemetry from mtMount azmuith or elevation.
    
    Parameters
    ----------
    command_track_target_data_frame : data_frame
        data_frame from efd
    mt_mount_data_frame: data_frame
        data_frame from efd should be for "azmuith" or "elevation"
    drive: string
        search for slews using "azmuith" or "elevation" drives
    kernel size: int
        size of smoothing and edge detection kernel
    height: float
        minimum height of edge detection peaks (if spurious slews are 
        identified this should be raised)
    vel_thresh: float
        the minimum max velocity of a slew to flag
    
    Returns
    -------
    starts : float
        slew start times that match the returned stop times
    stops : float
        slew stop times that match the returned start times
    
    """
    
    if drive not in ["azimuth", "elevation"]:
        print("drive must be azimuth elevation")
        return [], []
    command_track_target_positions = command_track_target_data_frame[drive]
    
    # convert command_track_target_times from tai to utc
    command_track_target_times = Time(
        command_track_target_times, format="unix_tai"
    ).unix

    # get starts from command_track_target
    starts = get_starts_command_track_target(
        command_track_target_times, command_track_target_position
    )

    # get stops from telemetry
    mt_times = Time(mt_mount_data_frame["timestamp"], format="unix_tai").unix
    mt_velocities = mt_mount_data_frame["actualVelocity"]
    
    _, stops = get_slews_edge_detection_telemetry(
        mt_times,
        mt_velocities,
        kernel_size=kernel_size,
        height=height,
        vel_thresh=vel_thresh,
    )

    # make sure starts and stops are paired correctly
    starts, stops = get_slew_pairs(starts, stops)

    return starts, stops


def get_slew_from_mtmount(mt_mount_data_frame):
    """
    Givien a dataframe of mtMount telemetry return identified slews.
    Parameters
    ----------
    mt_mount_data_frame: data_frame
        data_frame from efd should contain "timestamp" and "actualVelocity"
        columns
    
    Returns
    -------
    starts : float
        slew start times that match the returned stop times
    stops : float
        slew stop times that match the returned start times
    """

    mt_times = Time(mt_mount_data_frame["timestamp"], format="unix_tai").unix
    mt_velocities = mt_mount_data_frame["actualVelocity"]
    return get_slews_edge_detection_telemetry(mt_times, mt_velocities)
