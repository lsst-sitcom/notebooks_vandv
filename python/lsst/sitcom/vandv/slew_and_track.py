import asyncio
import logging

import numpy as np

from astropy.time import Time

from . import efd

__all__ = [
    "azel_grid_by_time",
    "generate_azel_sequence",
    "random_walk_azel_by_time",
    "take_images_for_time",
    "take_images_in_sync",
    "take_images_in_sync_for_time",
]


def azel_grid_by_time(total_time, _az_grid, _el_grid, logger=None):
    """
    Generate Az/El coordinates for a a long time so we can slew and track
    to these targets using a predefined az_grid and el_grid.

    Parameters
    ----------
    total_time : `float`
        Total observation time in seconds.
    _az_grid : `list` of `float`
        Azimuth coordinates to slew and track.
    _el_grid : `list` of `float`
        Elevation coordinates to slew and track.
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    step = 0
    timer_task = asyncio.create_task(asyncio.sleep(total_time))
    logger.info(
        f"{'Time':25s}{'Steps':>10s}{'Old Az':>10s}{'New Az':>10s}{'Old El':>10s}{'New El':>10s}"
    )

    generator = generate_azel_sequence(_az_grid, _el_grid)
    old_az, old_el = None, None

    while not timer_task.done():
        new_az, new_el = next(generator)

        if new_az == old_az and new_el == old_el:
            continue

        yield new_az, new_el

        t = Time.now().to_value("isot")

        if old_az and old_el:
            logger.info(
                f"{t:25s}{step:10d}{old_az:10.2f}{new_az:10.2f}{old_el:10.2f}{new_el:10.2f}"
            )

        old_az, old_el = new_az, new_el
        step += 1


def generate_azel_sequence(az_seq, el_seq, el_limit=90):
    """A generator that cicles through the input azimuth and elevation sequences
    forward and backwards.

    Parameters
    ----------
    az_seq : `list` [`float`]
        A sequence of azimuth values to cicle through
    el_seq : `list` [`float`]
        A sequence of elevation values to cicle through
    el_limit : `float`
        Cut off limit angle in elevation to skip points when going down.
    Yields
    ------
    `list`
        Values from the sequence.
    Notes
    -----
    This generator is designed to generate sequence of values cicling through
    the input forward and backwards. It will also reverse the list when moving
    backwards.
    Use it as follows:
    >>> az_seq = [0, 180]
    >>> el_seq = [15, 45]
    >>> seq_gen = generate_azel_sequence(az_seq, el_seq)
    >>> next(seq_gen)
    [0, 15]
    >>> next(seq_gen)
    [0, 45]
    >>> next(seq_gen)
    [180, 45]
    >>> next(seq_gen)
    [180, 15]
    >>> next(seq_gen)
    [0, 15]
    """
    i = 1
    for az in az_seq:
        for el in el_seq[::i]:
            if el > el_limit and i == -1:
                continue
            else:
                yield (az, el)
        i *= -1

        
def get_slew_icrs_target(start, end):
    """Returns a Pandas DataFrame containing the target Ra/Dec and the timestamps 
    that correspond to the beginning of a new slew. 
    
    Every time that we run a `mtcs.slew_icrs` command, it sends a 
    `cmd_raDecTarget` command to the `mtptg` component. We used it as a mark 
    that defines the start of a slew. 
    
    Parameters
    ----------
    start : `astropy.time.Time`
        Start time of the time range.
    end : `astropy.time.Time` or `astropy.time.TimeDelta`
        End time of the range either as an absolute time or
        a time offset from the start time.
    Returns
    -------
    
    """
    

def random_walk_azel_by_time(
    total_time,
    mtmount,
    radius=3.5,
    min_az=-200.0,
    max_az=+200,
    min_el=30,
    max_el=80,
    logger=None,
    big_slew_prob=0.05,
    big_slew_radius=7.0,
):
    """
    Generate Az/El coordinates for a a long time so we can slew and track
    to these targets.

    Parameters
    ----------
    total_time : `float`
        Total observation time in seconds.
    mtmount : `salobj.Remote`
        An instance of MTMount used to grab telemetry.
    radius : `float`, default 3.5
        Radius of the circle where the next coordinate will fall.
    min_az :  `float`, default -200
        Minimum accepted azimuth in deg.
    max_az :  `float`, default +200
        Maximum accepted azimuth in deg.
    min_el : `float`, default 30
        Minimum accepted elevation in deg.
    max_el : `float`, default 80
        Maximum accepted elevation in deb.
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    step = 0
    timer_task = asyncio.create_task(asyncio.sleep(total_time))
    logger.info(
        f"{'Time':25s}{'Steps':>10s}{'Old Az':>10s}{'New Az':>10s}{'Old El':>10s}{'New El':>10s}{'Offset':>10s}"
    )

    n_points = 10
    current_az = np.median(
        [mtmount.tel_azimuth.get().actualPosition for i in range(n_points)]
    )
    current_el = np.median(
        [mtmount.tel_elevation.get().actualPosition for i in range(n_points)]
    )

    while not timer_task.done():
        current_radius = (
            big_slew_radius if np.random.rand() <= big_slew_prob else radius
        )

        angle = 2 * np.pi * np.random.rand()

        offset_az = current_radius * np.cos(angle)
        new_az = current_az + offset_az

        offset_el = current_radius * np.sin(angle)
        new_el = current_el + offset_el

        if new_az <= min_az or new_az >= max_az:
            new_az = current_az - offset_az

        if new_el <= min_el or new_el >= max_el:
            new_el = current_el - offset_el

        offset = np.sqrt((current_az - new_az) ** 2 + (current_el - new_el) ** 2)

        t = Time.now().to_value("isot")
        logger.info(
            f"{t:25s}{step:10d}{current_az:10.2f}{new_az:10.2f}{current_el:10.2f}{new_el:10.2f}{offset:10.2f}"
        )

        yield new_az, new_el
        step += 1
        current_az, current_el = new_az, new_el


async def take_images_for_time(cam, exptime, reason, tracktime):
    """Takes images while tracking for some time. (not in sync)

    Parameters
    ----------
    cam : `lsst.ts.observatory.control.base_camera.BaseCamera`
        Contains a camera instance.
    exptime : `float`
        The exposure time.
    reason : `str`
        Reason passed to the `take_object` command.
    tracktime : `float`
        How long will we be tracking?

    Returns
    -------
    int : number of images obtained.
    """
    reason = reason.replace(" ", "_")
    timer_task = asyncio.create_task(asyncio.sleep(tracktime - exptime))
    n_images = 0

    while not timer_task.done():
        await cam.take_object(exptime, reason=reason)
        await asyncio.sleep(0.5)
        n_images += 1

    return n_images


async def take_images_in_sync(
    _camera_list, _exposure_times, _number_of_exposures, _reason, total_time
):
    """
    Take images in sync, which means keeping the images ID the same.
    This will increase overhead on the camera with shorter exposure time.

    Parameters
    ----------
    _camera_list : list of `GenericCamera`
        A list containing the `GenericCamera` for each Camera.
    _exposure_times : list of float
        A list containing the exposure time used on each camera.
    _reason : str
        Reason that goes to the metadata in each image.
    _number_of_exposures : float
        Total number of exposures for each camera.
    total_time : float
        Minimum time we should spend taking images (to keep tracking in a fixed position).
    """
    assert len(_camera_list) == len(_exposure_times)

    wait_time = asyncio.create_task(asyncio.sleep(total_time - max(exptimes)))

    for n in range(_number_of_exposures):
        tasks = [
            asyncio.create_task(cam.take_object(exptime, reason=_reason))
            for (cam, exptime) in zip(_camera_list, _exposure_times)
        ]

        # Wait until all the tasks are complete
        await asyncio.gather(*tasks)

    await wait_time
    

async def take_images_in_sync_for_nexp(cams, exptimes, reason, nexps, sleep=0.5):
    """Take images in sync while tracking until you get `nexps` exposures.
    
    Parameters
    ----------
    cams : list of `lsst.ts.observatory.control.base_camera.BaseCamera`
        A list containing a camera instance.
    exptimes : list of `float`
        A list of exposure times.
    reason : `str`
        Reason passed to the `take_object` command.
    tracktime : `float`
        How long will we be tracking?
    sleep : `float`
        Sleep time in seconds to compensate for readout time in the cameras.
    """
    reason = reason.replace(" ", "_")

    for i in range(nexps):
        tasks = [
            asyncio.create_task(take_images_with_sleep(cam, exptime, reason, sleep))
            for (cam, exptime) in zip(cams, exptimes)
        ]
        await asyncio.gather(*tasks)


async def take_images_in_sync_for_time(cams, exptimes, reason, tracktime, sleep=0.5):
    """Takes images in sync while tracking for some time.

    Parameters
    ----------
    cams : list of `lsst.ts.observatory.control.base_camera.BaseCamera`
        A list containing a camera instance.
    exptimes : list of `float`
        A list of exposure times.
    reason : `str`
        Reason passed to the `take_object` command.
    tracktime : `float`
        How long will we be tracking?
    sleep : `float`
        Sleep time in seconds to compensate for readout time in the cameras.
    """
    reason = reason.replace(" ", "_")
    timer_task = asyncio.create_task(asyncio.sleep(tracktime - max(exptimes)))
    n_images = 0

    while not timer_task.done():
        tasks = [
            asyncio.create_task(take_images_with_sleep(cam, exptime, reason, sleep))
            for (cam, exptime) in zip(cams, exptimes)
        ]
        await asyncio.gather(*tasks)

    return n_images


async def take_images_with_sleep(cam, exptime, reason, sleep):
    """Takes a single image with a generic camera and add a sleep
    after the task is complete.

    Parameters
    ----------
    cam : `lsst.ts.observatory.control.base_camera.BaseCamera`
        Camera Instance
    exptime : `float`
        Exposure time
    reason : `str`
        Reason passed to the `take_object` command.
    sleep : `float`
        Sleep time in seconds after complete image.
    """
    await cam.take_object(exptime, reason=reason)
    await asyncio.sleep(sleep)
