import asyncio
import logging

import numpy as np

from astropy.time import Time


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
    logger.info(f"{'Time':25s}{'Steps':>10s}{'New Az':>10s}{'New El':>10s}")
    
    generator = generate_azel_sequence(_az_grid, _el_grid)
    old_az, old_el = None, None
    
    while not timer_task.done():
    
        new_az, new_el = next(generator)
        
        if new_az == old_az and new_el == old_el:
            continue
        
        yield new_az, new_el
        
        t = Time.now().to_value("isot")
        logger.info(f"{t:25s}{step:10d}{new_az:10.2f}{new_el:10.2f}")

        old_az, old_el = new_az, new_el
        step += 1
        

def generate_azel_sequence(az_seq, el_seq):
    """A generator that cicles through the input azimuth and elevation sequences
    forward and backwards.
    
    Parameters
    ----------
    az_seq : `list` [`float`]
        A sequence of azimuth values to cicle through
    el_seq : `list` [`float`]
        A sequence of elevation values to cicle through     
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
    i, j = 1, 1
    while True:
        for az in az_seq[::j]:
            for el in el_seq[::i]:
                yield (az, el)
            i *= -1
        j *= -1
        
                
def random_walk_azel_by_time(total_time, 
                             mtmount,
                             radius=3.5, 
                             min_az=-200., 
                             max_az=+200, 
                             min_el=30, 
                             max_el=80, 
                             logger=None,
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
    logger.info(f"{'Time':25s}{'Steps':>10s}{'Old Az':>10s}{'Old El':>10s}"{'New Az':>10s}{'New El':>10s}")
    
    while not timer_task.done():
        
        current_az = mtmount.tel_azimuth.get()
        current_az = current_az.actualPosition
        # current_az = _az
        offset_az = np.sqrt(radius) * (2 * np.random.rand() - 1)
        new_az = current_az + offset_az
                
        current_el = mtmount.tel_elevation.get()
        current_el = current_el.actualPosition
        # current_el = _el
        offset_el = np.sqrt(radius) * (2 * np.random.rand() - 1)
        new_el = current_el + offset_el
        
        if new_az <= min_az or new_az >= max_az:
            new_az = current_az - offset_az
            
        if new_el <= min_el or new_el >= max_el:
            new_el = current_el - offset_el

        t = Time.now().to_value("isot")
        logger.info(
            f"{t:25s}{step:10d}{old_az:10.2f}{old_el:10.2f}{new_az:10.2f}{new_el:10.2f}")

        yield new_az, new_el
        step += 1