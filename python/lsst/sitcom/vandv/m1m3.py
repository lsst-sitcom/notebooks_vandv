import os
import asyncio

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def lut_elevation_forces(elevation, lut_file):
    """Returns the Elevation Forces for M1M3 based on the elevation angle
    and on a given look-up table.

    Parameters
    ----------
    elevation : float
        Elevation angle in degrees
    lut_file : string
        LUT name

    Returns
    -------
    array : the forces calculated from the lut

    See also
    --------
    The look-up tables here are the ones in
    https://github.com/lsst-ts/ts_m1m3support/tree/main/SettingFiles/Tables
    """
    ts_m1m3support = f"{os.environ['HOME']}/notebooks/lsst-ts/ts_m1m3support"

    if not os.path.exists(ts_m1m3support):
        raise OSError(f"Could not find: {ts_m1m3support}")

    lut_file = f"{ts_m1m3support}/SettingFiles/Tables/{lut_file}"
    lut_el = pd.read_csv(lut_file)

    n = len(lut_el.index)
    elevation_forces = np.zeros(n)

    zenith_angle = 90.0 - elevation

    for i in range(n):
        coeff = [lut_el["Coefficient %d" % j][i] for j in range(5, -1, -1)]
        mypoly = np.poly1d(coeff)
        elevation_forces[i] = mypoly(zenith_angle)

    return elevation_forces


def lut_elevation_xforces(elevation):
    """
    Return the Elevation xForces for M1M3 based on the Elevation angle.

    Parameters
    ----------
    elevation : float
        Elevation angle in degrees.

    Returns
    -------
    array : the xForces calculated from the lut.
    """
    lut_file = "ElevationXTable.csv"
    return lut_elevation_forces(elevation, lut_file)


def lut_elevation_yforces(elevation):
    """
    Return the Elevation yForces for M1M3 based on the Elevation angle.

    Parameters
    ----------
    elevation : float
        Elevation angle in degrees.

    Returns
    -------
    array : the xForces calculated from the lut.
    """
    lut_file = "ElevationYTable.csv"
    return lut_elevation_forces(elevation, lut_file)


def lut_elevation_zforces(elevation):
    """
    Return the Elevation zForces for M1M3 based on the Elevation angle.

    Parameters
    ----------
    elevation : float
        Elevation angle in degrees.

    Returns
    -------
    array : the zForces calculated from the lut.
    """
    lut_file = "ElevationZTable.csv"
    return lut_elevation_forces(elevation, lut_file)
