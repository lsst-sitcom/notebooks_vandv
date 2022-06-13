import os
import asyncio

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from matplotlib.gridspec import GridSpec

from lsst.ts.cRIOpy.M1M3FATable import FATABLE


def get_rms(s):
    """
    Gets the RMS of the zForces for each elevation angle
    
    s : pandas.Series
        DataFrame containing the all the applied z-forces on 
        M1M3 (lsst.sal.MTM1M3.forceActuatorData.zForce??) and the 
        elevation angle as "elevation".
    """
    
    lut = lut_elevation_zforces(s["elevation"])
    cols = [c for c in s.index for fa in FATABLE if f"zForce{fa[1]}" in c]
    
    zforces = np.array(s["mtm1m3.forceActuatorData.zForce101"])
    rms = np.sqrt((1 / zforces) * np.sum((lut - zforces) ** 2))
    return rms


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

    return np.array(elevation_forces)


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


def plot_m1m3_and_elevation(df, prefix=None):
    """
    Plots the forces applied in M1M3 at the +/-X and the +/-Y
    extremes as a function of time as has a function of the 
    elevation so we can have some better idea of how the 
    elevation affects these actuators.
    
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the all the applied z-forces on 
        M1M3 (lsst.sal.MTM1M3.forceActuatorData.zForce??). 
    """
    from lsst.ts.cRIOpy.M1M3FATable import FATABLE
    
    actuators_ids = [129, 229, 329, 429]
    actuators_ids_table = [fa[1] for fa in FATABLE]
    actuators_ids_idx = [actuators_ids_table.index(_id) for _id in actuators_ids]
    colors = ["C0", "C1", "C2", "C3"]
       
    # Prepare figure name
    figname = "M1M3 and Elevation"
    figname = f"{prefix} - {figname}" if prefix else figname

    # Create figure and axis
    fig, (ax0, ax1, ax2) = plt.subplots(
        constrained_layout=True,
        figsize=(15, 7), 
        num=figname, 
        nrows=3
    )
    ax0b = ax0.twinx()

    # Resample for simplicity
    df = df.resample("0.5S").mean()
    
    # Get rms
    df["rms"] = df.apply(get_rms, axis=1)
      
    # Plot forces vs time
    for i, c in zip(actuators_ids_idx, colors):
        zforce = df[f"mtm1m3.forceActuatorData.zForce{i}"].dropna()
        ax0.plot(zforce, "-", c=c, 
                 label=f"zForce{i} ({actuators_ids_table[i]})")
        
    # Plot elevation vs time
    el = df["elevation"].dropna()    
    ax0b.fill_between(el.index, 0, el, fc="black", alpha=0.1)
    
    # plot forces vs elevation
    for i, c in zip(actuators_ids_idx, colors):
        el = df["elevation"]
        zforce = df[f"mtm1m3.forceActuatorData.zForce{i}"]
        ax1.plot(el, zforce, "-", c=c, 
                 label=f"zForce{i} ({actuators_ids_table[i]})")        
    # plot rms
    ax2.plot(df["rms"], label="all actuators rms")
    
    # Tweak axes
    ax0.set_xlabel("Time")
    ax0.set_ylabel("zForce [N]")
    ax0.legend()
    
    ax0b.set_ylim(el.min() - 0.15 * np.array(el).ptp(), el.max() + 0.15 * np.array(el).ptp())
    ax0b.set_ylabel("Elevation (deg)")
    
    ax1.set_xlabel("Elevation [deg]")
    ax1.set_ylabel("zForce [N]")   
    ax1.legend()
    
    ax2.set_xlabel("Time")
    ax2.set_ylabel("RMS [N]")   
    ax2.legend()
    
    plt.show()
