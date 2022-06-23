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
    rms = np.sqrt((1 / zforces.size) * np.sum((lut - zforces) ** 2))
    return rms


def lut_elevation_forces(elevation, lut_fname, lut_path=None):
    """Returns the Elevation Forces for M1M3 based on the elevation angle
    and on a given look-up table.

    Parameters
    ----------
    elevation : float
        Elevation angle in degrees
    lut_fname : string
        LUT name
    lut_path : str or None
        The path to the directory that holds the `lut_file`. 
        If `None`, it falls back to 
        `$HOME/notebooks/lsst-ts/ts_m1m3support/SettingFiles/Tables/`

    Returns
    -------
    array : the forces calculated from the lut

    See also
    --------
    The look-up tables here are the ones in
    https://github.com/lsst-ts/ts_m1m3support/tree/main/SettingFiles/Tables
    """
    if lut_path is None:
        lut_path = (
            f"{os.environ['HOME']}/notebooks/lsst-ts/ts_m1m3support/"
            "SettingFiles/Tables/"
        )

    lut_file = os.path.join(lut_path, lut_fname)

    if not os.path.exist(lut_file):
        raise FileNotFoundError(
            f"Could not find LUT for M1M3. Check the path below\n"
            f"  {lut_file}"

    lut_el = pd.read_csv(lut_file)

    n = len(lut_el.index)
    elevation_forces = np.zeros(n)

    zenith_angle = 90.0 - elevation

    for i in range(n):
        coeff = [lut_el["Coefficient %d" % j][i] for j in range(5, -1, -1)]
        mypoly = np.poly1d(coeff)
        elevation_forces[i] = mypoly(zenith_angle)

    return np.array(elevation_forces)


def lut_elevation_xforces(elevation, lut_path=None):
    """
    Return the Elevation xForces for M1M3 based on the Elevation angle.

    Parameters
    ----------
    elevation : float
        Elevation angle in degrees.
    lut_path : str or None, optional
        The path to the directory containing the look-up table. If `None`
        it fallsback to the default location. 
        See the docstring for `lut_elevation_forces` for details.

    Returns
    -------
    array : the xForces calculated from the lut.
    """
    lut_file = "ElevationXTable.csv"
    return lut_elevation_forces(elevation, lut_file, lut_path=lut_path)


def lut_elevation_yforces(elevation, lut_path=None):
    """
    Return the Elevation yForces for M1M3 based on the Elevation angle.

    Parameters
    ----------
    elevation : float
        Elevation angle in degrees.
    lut_path : str or None, optional
        The path to the directory containing the look-up table. If `None`
        it fallsback to the default location. 
        See the docstring for `lut_elevation_forces` for details.

    Returns
    -------
    array : the xForces calculated from the lut.
    """
    lut_file = "ElevationYTable.csv"
    return lut_elevation_forces(elevation, lut_file, lut_path=lut_path)


def lut_elevation_zforces(elevation, lut_path=None):
    """
    Return the Elevation zForces for M1M3 based on the Elevation angle.

    Parameters
    ----------
    elevation : float
        Elevation angle in degrees.
    lut_path : str or None, optional
        The path to the directory containing the look-up table. If `None`
        it fallsback to the default location. 
        See the docstring for `lut_elevation_forces` for details.

    Returns
    -------
    array : the zForces calculated from the lut.
    """
    lut_file = "ElevationZTable.csv"
    return lut_elevation_forces(elevation, lut_file, lut_path=lut_path)


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
    ax2b = ax2.twinx()

    # Resample for simplicity
    df = df.resample("0.5S").mean()
    
    # Get rms
    df["rms"] = df.apply(get_rms, axis=1)
      
    # Plot forces vs time
    for i, c in zip(actuators_ids_idx, colors):
        zforce = df[f"mtm1m3.forceActuatorData.zForce{i}"].dropna()
        ax0.plot(zforce, "-", c=c, 
                 label=f"zForce{i} ({actuators_ids_table[i]})")
            
    # plot forces vs elevation
    for i, c in zip(actuators_ids_idx, colors):
        el = df["elevation"]
        zforce = df[f"mtm1m3.forceActuatorData.zForce{i}"]
        ax1.plot(el, zforce, "-", c=c, 
                 label=f"zForce{i} ({actuators_ids_table[i]})")        

    # plot rms vs time
    ax2.plot(df["rms"], label="all actuators rms")
    
    # Plot elevation vs time
    el = df["elevation"].dropna()    
    ax0b.fill_between(el.index, 0, el, fc="black", alpha=0.1)
    ax2b.fill_between(el.index, 0, el, fc="black", alpha=0.1)
    
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

    
async def plotM1M3Forces(mtm1m3):
    from lsst.ts.cRIOpy import M1M3FATable

    fel = await mtm1m3.evt_appliedElevationForces.aget(timeout=10.)
    faz = await mtm1m3.evt_appliedAzimuthForces.aget(timeout=10.)
    fth = await mtm1m3.evt_appliedThermalForces.aget(timeout=10.)
    fba = await mtm1m3.evt_appliedBalanceForces.aget(timeout=10.)
    fac = await mtm1m3.evt_appliedAccelerationForces.aget(timeout=10.)
    fve = await mtm1m3.evt_appliedVelocityForces.aget(timeout=10.)
    fst = await mtm1m3.evt_appliedStaticForces.aget(timeout=10.)
    fab = await mtm1m3.evt_appliedAberrationForces.aget(timeout=10.)
    fof = await mtm1m3.evt_appliedOffsetForces.aget(timeout=10.)
    fao = await mtm1m3.evt_appliedActiveOpticForces.aget(timeout=10.)
    fapp = await mtm1m3.evt_appliedForces.aget(timeout=10.)

    ftel = await mtm1m3.tel_forceActuatorData.next(flush=True, timeout=10.)
    
    # Get the position of the actuators
    fat = np.array(M1M3FATable.FATABLE)
    xact = np.float64(fat[:, M1M3FATable.FATABLE_XPOSITION])
    yact = np.float64(fat[:, M1M3FATable.FATABLE_YPOSITION])
        
    # Create the plot
    fig, ax = plt.subplots(3,1, figsize=(15,8))
    print(fel.xForces)
    ax[0].plot(fel.xForces, '-o', label='elevation');
    ax[0].plot(fba.xForces, label='FB')
    ax[0].plot(fst.xForces, label='static')
    ax[0].plot(ftel.xForce, '-v', label='measured')
    ax[0].legend()
    ax[0].set_title('XForces')
    
    ax[1].plot(fel.yForces, '-o', label='elevation');
    ax[1].plot(fba.yForces, label='FB')
    ax[1].plot(fst.yForces, label='static')
    ax[1].plot(ftel.yForce, '-v', label='measured')
    ax[1].legend()
    ax[1].set_title('YForces')
    ax[2].plot(fel.zForces, '-o', label='elevation');
    ax[2].plot(fba.zForces, label='FB')
    ax[2].plot(fst.zForces, label='static')
    ax[2].plot(fao.zForces, label='AOS')
    ax[2].plot(ftel.zForce, '-v', label='measured')
    ax[2].set_title('ZForces')
    ax[2].legend()
    
    fig2, ax=plt.subplots( 1,3, figsize = [15,4])
    aa = np.array(fao.zForces)
    img = ax[0].scatter(xact, yact, c=aa)
    ax[0].axis('equal')
    ax[0].set_title('AOS forces')
    fig.colorbar(img, ax=ax[0])

    aa = np.array(fel.zForces)
    img = ax[1].scatter(xact, yact, c=aa)
    ax[1].axis('equal')
    ax[1].set_title('elevation forces')
    fig.colorbar(img, ax=ax[1])
    
    aa = np.array(fst.zForces)
    img = ax[2].scatter(xact, yact, c=aa)
    ax[2].axis('equal')
    ax[2].set_title('static forces')
    fig.colorbar(img, ax=ax[2])