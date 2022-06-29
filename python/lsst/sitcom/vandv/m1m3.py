import asyncio
import os
import warnings

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from lsst_efd_client import EfdClient
from lsst.ts.cRIOpy.M1M3FATable import FATABLE

from .efd import query_last_n


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
            f"Could not find LUT for M1M3. Check the path below\n" f"  {lut_file}"
        )

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
        constrained_layout=True, figsize=(15, 7), num=figname, nrows=3
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
        ax0.plot(zforce, "-", c=c, label=f"zForce{i} ({actuators_ids_table[i]})")

    # plot forces vs elevation
    for i, c in zip(actuators_ids_idx, colors):
        el = df["elevation"]
        zforce = df[f"mtm1m3.forceActuatorData.zForce{i}"]
        ax1.plot(el, zforce, "-", c=c, label=f"zForce{i} ({actuators_ids_table[i]})")

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

    ax0b.set_ylim(
        el.min() - 0.15 * np.array(el).ptp(), el.max() + 0.15 * np.array(el).ptp()
    )
    ax0b.set_ylabel("Elevation (deg)")

    ax1.set_xlabel("Elevation [deg]")
    ax1.set_ylabel("zForce [N]")
    ax1.legend()

    ax2.set_xlabel("Time")
    ax2.set_ylabel("RMS [N]")
    ax2.legend()

    plt.show()


async def show_last_forces_efd(client, lower_t=None, upper_t=None, execution=None):
    """Plots an snashot of the current M1M3 status using the
    most recent data within the time range that was published to the
    EFD.

    Parameters
    ----------
    client : lsst_efd_client.EfdClient
        A live connection to the EFD.
    lower_t : `astropy.time.Time`, optional
        Lower time used in the query. (default: `upper_t - 15m`)
    upper_t : `astropy.time.Time`, optional
        Upper time used in the query. (default: `Time.now()`)
    execution : str
        Test execution id (e.g. LVV-EXXXX). 
    """
    from lsst.ts.cRIOpy import M1M3FATable
    
    # Number of actuators in X, Y and Z
    x_size = 12 
    y_size = 100
    z_size = 156

    # Topics for plotting 
    forces = {
        # "fapp": "lsst.sal.MTM1M3.logevent_appliedForces",  # Not working
        "fel": "lsst.sal.MTM1M3.logevent_appliedElevationForces",
        # "faz": "lsst.sal.MTM1M3.logevent_appliedAzimuthForces",
        # "fth": "lsst.sal.MTM1M3.logevent_appliedThermalForces",
        "fba": "lsst.sal.MTM1M3.logevent_appliedBalanceForces",
        # "fac": "lsst.sal.MTM1M3.logevent_appliedAccelerationForces",
        # "fve": "lsst.sal.MTM1M3.logevent_appliedVelocityForces",
        "fst": "lsst.sal.MTM1M3.logevent_appliedStaticForces",
        # "fab": "lsst.sal.MTM1M3.logevent_appliedAberrationForces",
        # "fof": "lsst.sal.MTM1M3.logevent_appliedOffsetForces",
        "fao": "lsst.sal.MTM1M3.logevent_appliedActiveOpticForces",
        "ftel": "lsst.sal.MTM1M3.forceActuatorData",
    }

    fx, fy, fz = {}, {}, {}
    for key, topic in forces.items():

        print(f"Query {topic}")
        await asyncio.sleep(1)

        df = await query_last_n(
            client, 
            topic, 
            fields="*", 
            upper_t=upper_t,
            lower_t=lower_t,
            debug=True
        )
        
        # Ugly way of extracting values        
        if key in ["ftel"]:        
            fx[key] = np.array([df[f"xForce{i}"] for i in range(x_size)]).squeeze()
            fy[key] = np.array([df[f"yForce{i}"] for i in range(y_size)]).squeeze()
            fz[key] = np.array([df[f"zForce{i}"] for i in range(z_size)]).squeeze()
        elif key in ["fao"]:
            fx[key] = np.empty(x_size)
            fy[key] = np.empty(y_size)
            fz[key] = np.array([df[f"zForces{i}"] for i in range(z_size)]).squeeze()
        else: 
            fx[key] = np.array([df[f"xForces{i}"] for i in range(x_size)]).squeeze()
            fy[key] = np.array([df[f"yForces{i}"] for i in range(y_size)]).squeeze()
            fz[key] = np.array([df[f"zForces{i}"] for i in range(z_size)]).squeeze()

    # Get the position of the actuators
    fat = np.array(M1M3FATable.FATABLE)
    xact = np.float64(fat[:, M1M3FATable.FATABLE_XPOSITION])
    yact = np.float64(fat[:, M1M3FATable.FATABLE_YPOSITION])

    # Create the plot
    fig = plt.figure(figsize=(15, 15), dpi=120)
    gs = gridspec.GridSpec(5, 3)

    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(fx["fel"], "C0o-", label="appliedElevationForces")
    ax1.plot(fx["fba"], "C1x--", label="appliedBalanceForces")
    ax1.plot(fx["fst"], "C2+:", label="appliedStaticForces")
    ax1.plot(fx["ftel"], "C3.-", label="forceActuatorData")
    ax1.set_ylabel("xForces")
    ax1.set_xlabel("Actuators Index")

    ax2 = fig.add_subplot(gs[1, :])
    ax2.plot(fy["fel"], "C0o-", label="appliedElevationForces")
    ax2.plot(fy["fba"], "C1x--", label="appliedBalanceForces")
    ax2.plot(fy["fst"], "C2+:", label="appliedStaticForces")
    ax2.plot(fy["ftel"], "C3.-", label="forceActuatorData")
    ax2.set_ylabel("yForces")
    ax2.set_xlabel("Actuators Index")

    ax3 = fig.add_subplot(gs[2, :])
    ax3.plot(fz["fel"], "C0o-", label="appliedElevationForces")
    ax3.plot(fz["fba"], "C1x--", label="appliedBalanceForces")
    ax3.plot(fz["fst"], "C2+:", label="appliedStaticForces")
    ax3.plot(fz["ftel"], "C3.-", label="forceActuatorData")
    ax3.set_ylabel("yForces")
    ax3.set_xlabel("Actuators Index")
    
    ax2.legend()

    short_list = ["fao", "fel", "fst"]
    long_list = [
        "appliedActiveOpticForces", 
        "appliedElevationForces",
        "appliedStaticForces",
    ]
    
    for i, (short, long) in enumerate(zip(short_list, long_list)):
        ax = fig.add_subplot(gs[3:, i])
        im = ax.scatter(xact, yact, c=fz[short])
        ax.axis("equal")
        ax.set_title(long)
        fig.colorbar(im, ax=ax)

    fig.tight_layout()
    
    if execution:
        time = df.index.strftime("%y%m%d_%H%M")[0]
        os.makedirs("./plots", exist_ok=True)
        fig.savefig(os.path.join("./plots", f"{execution}_m1m3_snapshot_{time}.png"))