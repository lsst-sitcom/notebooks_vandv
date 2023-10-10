import asyncio
import os
import re
import warnings

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from lsst.ts.xml.tables.m1m3 import actuator_id_to_index, FATable

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
    cols = [c for c in s.index for fa in FATable if f"zForce{fa.z_index}" in c]

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
        M1M3 (lsst.sal.MTM1M3.forceActuatorData.zForce).
    """
    actuators_ids = [129, 229, 329, 429]
    actuators_ids_idx = [actuator_id_to_index(actuator_id) for actuator_id in actuators_ids]
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
        ax0.plot(zforce, "-", c=c, label=f"zForce{i} ({actuators_ids[i]})")

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
            client, topic, fields="*", upper_t=upper_t, lower_t=lower_t, debug=True
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
    xact = [fa.x_position for fa in FATable]
    yact = [fa.y_position for fa in FATable]

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
        im = ax.scatter(xact, yact, c=fz[short], s=100)
        ax.axis("equal")
        ax.set_title(long)
        fig.colorbar(im, ax=ax)

    fig.tight_layout()

    if execution:
        time = df.index.strftime("%y%m%d_%H%M")[0]
        os.makedirs("./plots", exist_ok=True)
        fig.savefig(os.path.join("./plots", f"{execution}_m1m3_snapshot_{time}.png"))


def timeline_zforces(
    ax,
    df,
    column="zForce",
    labels=None,
    colors=None,
    ls=None,
    indexes=None,
    ids=None,
    elevation=None,
):
    """Shows the z-forces applied on M1M3 as a time-line.

    Parameters
    ----------
    ax : matplotlib.Axes
        Axes where the plot will be held.
    df : pd.DataFrame
        Dataframe obtained from `lsst.sal.MTM1M3.forceActuatorData`
        that will be plot.
    column : str, optional
        Column prefix used to select which forces to plot.
        Default: zForce.
    colors : list of strings, optional
        Colors applied to each data.
    ls : list of strings, optional
        Linestyle applied to each data.
    indexes : list of int, optional
        List of the indexes associated to each actuator (0-based)
    ids : list of in, optional
        List of the IDs associated to each actuator.
        Default: [129, 229, 329, 429].
    elevation : pd.DataFrame, optional
        Time-series containing the elevation angle obtained from
        `lsst.sal.MTMount.elevation`.
    """
    if ids and indexes:
        raise ValueError(
            "Both `ids` and `indexes` where provided when"
            " only one of them was expected."
        )

    actuators_ids_table = [fa.actuator_id for fa in FATable]

    if ids:
        indexes = [actuators_ids_table.index(_id) for _id in ids]
    elif indexes is not None:
        pass
    else:
        ids = [129, 229, 329, 429]
        indexes = [actuators_ids_table.index(_id) for _id in ids]

    for idx in indexes:
        forces = df[f"{column}{idx}"].dropna()
        ax.plot(forces, label=f"{column}{idx} ({actuators_ids_table[idx]})")

    if elevation is not None:
        el = elevation["actualPosition"].dropna()
        ax_twin = ax.twinx()
        ax_twin.fill_between(el.index, 0, el, fc="black", alpha=0.1)
        ax_twin.set_ylim(
            el.min() - 0.1 * el.values.ptp(), el.max() + 0.1 * el.values.ptp()
        )
        ax_twin.set_ylabel("Elevation [deg]")

    ax.set_title("zForce Actuator Data")
    ax.set_xlabel("Time")
    ax.set_ylabel("zForce [N]")
    ax.grid(":", alpha=0.2)
    ax.legend()
    return ax


def snapshot_forces(ax, series, prefix, labels=None):
    """Plot a snapshot of the xForces in pd.Series.

    Parameters
    ----------
    ax : matplotlib.Axes
        Axis that will hold the plot
    series : list
        List of series containing the xForce(s).
    prefix : str
        Prefix used to select the columns
    label : list, optional
        List of labels for the series.
    """
    if labels and (len(series) != len(labels)):
        raise ValueError(
            "Expected the number of elemenets in `series` and `labels`"
            " to be the same"
        )

    if labels is None:
        labels = [""] * len(series)

    for s, l in zip(series, labels):
        data = [s[c] for c in s.index if prefix in c]
        ax.plot(data, label=l)

    ax.set_title(prefix)
    ax.set_xlabel("Actuators Index")
    ax.set_ylabel(prefix)
    ax.grid(":", alpha=0.2)

    if all(labels):
        ax.legend()

    return ax


def snapshot_xforces(ax, series, labels=None):
    ax = snapshot_forces(ax, series, "xForce", labels=labels)
    return ax


def snapshot_yforces(ax, series, labels=None):
    ax = snapshot_forces(ax, series, "yForce", labels=labels)
    return ax


def snapshot_zforces(ax, series, labels=None):
    ax = snapshot_forces(ax, series, "zForce", labels=labels)
    return ax


def snapshot_zforces_overview(
    ax, series, prefix="zForce", title="", size=100, show_ids=True, show_mirrors=True
):
    """Show the force intensity on each actuator.

    Parameters
    ----------
    ax : matplotlib.Axes
        Axis that will hold the plot.
    series : pd.Series
        Series containing the forces.
    prefix : str
        Prefix used to select the columns.
    title : str, optional
        Label for the plot.
    size : int, optional
        Dot size in points.
    show_ids: bool, optional
        Show each actuator ID (default: False)
    show_mirrors: bool, optional
        Show the mirrors area (default: False)

    See also
    --------
    https://docushare.lsst.org/docushare/dsweb/Get/LSE-11/
    LSE-11_OpticalDesignSummary_rel3.5_20190819.pdf (Figure 15)
    """
    from lsst.ts.criopy import M1M3FATable
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    # Show mirror area
    if show_mirrors:
        m1_outer_diameter = 8.405  # meters
        m1_inner_diameter = 5.116  # meters
        m3_outer_diameter = 5.016  # meters
        m3_inner_diameter = 1.100  # meters

        m1_mirror = plt.Circle((0, 0), m1_outer_diameter / 2.0, fc="k", alpha=0.2)
        ax.add_patch(m1_mirror)

        m1_inner = plt.Circle((0, 0), m1_inner_diameter / 2.0, fc="w")
        ax.add_patch(m1_inner)

        m3_mirror = plt.Circle((0, 0), m3_outer_diameter / 2.0, fc="k", alpha=0.2)
        ax.add_patch(m3_mirror)

        m3_inner = plt.Circle((0, 0), m3_inner_diameter / 2.0, fc="w")
        ax.add_patch(m3_inner)

    # Show actuators and their values
    cols = [c for c in series.index if prefix in c]
    idxs = [int(s) for c in cols for s in re.findall(r"\d+", c)]

    # Get the position of the actuators
    ids = [fa.actuator_id for fa in FATable]
    xact = -np.float64([fa.x_position for fa in FATable])
    yact = -np.float64([fa.y_position for fa in FATable])

    data = series[cols]
    im = ax.scatter(xact, yact, c=data, s=size)

    if show_ids:
        for x, y, _id in zip(xact, yact, ids):
            ax.text(
                x,
                y,
                f"{_id}",
                color="w",
                ha="center",
                va="center",
                fontsize=0.05 * size,
            )

        off = m1_outer_diameter * 0.45
        ax.annotate(
            "x",
            xy=(1.0 - off, 0 - off),
            xytext=(-0.5 - off, 0 - off),
            arrowprops=dict(arrowstyle="->"),
            ha="center",
            va="center",
        )
        ax.annotate(
            "y",
            xy=(0 - off, 1.0 - off),
            xytext=(0 - off, -0.5 - off),
            arrowprops=dict(arrowstyle="->"),
            ha="center",
            va="center",
        )

    ax.axis("equal")
    ax.set_title(title)
    ax.set_axis_off()

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = ax.figure.colorbar(im, cax=cax, orientation="vertical")

    cbar.set_label("Force Intensity [N]")

    cbar.ax.tick_params(axis="y", labelsize=6)

    return ax
