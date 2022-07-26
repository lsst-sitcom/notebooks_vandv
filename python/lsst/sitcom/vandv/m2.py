import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

from matplotlib.gridspec import GridSpec

from .efd import query_last_n


def plot_m2_actuators():
    """
    Plots/draw the position of the M2 actuators.

    See also
    --------
    https://docushare.lsst.org/docushare/dsweb/Get/Document-21545/
    SPIE%209906-239%20M2%20assembly%20final%20design%20-%20Final%20Submital.pdf
    """
    fig, ax = plt.subplots(figsize=(10, 10))

    actuators_position = np.loadtxt(
        f"{os.environ['HOME']}/notebooks/"
        f"lsst-sitcom/M2_FEA/data/M2_1um_72_force.txt"
    )

    act_id = actuators_position[:, 0]
    act_x = actuators_position[:, 1]  # meters
    act_y = actuators_position[:, 2]  # meters
    rows = "BCD"

    outer_diameter = 3.470  # meters
    inner_diameter = 1.775  # meters

    mirror = plt.Circle((0, 0), outer_diameter / 2.0, fc="k", alpha=0.2)
    ax.add_patch(mirror)

    inner = plt.Circle((0, 0), inner_diameter / 2.0, fc="w")
    ax.add_patch(inner)

    img = ax.scatter(act_x, act_y, s=1500)

    for x, y, t in zip(act_x, act_y, act_id):
        _id = np.mod(t, 10000)
        _rad = rows[int((t - _id) / 10000 - 2)]
        ax.text(x, y, f"{_rad}{_id:.0f}", color="w", ha="center", va="center")

    plt.show()


def plotM2Forces(axialForces, tangentForces, lut_path=None, size=100):
    """
    Plots the tangencial and axial forces for M2.

    Parameters
    ----------
    axialForces : telemetry
        Telemetry obtained from `mtm2.tel_axialForce.aget` containing
        the masured and the applied axial forces as 72 element arrays.
    tangentForces : telemetry
        Telemetry obtained from `mtm2.tel_tangentForce.aget` containing
        the measured and the applied tangencial forces as 6 element arrays.
    lut_path : str or None
        If None, the path to the look-up table falls back to
        `$HOME/notebooks/lsst-sitcom/M2_FEA/data/M2_1um_72_force.txt"
    size : int, optional
        The since of the dots in points (matplotlib).
    """
    if lut_path is None:
        lut_path = f"{os.environ['HOME']}/notebooks/" f"lsst-sitcom/M2_FEA/"

    lut_fname = os.path.join(lut_path, "data/M2_1um_72_force.txt")
    if not os.path.exists(lut_fname):
        raise FileNotFoundError(
            f"Could not find LUT for m2. Check the path below\n" f"  {lut_name}"
        )

    lut = np.loadtxt(lut_fname)

    # to have +x going to right, and +y going up, we need to transpose and reverse x and y
    xact = -lut[:, 2]
    yact = -lut[:, 1]

    fig = plt.figure(figsize=(18, 13))
    gs = GridSpec(3, 3)

    ax0 = fig.add_subplot(gs[0, :])
    ax0.plot(axialForces.measured, label="measured")
    ax0.plot(axialForces.applied, label="applied")
    ax0.plot(axialForces.hardpointCorrection, "o", label="FB")
    ax0.plot(axialForces.lutGravity, label="LUT G")
    ax0.set_ylabel("Axial Forces")
    ax0.set_xlabel("Axial Actuators Indexes")
    ax0.legend()

    ax1 = fig.add_subplot(gs[1, :])
    ax1.plot(tangentForces.measured, label="measured")
    ax1.plot(tangentForces.applied, label="applied")
    ax1.plot(tangentForces.hardpointCorrection, "o", label="FB")
    ax1.plot(tangentForces.lutGravity, label="LUT G")
    ax1.set_ylabel("Tangent Forces")
    ax1.set_xlabel("Tangent Actuators Indexes")
    ax1.legend()

    afm = np.array(axialForces.measured)
    ax2 = fig.add_subplot(gs[2, 0])
    img = ax2.scatter(xact, yact, c=afm, s=size)

    ax2.axis("equal")
    ax2.set_title("measured\n axial forces")
    fig.colorbar(img, ax=ax2)

    afa = np.array(axialForces.applied)
    ax3 = fig.add_subplot(gs[2, 1])
    img = ax3.scatter(xact, yact, c=afa, s=size)

    ax3.axis("equal")
    ax3.set_title("applied\n axial forces")
    fig.colorbar(img, ax=ax3)

    ax4 = fig.add_subplot(gs[2, 2])
    img = ax4.scatter(xact, yact, c=afm - afa, s=size)

    ax4.axis("equal")
    ax4.set_title("Measured minus Applied \nAxial Forces")
    fig.colorbar(img, ax=ax4)

    fig.tight_layout()
    return fig


async def show_last_forces_efd(
    client, lower_t=None, upper_t=None, execution=None, lut_path=None
):
    """Plots an snashot of the current M2 status using the
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
    execution : str, optional
        Test execution id (e.g. LVV-EXXXX).
    lut_path : str, optional
        Alternative path to a local copy of the `lsst-sitcom/M2_FEA` repository.
    """
    axialForces = await query_last_n(
        client,
        "lsst.sal.MTM2.axialForce",
        fields="*",
        upper_t=upper_t,
        lower_t=lower_t,
        debug=True,
    )

    tangentForces = await query_last_n(
        client,
        "lsst.sal.MTM2.tangentForce",
        fields="*",
        upper_t=upper_t,
        lower_t=lower_t,
        debug=True,
    )

    # Ugly way to convert dataframes into fake telemetries
    class TelAxialForces:
        measured = np.array([axialForces[f"measured{i}"] for i in range(72)])
        applied = np.array([axialForces[f"applied{i}"] for i in range(72)])
        lutGravity = np.array([axialForces[f"lutGravity{i}"] for i in range(72)])
        hardpointCorrection = np.array(
            [axialForces[f"hardpointCorrection{i}"] for i in range(72)]
        )

    class TelTangentForces:
        measured = np.array([tangentForces[f"measured{i}"] for i in range(6)])
        applied = np.array([tangentForces[f"applied{i}"] for i in range(6)])
        lutGravity = np.array([tangentForces[f"lutGravity{i}"] for i in range(6)])
        hardpointCorrection = np.array(
            [tangentForces[f"hardpointCorrection{i}"] for i in range(6)]
        )

    fig = plotM2Forces(TelAxialForces, TelTangentForces, lut_path=lut_path)
    plt.show()


def load_m2_lut(lut_path=None, lut_file=None):
    """Load the M2 Look-Up Table (LUT)

    Parameters
    ----------
    lut_path : str, optional
        Path to the repository containing the M2 LUT.
        Default: $HOME/notebooks/lsst-sitcom/M2_FEA
    lut_file : str, optional
        Name of the M2 LUT file.
        Default: data/data/M2_1um_72_force.txt
    """

    if lut_path is None:
        lut_path = f"{os.environ['HOME']}/notebooks/lsst-sitcom/M2_FEA/"

    if lut_file is None:
        lut_file = "data/M2_1um_72_force.txt"

    lut_fname = os.path.join(lut_path, lut_file)

    if not os.path.exists(lut_fname):
        raise FileNotFoundError(
            f"Could not find LUT for m2. Check the path below\n" f"  {lut_name}"
        )

    return np.loadtxt(lut_fname)


def convert_numid_to_strid(t):
    """Converts the numeric ID used inside the M2 LUT to their
    string ID. The actuators in the 20000 correspond to the B actuators
    (outer ring), 30000 to C actuators (middle ring), and 40000 to
    D actuators (inner ring).

    Parameter
    ---------
    t : int
        Actuator ID from the LUT.
    """
    rows = "BCD"
    _id = int(np.mod(t, 10000))
    _rad = rows[int((t - _id) / 10000 - 2)]
    return f"{_rad}{_id:0}"


def timeline_axial_forces(
    ax,
    df,
    column="measured",
    indexes=None,
    ids=None,
    elevation=None,
    lut_path=None,
    lut_file=None,
):
    """Shows the axial forces on multiple actuators on M2 as a time-line.

    Parameters
    ----------
    ax : matplotlib.Axes
        Axes where the plot will be held.
    df : pd.DataFrame
        Dataframe obtained from `lsst.sal.MTM2.axialForce`
        that will be plot.
    column : list of str, optional
        Column prefix used to select which forces to plot.
        Default: "applied"
    indexes : list of int, optional
        List of the indexes associated to each actuator (0-based)
    ids : list of in, optional
        List of the IDs associated to each actuator.
        Default: ["B1", "B16"].
    elevation : pd.DataFrame, optional
        Time-series containing the elevation angle obtained from
        `lsst.sal.MTMount.elevation`.
    lut_path : str, optional
        Path to the repository containing the M2 LUT.
        See `lsst.sitcom.vandv.m2.loat_m2_lut`.
    lut_file : str, optional
        Name of the M2 LUT file.
        See `lsst.sitcom.vandv.m2.loat_m2_lut`.
    """

    if ids and indexes:
        raise ValueError(
            "Both `ids` and `indexes` where provided when"
            " only one of them was expected."
        )

    lut = load_m2_lut(lut_path=lut_path, lut_file=lut_file)

    # Convert 200??/300??/400?? ids to B??/C??/D?? ids
    actuators_ids_table = [convert_numid_to_strid(t) for t in lut[:, 0]]

    if ids:
        indexes = [actuators_ids_table.index(_id) for _id in ids]
    elif indexes is not None:
        pass
    else:
        ids = ["B1", "B16"]
        indexes = [actuators_ids_table.index(_id) for _id in ids]

    for idx in indexes:
        forces = df[f"{column}{idx}"].dropna()
        ax.plot(forces, label=f"{column}{idx} ({actuators_ids_table[idx]})")

    ax.set_title(f"{column.capitalize()} Axial Force")
    ax.set_xlabel("Time")
    ax.set_ylabel("zForce [N]")
    ax.grid(":", alpha=0.2)
    ax.legend()

    if elevation is not None:
        el = elevation["actualPosition"].dropna()
        ax_twin = ax.twinx()
        ax_twin.fill_between(el.index, 0, el, fc="black", alpha=0.1)
        ax_twin.set_ylim(
            el.min() - 0.1 * el.values.ptp(), el.max() + 0.1 * el.values.ptp()
        )
        ax_twin.set_ylabel("Elevation [deg]")

    return ax


def timeline_axial_forces_per_act(
    ax, df, act="B1", idx=None, cols=None, elevation=None, lut_path=None, lut_file=None
):
    """Shows the axial forces on multiple actuators on M2 as a time-line.

    Parameters
    ----------
    ax : matplotlib.Axes
        Axes where the plot will be held.
    df : pd.DataFrame
        Dataframe obtained from `lsst.sal.MTM2.axialForce`
        that will be plot.
    act: str, optional
        IDs associated to the actuator that will be plot.
        Default: "B1"
    idx : int, optional
        Index in the LUT of the actuator that will be plot.
        Default: None
    cols : list of str, optional
        Column prefixes used to select which forces to plot.
        Default: ["applied", "hardpointCorrection", "lutGravity", "lutTemperature", "measured"]
    elevation : pd.DataFrame, optional
        Time-series containing the elevation angle obtained from
        `lsst.sal.MTMount.elevation`.
    lut_path : str, optional
        Path to the repository containing the M2 LUT.
        See `lsst.sitcom.vandv.m2.loat_m2_lut`.
    lut_file : str, optional
        Name of the M2 LUT file.
        See `lsst.sitcom.vandv.m2.loat_m2_lut`.
    """

    if act and idx:
        raise ValueError(
            "Both `act` and `idx` where provided when" " only one of them was expected."
        )

    lut = load_m2_lut(lut_path=lut_path, lut_file=lut_file)

    # Convert 200??/300??/400?? ids to B??/C??/D?? ids
    actuators_ids_table = [convert_numid_to_strid(t) for t in lut[:, 0]]

    # Deal with default values
    act = "B1" if act is None else act
    idx = actuators_ids_table.index(act)

    if cols is None:
        cols = [
            "applied",
            "hardpointCorrection",
            "lutGravity",
            "lutTemperature",
            "measured",
        ]

    # Plot data for each column
    for col in cols:
        forces = df[f"{col}{idx}"].dropna()
        ax.plot(forces, label=f"{col}{idx}")

    # Cosmetics
    ax.set_title(f"Axial Forces for {act}")
    ax.set_xlabel("Time")
    ax.set_ylabel("Forces [N]")
    ax.grid(":", alpha=0.2)
    ax.legend()

    # Plot elevation
    if elevation is not None:
        el = elevation["actualPosition"].dropna()
        ax_twin = ax.twinx()
        ax_twin.fill_between(el.index, 0, el, fc="black", alpha=0.1)
        ax_twin.set_ylim(
            el.min() - 0.1 * el.values.ptp(), el.max() + 0.1 * el.values.ptp()
        )
        ax_twin.set_ylabel("Elevation [deg]")

    return ax


def snapshot_zforces_overview(
    ax,
    series,
    prefix="measured",
    show_ids=True,
    show_mirrors=True,
    lut_path=None,
    lut_file=None,
    ms=250,
    fs=10,
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
    https://docushare.lsst.org/docushare/dsweb/Get/Document-21545/
    SPIE%209906-239%20M2%20assembly%20final%20design%20-%20Final%20Submital.pdf
    """
    import re
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    # Show mirror area
    if show_mirrors:
        outer_diameter = 3.470  # meters
        inner_diameter = 1.775  # meters

        mirror = plt.Circle((0, 0), outer_diameter / 2.0, fc="k", alpha=0.2)
        ax.add_patch(mirror)

        inner = plt.Circle((0, 0), inner_diameter / 2.0, fc="w")
        ax.add_patch(inner)

    # Show actuators and their values
    cols = [c for c in series.index if prefix in c]
    idxs = [int(s) for c in cols for s in re.findall(r"\d+", c)]

    # Get the position of the actuators
    lut = load_m2_lut(lut_path=lut_path, lut_file=lut_file)

    ids = lut[idxs, 0]
    act_x = lut[idxs, 1]  # meters
    act_y = lut[idxs, 2]  # meters

    data = series[cols]
    im = ax.scatter(act_x, act_y, c=data, s=ms)

    if show_ids:
        # Convert 200??/300??/400?? ids to B??/C??/D?? ids
        ids = [convert_numid_to_strid(t) for t in ids]
        for x, y, _id in zip(act_x, act_y, ids):
            ax.text(
                x,
                y,
                f"{_id}",
                color="w",
                ha="center",
                va="center",
                fontsize=fs,
            )

        ax.annotate("x", xy=(0.5, 0), xytext=(0, 0), arrowprops=dict(arrowstyle="->"))

    ax.axis("equal")
    ax.set_title(f"{prefix} axial forces")
    ax.set_axis_off()

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = ax.figure.colorbar(im, cax=cax, orientation="vertical")
    cbar.set_label("Force Intensity [N]")
    cbar.ax.tick_params(axis="y", labelsize=6)

    return ax
