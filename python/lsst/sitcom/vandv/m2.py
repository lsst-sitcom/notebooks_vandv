import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

from matplotlib.gridspec import GridSpec


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
        lut_path = (
            f"{os.environ['HOME']}/notebooks/"
            f"lsst-sitcom/M2_FEA/data/M2_1um_72_force.txt"
        )

    if not os.path.exist(lut_fname):
        raise FileNotFoundError(
            f"Could not find LUT for m2. Check the path below\n" f"  {lut_name}"
        )

    lut = np.loadtxt(lut_path)

    # to have +x going to right, and +y going up, we need to transpose and reverse x and y
    xact = -aa[:, 2]
    yact = -aa[:, 1]

    fig = plt.figure(figsize=(18, 13))
    gs = GridSpec(3, 3)

    ax0 = fig.add_subplot(gs[0, :])
    ax0.plot(axialForces.measured, label="measured")
    ax0.plot(axialForces.applied, label="applied")
    ax0.plot(axialForces.hardpointCorrection, "o", label="FB")
    ax0.plot(axialForces.lutGravity, label="LUT G")
    ax0.legend()

    ax1 = fig.add_subplot(gs[1, :])
    ax1.plot(tangentForces.measured, label="measured")
    ax1.plot(tangentForces.applied, label="applied")
    ax1.plot(tangentForces.hardpointCorrection, "o", label="FB")
    ax1.plot(tangentForces.lutGravity, label="LUT G")
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
    ax4.set_title("measured\n tangencial forces")
    fig.colorbar(img, ax=ax4)
