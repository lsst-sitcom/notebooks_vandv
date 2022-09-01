import asyncio
import logging
import os

import numpy as np
import pandas as pd
import yaml


__all__ = [
    "check_hexapod_lut",
    "get_hexapod_configuration",
    "print_hexapod_compensation_values",
    "print_hexapod_uncompensation_values",
]


log = logging.getLogger(__name__)


async def check_hexapod_lut(component, timeout=10.0):
    """Get the compensation offsets and print them. This is a way of telling
    if the component (hexapod) has enough input to do look-up table (LUT)
    compensation or not.

    Parameters
    ----------
    component : str
        Name of the component: mthexapod_1 for Camhex or mthexapod_2 for M2Hex
    timeout : float

    Note
    ----
    The target events are what the hexa CSC checks. If one is missing,
    the entire LUT will not be applied. It also needs to see an
    `uncompensatedPosition` (a move would trigger that) in order to
    move to the compensatedPosition.
    """
    print(
        "Does the hexapod has enough inputs to do LUT compensation? "
        "(If the below times out, we do not.)"
    )

    lut_mode = await component.evt_compensationMode.aget(timeout=10)
    offset = await component.evt_compensationOffset.aget(timeout=10.0)

    print("compsensation mode enabled?", lut_mode.enabled)
    print("mount elevation = ", offset.elevation)
    print("mount azimth = ", offset.azimuth)
    print("rotator angle = ", offset.rotation)
    print("temperature from sensors on the hex = ", offset.temperature)
    print("\n".join(f"{i} = {getattr(offset, i):.2f}" for i in "xyzuvw"))


async def get_hexapod_configuration(component, timeout=10.0):
    """
    Retrieves and prints the hexapod (CamHex or M2Hex) configuration.

    Parameters
    ----------
    component : str
        Name of the component: mthexapod_1 for Camhex or mthexapod_2 for M2Hex
    timeout : float
    """
    print(component.evt_configurationApplied.get())

    cfg = await component.evt_configuration.aget(timeout=10.0)

    print(
        f"\nPivot at ({cfg.pivotX}, {cfg.pivotY}, {cfg.pivotZ}) microns"
        f"\n maxXY = {cfg.maxXY} microns, maxZ = {cfg.maxZ} microns"
        f"\n maxUV = {cfg.maxUV} deg, maxW = {cfg.maxW} deg"
    )


async def print_hexapod_compensation_values(component, timeout=10.0):
    """Prints out the hexapod conpensation values."""
    posU = await component.evt_compensatedPosition.aget(timeout=10.0)
    print("Compensated position")
    print(
        " ".join(f"{p:10.2f} um" for p in [getattr(posU, i) for i in "xyz"]), end="    "
    )
    print(
        " ".join(f"{p:10.6f} deg" for p in [getattr(posU, i) for i in "uvw"]),
        "  ",
        pd.to_datetime(posU.private_sndStamp, unit="s"),
    )


async def print_hexapod_position(component, timeout=10.0):
    """Prints out the current hexapod position"""
    pos = await component.tel_application.next(flush=True, timeout=10.0)
    print("Current Hexapod position")
    print(" ".join(f"{p:10.2f}" for p in pos.position[:3]), end=" ")
    print(" ".join(f"{p:10.6f}" for p in pos.position[3:]))


async def print_hexapod_uncompensation_values(component, timeout=10.0):
    """Prints out the hexapod unconpensation values."""
    posU = await component.evt_uncompensatedPosition.aget(timeout=10.0)
    print("Uncompensated position")
    print(
        " ".join(f"{p:10.2f} um" for p in [getattr(posU, i) for i in "xyz"]), end="    "
    )
    print(
        " ".join(f"{p:10.6f} deg" for p in [getattr(posU, i) for i in "uvw"]),
        "  ",
        pd.to_datetime(posU.private_sndStamp, unit="s"),
    )


def coeffs_from_lut(index, lut_path=None):
    """Reads the elevation and temperature coefficients from the Look-Up Table

    Parameters
    ----------
    index : 1 or 2
        The SAL index for the hexapod (1 = Camera Hexapod, 2 = M2 Hexapod)
    lut_path : str or None
        If None, the path to the look-up table falls back to
        `$HOME/notebooks/lsst-ts/ts_config_mttcs/MTHexapod/v1/default.yaml`

    Returns
    -------
    elevCoeff : array
        Elevation coefficients
    tCoeff : array
        Temperature coefficients
    lut_path : str, optional
        Alternative path to a local copy of the `lsst-ts/ts_config_mttcs` repository.
    """
    if not lut_path:
        lut_path = f"{os.environ['HOME']}/notebooks/lsst-ts/ts_config_mttcs/"

    lut_fname = os.path.join(lut_path, "MTHexapod/v1/default.yaml")
    if not os.path.exists(lut_fname):
        raise FileNotFoundError(
            f"Could not find LUT for hexapod. Check the path below\n" f"  {lut_name}"
        )

    with open(lut_fname, "r") as stream:
        lut_stream = yaml.safe_load(stream)

    if index == 1:
        elevCoeff = lut_stream["camera_config"]["elevation_coeffs"]
        tCoeff = lut_stream["camera_config"]["temperature_coeffs"]

    elif index == 2:
        elevCoeff = lut_stream["m2_config"]["elevation_coeffs"]
        tCoeff = lut_stream["m2_config"]["temperature_coeffs"]

    else:
        raise ValueError("Index shall be 1 or 2")

    return elevCoeff, tCoeff


async def print_predicted_compensation(elevCoeff, elev):
    """Deals with the elevation component of the LUT only, for now.
    We will add temperature, azimuth, and rotator angle when they are implemented.
    """
    pred = []
    print("Predicted LUT compensation:")
    for i in range(6):
        coeff = elevCoeff[i]  # starts with C0
        mypoly = np.polynomial.Polynomial(coeff)
        pred.append(mypoly(elev))
    print(" ".join(f"{p:10.2f}" for p in pred))


def timeline_position(ax, dfs, column="z", elevation=None, symbols=None, names=None):
    """Show the Camera/M2 Hexapod positions as a timeline.

    Parameters
    ----------
    ax : matplotlib.Axes
        Axes that will hold the plot.
    dfs : list of pd.DataFrame
        List of dataframes containing the positions.
    column : str
        What axis to plot.
        Default: "z"
    symbols : list of str
        List of matplotlib symbols that will be applied to each plot.
        It can also be a combination of color symbols like "C0o-".
    names : list of str
        List of labels for each line.
    """
    column = column.lower()

    # Validate Inputs
    if column in "xyz":
        unit = "um"
    elif column in "uvw":
        unit = "urad"
    else:
        raise ValueError("Expected column to be x/y/z or u/v/w")

    if symbols:
        if len(symbols) != len(dfs):
            raise ValueError(
                "Expected number of elements in `dfs` and in `symbols` to be the same."
            )
    else:
        symbols = [""] * len(dfs)

    if names:
        if len(names) != len(dfs):
            raise ValueError(
                "Expected number of elements in `dfs` and in `names` to be the same."
            )
    else:
        names = [""] * len(dfs)

    # Plot Data
    for i, (df, s, name) in enumerate(zip(dfs, symbols, names)):
        s = "-" if s == "" else s
        try:
            ax.plot(df[column], f"{s}", label=name)
        except KeyError:
            log.warning(f"Column {column} not found in {i}-th dataframe {name}")

    # Customize Axis
    ax.grid(":", alpha=0.2)
    ax.set_xlabel("Time")
    ax.set_ylabel(f"{column}\n [{unit}]")

    if elevation is not None:
        el = elevation["actualPosition"].dropna()
        ax_twin = ax.twinx()
        ax_twin.fill_between(el.index, 0, el, fc="black", alpha=0.1)
        ax_twin.set_ylim(
            el.min() - 0.1 * el.values.ptp(), el.max() + 0.1 * el.values.ptp()
        )
        ax_twin.set_ylabel("Elevation\n [deg]")

    return ax


def get_lut_positions(index, elevation, lut_path=None):
    """Get the x/y/z/u/v/w position for an hexapod given the elevation angle"""
    elevCoeff, tempCoeff = coeffs_from_lut(index=index, lut_path=lut_path)

    pos = [np.zeros_like(elevation)] * 6
    for i in range(6):
        coeff = elevCoeff[i]  # starts with C0
        mypoly = np.polynomial.Polynomial(coeff)
        pos[i] = mypoly(elevation)

    return np.array(pos).T
