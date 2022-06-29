import asyncio

import numpy as np
import os
import pandas as pd
import yaml


__all__ = [
    "check_hexapod_lut",
    "get_hexapod_configuration",
    "print_hexapod_compensation_values",
    "print_hexapod_uncompensation_values",
]


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
    print(" ".join(f"{p:10.2f}" for p in [getattr(posU, i) for i in "xyz"]), end="    ")
    print(
        " ".join(f"{p:10.6f}" for p in [getattr(posU, i) for i in "uvw"]),
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
    """
    if not lut_fname:
        lut_fname = (
            f"{os.environ['HOME']}/notebooks/lsst-ts/ts_config_mttcs/"
            f"MTHexapod/v1/default.yaml"
        )

    if not os.path.exist(lut_fname):
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
