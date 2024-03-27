import numpy as np
import pandas as pd

def extract_component_data(offsets: pd.DataFrame, component: str) -> pd.DataFrame:
    """Extracts the component data from the offsets dataframe
    Parameters
    ----------
    offsets: pandas.DataFrame
        Dataframe with the offsets data from the EFD
    component: str
        Component to extract the data for. Can be 'M2', 'Camera'
    Returns
    -------
    pandas.DataFrame
        Dataframe with the offsets data for the specified component
    """

    def extract_target_angles(x: str) -> list[float]:
        """Extracts the target angles from the target string"""
        parts = x.split("_")

        if parts[1] == 'CENTRAL':
            return [float(parts[2]), float(parts[3]), float(parts[4])]
        else:
            return [float(parts[1]), float(parts[2]), float(parts[3])]

    if component == "Camera":
        offsets = offsets.loc[offsets["target"].str.startswith("FrameCAM")].copy()
    elif component == "M2":
        offsets = offsets.loc[offsets["target"].str.startswith("FrameM2")].copy()
    elif component == "TMA_CENTRAL":
        offsets = offsets.loc[offsets["target"].str.startswith("FrameTMA_CENTRAL")].copy()

    target_angles = offsets["target"].apply(lambda x: extract_target_angles(x))
    offsets["target_elevation"] = target_angles.apply(lambda x: x[0])
    offsets["target_azimuth"] = target_angles.apply(lambda x: x[1])
    offsets["target_rotation"] = target_angles.apply(lambda x: x[2])

    # Convert the target angles to um
    offsets["dX"] *= 1e3
    offsets["dY"] *= 1e3
    offsets["dZ"] *= 1e3
    return offsets