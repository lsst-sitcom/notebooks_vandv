import numpy as np
import pandas as pd
import functools
import asyncio

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

    def extract_target_angles(offsets: pd.DataFrame) -> pd.DataFrame:
        """Modifies the input DataFrame to include extracted target angles."""
        # Initialize columns to zeros
        offsets["target_elevation"] = 0.0
        offsets["target_azimuth"] = 0.0
        offsets["target_rotation"] = 0.0
    
        # Iterate through the DataFrame and extract angles
        for index, row in offsets.iterrows():
            parts = row["target"].split("_")
            if parts[1] == 'CENTRAL':
                angles = [float(parts[2]), float(parts[3]), float(parts[4])]
            else:
                angles = [float(parts[1]), float(parts[2]), float(parts[3])]
            offsets.at[index, "target_elevation"] = angles[0]
            offsets.at[index, "target_azimuth"] = angles[1]
            offsets.at[index, "target_rotation"] = angles[2]
        
        return offsets

    if component == "Camera":
        offsets = offsets.loc[offsets["target"].str.contains("FrameCAM")].copy()
    elif component == "M2":
        offsets = offsets.loc[offsets["target"].str.contains("FrameM2")].copy()
    elif component == "TMA_CENTRAL":
        offsets = offsets.loc[offsets["target"].str.contains("FrameTMA_CENTRAL")].copy()

    processed_offsets = extract_target_angles(offsets)

    # Convert the target angles to um
    processed_offsets["dX"] *= 1e3
    processed_offsets["dY"] *= 1e3
    processed_offsets["dZ"] *= 1e3
    return processed_offsets

