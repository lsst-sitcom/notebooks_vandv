
import asyncio
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

from astropy.time import Time
from logging import CRITICAL, ERROR

from lsst.summit.utils.efdUtils import (
    EfdClient,
    getDayObsEndTime,
    getDayObsForTime,
    getDayObsStartTime, 
    makeEfdClient
)

async def query_m1m3_faults(day_obs_start: int, day_obs_end: int) -> pd.DataFrame:
    """
    Query M1M3 faults from the EFD.

    Parameters
    ----------
    day_obs_start : int
        Start day for the query.
    day_obs_end : int
        End day for the query.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the M1M3 faults.
    """
    # Create an EFD client instance
    client = makeEfdClient()

    # Get the start and end times for the observation days
    start_time = getDayObsStartTime(day_obs_start)
    end_time = getDayObsEndTime(day_obs_end)
    print(f"Querying M1M3 faults from {start_time.isot} to {end_time.isot}")
    
    # Query the following columns
    # ['filePath', 'functionName', 'level', 'lineNumber', 'message', 'name', 'traceback'],
    query = f"""
        SELECT FIRST("filePath") as filePath,
               FIRST("functionName") as functionName,
               FIRST("level") as level,
               FIRST("name") as first_name,
               FIRST("lineNumber") as lineNumber,
               FIRST("message") as message,
               FIRST("traceback") as traceback
        FROM "lsst.sal.MTM1M3.logevent_logMessage"
        WHERE time >= '{start_time.isot}Z'
        AND time <= '{end_time.isot}Z'
        AND level > 30
        GROUP BY "name", time(1s) fill(none)
    """
    df = await client.influx_client.query(query)
    df = await add_elevation_to_df(df, client)
    return df


async def query_tma_elevation(start_time: Time, end_time: Time, client: EfdClient) -> pd.DataFrame:
    """
    Query TMA elevation from the EFD.

    Parameters
    ----------
    start_time : Time
        Start time for the query.
    end_time : Time
        End time for the query.
    client : EfdClient
        EFD client to use for querying.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the TMA elevation.
    """

    query = f"""
        SELECT MEAN("actualPosition") as elevation
        FROM "lsst.sal.MTMount.elevation"
        WHERE time >= '{Time(start_time).isot}Z'
        AND time <= '{Time(end_time).isot}Z'
        GROUP BY time(1s) fill(none)
    """

    df = await client.influx_client.query(query)

    return pd.DataFrame(df)


async def map_faults_to_elevation(t_end: Time, client: EfdClient) -> float:
    """
    Map M1M3 faults to the TMA elevation at the end time.
    
    Parameters
    ----------
    t_end : Time
        End time for the query.
    client : EfdClient
        EFD client to use for querying elevation data.
        
    Returns
    -------
    float
        Elevation at the end time, or None if no data is available.
    """
    t_start = t_end - pd.Timedelta(seconds=1)
    df = await query_tma_elevation(t_start, t_end, client)

    if df.empty:
        return None

    return df['elevation'].iloc[0]


async def add_elevation_to_df(df: pd.DataFrame, client: EfdClient) -> pd.DataFrame:
    """
    Add elevation data to the DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing M1M3 faults.
    client : EfdClient
        EFD client to use for querying elevation data.

    Returns
    -------
    pd.DataFrame
        DataFrame with elevation data added.
    """
    timestamps = df.index.to_list()
    tasks = [map_faults_to_elevation(ts, client) for ts in timestamps]
    elevations = await asyncio.gather(*tasks)

    df = df.copy()
    df["elevation"] = elevations
    return df


def plot_faults_vs_elevation(df: pd.DataFrame):
    """
    Plot M1M3 faults against TMA elevation.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing M1M3 faults and elevation data.
    """
    d_start = getDayObsForTime(Time(df.index[0]))
    d_end = getDayObsForTime(Time(df.index[-1]))
    
    mask_error = df['level'] == ERROR
    mask_critical = df['level'] == CRITICAL
    
    fig, axs = plt.subplots(2, 1, figsize=(12, 6), sharex=False)
    
    # The first axis contains a histogram of the elevation values when 
    # the faults occurred.
    axs[0].hist(df['elevation'][mask_error], bins=30, alpha=0.5, 
                color='red', label='Error Elevation', log=True)
    axs[0].hist(df['elevation'][mask_critical], bins=30,
                alpha=0.5, color='black', label='Critical Elevation', log=True)
    
    axs[0].set_title(f'Data from {d_start} to {d_end}')
    axs[0].set_xlabel('Fault Occurrences')
    axs[0].set_ylabel('Count')
    axs[0].legend()

    # The second axis contains a time plot showing when the faults occurred
    # and the TMA elevation at that time.
    axs[1].scatter(df.index[mask_error], df['elevation'][mask_error], 
                   color='red', label='Error', s=15)
    axs[1].scatter(df.index[mask_critical], df['elevation'][mask_critical], 
                   color='black', label='Critical', s=15)

    axs[1].set_xlabel('Time (UTC)')
    axs[1].set_ylabel('TMA Elevation (degrees)')
    axs[1].grid(":", alpha=0.3)
    
    fig.suptitle('M1M3 Faults vs TMA Elevation', fontsize=16)

    fig.tight_layout()
    plt.show()
    
    return fig 
