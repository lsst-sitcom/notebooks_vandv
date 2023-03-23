import pandas as pd


def unpack_tma_accel(df, axis, time_format="seconds"):
    """Unpack TMA accelerometer data from given dataframe.

    Parameters
    ----------
    df: pd.DataFrame
        The dataframe of values.
    axis: str
        Must be "X", "Y", or "Z".
    time_format: str
        If "seconds", returns relative time in seconds.
        If "stamps", returns time stamps.

    Returns
    -------
    np.ndarray
        Time of acceleration data.
    np.ndarray
        Accelerations in m/s^2.
    """
    # validate parameters
    if axis not in ["X", "Y", "Z"]:
        raise ValueError(f"axis {axis} invalid; must be 'X', 'Y', or 'Z'.")
    if time_format not in ["seconds", "stamps"]:
        raise ValueError(
            f"time_format {time_format} invalid; must be 'seconds' or 'stamps'."
        )

    # now, let's make sure the timestamps are in order
    df = df.sort_index()

    # pull out the initial time stamp and the intervals
    stamp0 = df.index[0]
    intervals = df.interval.to_numpy()

    # select acceleration columns for the specified axis
    df = df[[col for col in df.columns if axis in col]]

    # rename columns with integer time step
    df = df.rename(columns=lambda col: int(col.split(axis)[1]))

    # put the columns in order
    df = df.sort_index(axis=1)

    # convert index to dt in seconds
    df.index = (df.index - df.index[0]).total_seconds()

    # pull out the times (again, in seconds)
    row_times = df.index.to_numpy()
    column_times = intervals[:, None] * df.columns.to_numpy()[None, :]
    times = row_times[:, None] + column_times

    # finally, extract the times and accelerometer data
    t = times.flatten()
    accel = df.to_numpy().flatten()

    # convert to time stamps?
    if time_format == "stamps":
        t = stamp0 + pd.to_timedelta(t, "s")

    return t, accel
