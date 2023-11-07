import numpy as np
import pandas as pd
import h5py
from scipy.interpolate import interp1d
from scipy.signal import find_peaks
from lsst.summit.utils.efdUtils import getEfdData
from lsst.ts.xml.tables.m1m3 import HP_COUNT

MEASURED_FORCES_TOPICS = [f"measuredForce{i}" for i in range(HP_COUNT)]

# Functions to get data
key_m1m3_dict = {
    "1 X": "m1m3_x_1",
    "1 Y": "m1m3_y_1",
    "1 Z": "m1m3_z_1",
    "2 X": "m1m3_x_2",
    "2 Y": "m1m3_z_2",  # note these two have been
    "2 Z": "m1m3_y_2",  # switched pending SUMMIT-7911
    "3 X": "m1m3_x_3",
    "3 Y": "m1m3_y_3",
    "3 Z": "m1m3_z_3",
}
key_m2_dict = {
    "1 X": "m2_x_1",
    "1 Y": "m2_y_1",
    "1 Z": "m2_z_1",
    "2 X": "m2_x_2",
    "2 Y": "m2_z_2",
    "2 Z": "m2_y_2",
    "3 X": "m2_x_3",
    "3 Y": "m2_z_3",
    "3 Z": "m2_y_3",
    "4 X": "m2_x_4",
    "4 Y": "m2_y_4",
    "4 Z": "m2_z_4",
    "5 X": "m2_x_5",
    "5 Y": "m2_z_5",
    "5 Z": "m2_y_5",
    "6 X": "m2_x_6",
    "6 Y": "m2_z_6",
    "6 Z": "m2_y_6",
}


def vms_data_to_pandas(filename, vms_type, begin_time=None, end_time=None):
    """
    Converts VMS data in the given HDF5 file to a Pandas DataFrame.

    Args:
    filename: Path to the HDF5 file containing the VMS data.
    vms_type: The type of VMS data in the file. Must be "m1m3", "m2", or
      "rotator".
    begin_time: The start time of the data to include in the DataFrame. If None,
      all data will be included.
    end_time: The end time of the data to include in the DataFrame. If None, all
      data will be included.

    Returns:
    A Pandas DataFrame containing the VMS data.
    """
    if vms_type == "m1m3":
        key_dict = key_m1m3_dict
    elif vms_type == "m2":
        key_dict = key_m2_dict
    elif vms_type == "rotator":
        raise NotImplementedError
    else:
        raise ValueError("vms_type must be m1m3,m2, or rotator")

    f = h5py.File(filename, "r")
    times = f["timestamp"][::1]
    dkeys = "XYZ"

    data_dict = {}
    if (begin_time is not None) & (end_time is not None):
        sel = (times > begin_time) & (times < end_time)
    else:
        sel = np.ones(times.size).astype(bool)
    data_dict["times"] = times[sel]
    for key in key_dict.keys():
        data_dict[key_dict[key]] = f[key][::1][sel]
    data_frame = pd.DataFrame(data_dict)
    for j in np.arange(int(len(key_dict) / 3)) + 1:
        data_frame[f"total_{j}"] = np.linalg.norm(
            data_frame[[f"{vms_type}_{i}_{j}" for i in ["x", "y", "z"]]].values, axis=1
        )

    return data_frame


def get_efd_data(begin, end, client, resample_frequency="10ms"):
    """
    Extract all the MTMount data from the EFD and add to dict.

    Parameters
    ==========
    begin : str
        The start time of the query.
    end : str
        The end time of the query.
    client : object
        influx client
    resample_frequency : str
        Frequency used to resample the data.
        See ``pandas.core.resample.Resampler.interpolate`` for more information.

    Returns:
        dict: A dictionary containing the MTMount data.
    """

    query_dict = {}

    query_dict["el"] = (
        getEfdData(
            client,
            "lsst.sal.MTMount.elevation",
            columns=["*"],
            begin=begin,
            end=end,
            prePadding=0,
            postPadding=0,
            warn=False,
        )
        # .resample(resample_frequency)
        # .interpolate("linear")
    )

    query_dict["az"] = (
        getEfdData(
            client,
            "lsst.sal.MTMount.azimuth",
            columns=["*"],
            begin=begin,
            end=end,
            prePadding=0,
            postPadding=0,
            warn=False,
        )
        # .resample(resample_frequency)
        # .interpolate("linear")
    )

    query_dict["ims"] = (
        getEfdData(
            client,
            "lsst.sal.MTM1M3.imsData",
            columns=["*"],
            begin=begin,
            end=end,
            prePadding=0,
            postPadding=0,
            warn=False,
        )
        # .resample(resample_frequency)
        # .interpolate("linear")
    )

    # Quick and dirty solution for an issue with the client
    cols = ["timestamp"] + MEASURED_FORCES_TOPICS

    query_dict["hp"] = (
        getEfdData(
            client,
            "lsst.sal.MTM1M3.hardpointActuatorData",
            columns=cols,
            begin=begin,
            end=end,
            prePadding=0,
            postPadding=0,
            warn=False,
        )
        # .resample(resample_frequency)
        # .interpolate("linear")
    )

    return query_dict


def get_freq_psd(vals, timestep):
    """
    Calculates the frequency power spectrum of a signal.

    Parameters
    ==========
    vals : np.array
        The signal values.
    timestep : float
        The time step between samples.

    Returns
    =======
    frequencies, psd : tuple
        The frequencies and power spectral density.
    """
    # Remove the mean from the signal.
    meanval = np.mean(vals)
    signal = vals - meanval

    # Calculate the length of the signal.
    N = len(signal)

    # Calculate the power spectral density.
    psd = np.abs(np.fft.rfft(np.array(signal) * 1)) ** 2

    # Calculate the frequencies.
    frequencies = np.fft.rfftfreq(N, timestep)

    return (frequencies, psd)


def get_peak_points(freq, psd, height=0.01):
    """
    Get the peak points of the power spectral density (PSD).

    Args:
        freq (numpy.ndarray): The frequency vector.
        psd (numpy.ndarray): The power spectral density.
        height (float): The minimum peak height.

    Returns:
        numpy.ndarray: The peak points.
    """

    # Find the peak indices and heights.
    peak_ind, peak_dict = find_peaks(psd, height=height)
    peaks = freq[peak_ind]

    # If there are no peaks, return None.
    if len(peaks) < 1:
        return None

    # Find the sub-peaks within each group of peaks that are close in frequency.
    points = []
    for i, peak in enumerate(peaks):
        sel = abs(peaks - peak) < 1
        sub_peaks = peaks[sel]
        sub_heights = peak_dict["peak_heights"][sel]
        points.append(sub_peaks[np.argmax(sub_heights)])

    # Return the unique peak points.
    return np.unique(np.array(points))


def resample_times(timestamp, yval, new_delta_t):
    new_x = np.arange(timestamp.min(), timestamp.max() + new_delta_t, new_delta_t)
    interp_y = interp1d(timestamp, yval, bounds_error=False)
    new_y = interp_y(new_x)
    sel = ~np.isnan(new_y)
    return new_x[sel], new_y[sel]
