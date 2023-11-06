import numpy as np
import pandas as pd

from lsst.summit.utils.efdUtils import getEfdData
from lsst.ts.xml.tables.m1m3 import HP_COUNT

MEASURED_FORCES_TOPICS = [f"measuredForce{i}" for i in range(HP_COUNT)]


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
        .resample(resample_frequency)
        .interpolate("linear")
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
        .resample(resample_frequency)
        .interpolate("linear")
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
        .resample(resample_frequency)
        .interpolate("linear")
    )

    # Quick and dirty solution for an issue with the client
    cols = ["timestamp"] + MEASURED_FORCES_TOPICS

    hp_data = []
    for col in cols:
        temp_df = (
            getEfdData(
                client,
                "lsst.sal.MTM1M3.hardpointActuatorData",
                columns=col,
                begin=begin,
                end=end,
                prePadding=0,
                postPadding=0,
                warn=False,
            )
            .resample(resample_frequency)
            .interpolate("linear")
        )
        hp_data.append(temp_df)

    query_dict["hp"] = pd.concat(hp_data, axis=1)

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
