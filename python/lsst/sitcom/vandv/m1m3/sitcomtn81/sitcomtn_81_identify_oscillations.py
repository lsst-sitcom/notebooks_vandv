"""
Currently this script tries to identify oscillation events in the
'lsst.sal.MTM1M3.hardpointActuatorData' measuredForces

For now set the start_date, end_date and window (seconds) and the script
will break up EFD queries search for events and save the results in a `./data`
directory.
"""

import asyncio
import os

import pandas as pd
import numpy as np
from astropy.time import Time
import math

from lsst_efd_client import EfdClient
from lsst.summit.utils.tmaUtils import TMAEventMaker, TMAState
from lsst.summit.utils.efdUtils import getEfdData, calcNextDay


from scipy.signal import find_peaks
from scipy.interpolate import interp1d
from tqdm import tqdm

def clipDataToEvent(df, event, padding):
    """Clip a padded dataframe to an event.

    Parameters
    ----------
    df : `pandas.DataFrame`
        The dataframe to clip.
    event : `lsst.summit.utils.efdUtils.TmaEvent`
        The event to clip to.

    Returns
    -------
    clipped : `pandas.DataFrame`
        The clipped dataframe.
    """
    mask = (df["private_sndStamp"] >= event.begin.value - padding) & (
        df["private_sndStamp"] <= event.end.value + padding
    )
    clipped_df = df.loc[mask].copy()
    return clipped_df

class IdentifyOscillationEvents:
    """
    A class used to identify oscillation events in mechanical systems,
    specifically targeting actuators.

    Attributes
    ----------
    force : str
        The force mode of the system, default is set to "two".
    rolling_std_window : int
        The window size for calculating the rolling standard deviation,
        default is 100 (approximately 2 seconds).
    association_window_1 : int
        The time window in seconds to combine peaks in the same actuator,
        default is 2 seconds.
    association_window_2 : int
        The time window in seconds to combine peaks across actuators,
        default is 4 seconds.
    slew_speed_min : float
        The minimum slew speed used for identifying when the system is slewing,
        default is 0.01.
    peak_height : int
        The threshold height for identifying peaks, default is 100.

    Methods
    -------
    add_timestamp(data)
        Adds a correct timestamp column in UTC to the given data.

    combine_peaks_accross_actuators(peak_frame, window=4)
        Combines identified peaks in different actuators within a
        specified window and processes them.

    get_slews(day_obs)
        Asynchronously retrieves slewing events for a given observation day.

    get_data(event, client)
        Asynchronously extracts and processes MTMount data from the EFD for
        a given event.

    identify(query_dict)
        Identifies oscillation events based on the processed query data.

    run(dayObs)
        Asynchronously processes data for a given observation day and
        identifies oscillation events.
    """

    def __init__(self, efd_instance="idf_efd"):
        self.force = "two"
        self.rolling_std_window = 100  # 100 is ~ 2 second window
        self.association_window_1 = (
            6  # window in seconds to combine peaks in same actuator
        )
        self.association_window_2 = (
            7  # window in seconds to combine peaks accross actuators
        )
        self.slew_speed_min = 0.01  # used for identifiying when we are slewing
        self.peak_height = 50
        self.efd_instance = efd_instance
        self.client = EfdClient(self.efd_instance)

    def add_timestamp(self, data):
        """
        Adds a correct timestamp column in UTC format to the provided data if
        not present.

        Parameters
        ----------
        data : DataFrame
            The data to which the timestamp will be added.

        Returns
        -------
        DataFrame
            The data with the added 'snd_timestamp_utc' column.
        """
        if "snd_timestamp_utc" not in data.columns:
            data["snd_timestamp_utc"] = Time(
                data["private_sndStamp"], format="unix_tai"
            ).unix
        return data

    def combine_peaks_accross_actuators(self, peak_frame, window=4):
        """
        Takes a set of identified peaks in different actuators and associates
        nearby ones within a ``window`` seconds.

        Parameters
        ----------
        peak_frame : DataFrame
            The DataFrame containing peak data from various actuators.
        window : int, optional
            The time window in seconds to combine peaks across actuators
            (default is 4).

        Returns
        -------
        DataFrame
            A DataFrame containing combined peak information with maximum
            heights and the number of actuators that saw that peak.
        """
        super_heights = []
        super_times = []
        super_counts = []
        super_actuators = []
        super_rmean = []
        for peak in peak_frame["times"].values:
            sel = abs(peak_frame["times"] - peak) < window
            subframe = peak_frame[sel]
            count = len(np.unique(subframe["actuators"]))
            if count > 3:
                max_height = subframe["heights"].max()
                super_counts.append(count)
                super_heights.append(max_height)
                super_times.append(
                    subframe["times"][subframe["heights"] == max_height].values
                )
                super_actuators.append(
                    subframe["actuators"][subframe["heights"] == max_height].values
                )
                super_rmean.append(
                    subframe["rmean_diff"][subframe["heights"] == max_height].values
                )
        if len(super_times) == 0:
            return pd.DataFrame(
                {
                    "times": super_times,
                    "heights": super_heights,
                    "rmean_diff": super_rmean,
                    "counts": super_counts,
                    "actuators": super_actuators,
                }
            )
        super_times = np.concatenate(super_times)
        super_actuators = np.concatenate(super_actuators)
        super_rmean = np.concatenate(super_rmean)
        super_heights = np.array(super_heights)
        super_counts = np.array(super_counts)

        super_times, super_inds = np.unique(super_times, return_index=True)
        super_heights = super_heights[super_inds]
        super_actuators = super_actuators[super_inds]
        super_counts = super_counts[super_inds]
        super_rmean = super_rmean[super_inds]

        return pd.DataFrame(
            {
                "times": super_times,
                "heights": super_heights,
                "rmean_diff": super_rmean,
                "counts": super_counts,
                "actuators": super_actuators,
            }
        )

    async def get_slews(self, day_obs):
        """
        Asynchronously retrieves slewing events for a given observation day.

        Parameters
        ----------
        day_obs : int
            The observation day for which slewing events are to be retrieved.

        Returns
        -------
        list
            A list of slewing events.
        """
        eventMaker = TMAEventMaker(self.client)
        events = eventMaker.getEvents(int(day_obs),)
        slews = [e for e in events if e.type == TMAState.SLEWING]
        return slews

    async def get_data(self, begin, end, client):
        """
        Asynchronously extracts and processes MTMount data from the EFD for a
        given event.

        Parameters
        ----------
        event : EventObject
            The event object containing event data.
        client : EFDClient
            The EFD client used to extract data.
        """

        # Get EFD client options are usdf_efd or summit_efd

        # Get EFD client options are usdf_efd or summit_efd

        query_dict = {}
        query_dict["day_obs"] = self.day_obs
        
        query_dict["hpmf"] = getEfdData(
            client,
            "lsst.sal.MTM1M3.hardpointActuatorData",
            begin=begin,
            end=end,
            prePadding=5,
            postPadding=5,
            columns=[
                "private_sndStamp",
                "measuredForce0",
                "measuredForce1",
                "measuredForce2",
                "measuredForce3",
                "measuredForce4",
                "measuredForce5",
            ],
            warn=False,
        )
        if "private_sndStamp" not in query_dict["hpmf"].keys():
            print("no hpmf data")
            query_dict = None
            return
        query_dict["hpmf"] = self.add_timestamp(query_dict["hpmf"])

        query_dict["el"] = getEfdData(
            client,
            "lsst.sal.MTMount.elevation",
            columns=["private_sndStamp", "actualPosition", "actualVelocity"],
            begin=begin,
            end=end,
            prePadding=10,
            postPadding=10,
            warn=False,
        )

        if "private_sndStamp" not in query_dict["el"].keys():
            print("no el data")
            query_dict = None
            return
        query_dict["el"] = self.add_timestamp(query_dict["el"])

        query_dict["az"] = getEfdData(
            client,
            "lsst.sal.MTMount.azimuth",
            columns=["private_sndStamp", "actualPosition", "actualVelocity"],
            begin=begin,
            end=end,
            prePadding=5,
            postPadding=5,
            warn=False,
        )

        if "private_sndStamp" not in query_dict["az"].keys():
            print("no az data")
            query_dict = None
            return
        query_dict["az"] = self.add_timestamp(query_dict["az"])
        return query_dict

    def get_single_slew_data_dict(self, event):
        slew_dict = {}
        for key in self.query_dict.keys():
            if isinstance(self.query_dict[key], pd.DataFrame):
                slew_dict[key] = clipDataToEvent(
                    self.query_dict[key], event, padding=10
                )
            else:
                slew_dict[key] = self.query_dict[key]
        slew_dict["seq_num"] = event.seqNum
        for key in self.query_dict.keys():
            if key != "day_obs":
                if len(slew_dict[key]) < 1:
                    return None
        return slew_dict

    def identify(self, data_dict):
        """
        Identifies oscillation events based on the processed query data.

        Parameters
        ----------
        data_dict : dict
            The dictionary containing query data for oscillation event
            identification.

        Returns
        -------
        DataFrame or None
            A DataFrame containing identified oscillation events,
            or None if no events are identified.
        """
        if data_dict is None:
            return None

        peak_dict = {}
        peak_frame = pd.DataFrame({"times": [], 
                                   "heights": [], 
                                   "actuators": [],
                                   "rmean_diff":[]})
        for i in range(6):
            # this loop identifies rolling std peaks in the measured force
            rolling_std_val = (
                data_dict["hpmf"][f"measuredForce{i}"]
                .rolling(self.rolling_std_window)
                .std()
            )  # 100 is ~ 2 second window
            rolling_mean_val = (
                data_dict["hpmf"][f"measuredForce{i}"]
                .rolling(self.rolling_std_window)
                .mean()
            )
            peak_indicies = find_peaks(rolling_std_val, height=self.peak_height)[0]

            # keep time and height of peaks
            peak_dict[f"hp_{i}_peak_times"] = data_dict["hpmf"]["snd_timestamp_utc"][
                peak_indicies
            ].values
            peak_dict[f"hp_{i}_peak_heights"] = rolling_std_val[peak_indicies].values
            start_ind = [np.max([i - 500, 0]) for i in peak_indicies]
            stop_ind = [
                np.min([i + 500, len(rolling_mean_val) - 1]) for i in peak_indicies
            ]
            peak_dict[f"hp_{i}_peak_rmean_diff"] = (
                rolling_mean_val[stop_ind].values - rolling_mean_val[start_ind].values
            )

            # for each peak combine by looking at all peaks within
            # a window and keeping the one with the largest height then np.unique that
            super_heights = []
            super_times = []
            super_rmean = []

            for j, peak in enumerate(peak_dict[f"hp_{i}_peak_times"]):
                sel_peaks = (
                    abs(peak_dict[f"hp_{i}_peak_times"] - peak)
                    < self.association_window_1
                )
                max_height = np.max(peak_dict[f"hp_{i}_peak_heights"][sel_peaks])
                # max_rmean=np.max(peak_dict[f"hp_{i}_peak_long_mean"][sel_peaks])
                max_time = peak_dict[f"hp_{i}_peak_times"][sel_peaks][
                    np.where(peak_dict[f"hp_{i}_peak_heights"][sel_peaks] == max_height)
                ]
                max_index = np.where(peak_dict[f"hp_{i}_peak_times"] == max_time)[0]
                super_times.append(peak_dict[f"hp_{i}_peak_times"][max_index])
                super_heights.append(peak_dict[f"hp_{i}_peak_heights"][max_index])
                super_rmean.append(peak_dict[f"hp_{i}_peak_rmean_diff"][max_index])
            if len(super_times) > 0:

                peak_dict[f"hp_{i}_peak_times"], time_index = np.unique(np.concatenate(super_times), return_index=True)
                peak_dict[f"hp_{i}_peak_heights"] = np.concatenate(super_heights)[time_index]
                peak_dict[f"hp_{i}_peak_rmean_diff"] = np.concatenate(super_rmean)[time_index]
                try:
                    new_frame=pd.DataFrame(
                                {
                                    "times": peak_dict[f"hp_{i}_peak_times"],
                                    "heights": peak_dict[f"hp_{i}_peak_heights"],
                                    "rmean_diff": peak_dict[f"hp_{i}_peak_rmean_diff"],
                                    "actuators": i,
                                }
                            )
                except:
                    import pdb; pdb.set_trace()
                peak_frame = pd.concat(
                    [
                        peak_frame,
                        new_frame,
                    ], 
                )
        peak_frame = peak_frame.sort_values("times")

        # next we want to combine peaks across actuators
        overall_frame = self.combine_peaks_accross_actuators(
            peak_frame, window=self.association_window_2
        )

        # identify when we are slewing
        overall_frame["slew_state"] = False
        slew_speed_el = interp1d(
            data_dict["el"]["snd_timestamp_utc"],
            abs(data_dict["el"]["actualVelocity"].rolling(10).mean()),
            bounds_error=False,
        )
        slew_speed_az = interp1d(
            data_dict["az"]["snd_timestamp_utc"],
            abs(data_dict["az"]["actualVelocity"].rolling(10).mean()),
            bounds_error=False,
        )

        slew_velocity_el = interp1d(
            data_dict["el"]["snd_timestamp_utc"],
            (data_dict["el"]["actualVelocity"].rolling(10).mean()),
            bounds_error=False,
        )
        slew_velocity_az = interp1d(
            data_dict["az"]["snd_timestamp_utc"],
            (data_dict["az"]["actualVelocity"].rolling(10).mean()),
            bounds_error=False,
        )

        slew_position = interp1d(
            data_dict["el"]["snd_timestamp_utc"],
            (data_dict["el"]["actualPosition"].rolling(10).mean()),
            bounds_error=False,
        )

        sel = slew_speed_el(overall_frame["times"]) > self.slew_speed_min
        sel |= slew_speed_az(overall_frame["times"]) > self.slew_speed_min
        overall_frame.loc[sel, "slew_state"] = True
        overall_frame["elevation_velocity"] = slew_velocity_el(overall_frame["times"])
        overall_frame["azimuth_velocity"] = slew_velocity_az(overall_frame["times"])
        overall_frame["elevation_position"] = slew_position(overall_frame["times"])
        overall_frame = overall_frame.loc[overall_frame["slew_state"] == True, :]

        if len(overall_frame) > 0:
            overall_frame["seq_num"] = data_dict["seq_num"]
            overall_frame["day_obs"] = self.day_obs
            return overall_frame
        else:
            return None

    async def run(self, day_obs):
        """
        Asynchronously processes data for a given observation day and identifies
        potential oscillation events.

        Parameters
        ----------
        dayObs : int
            The observation day for which data is processed and events are identified.

        Returns
        -------
        DataFrame or None
            A DataFrame containing all identified events for the given day, or None if no events
            are identified.
        """
        
        self.day_obs = day_obs
        self.slews = await self.get_slews(day_obs)
        if len(self.slews) == 0:
            print("no data")
            return None
        else:
            print(f"{day_obs} {len(self.slews)} slews identified")
        self.slews = self.slews

        # make query
        if len(self.slews) < 2000:
            self.query_dict = await self.get_data(self.slews[0].begin, self.slews[-1].end, self.client)
        else:
            chunk_size= 1
            # Calculate the number of chunks needed
            nchunks = math.ceil(len(self.slews) / chunk_size)
            
            # Initialize an empty list to store results from each chunk
            chunked_results = []

            # Iterate over the chunks
            print("chunking")
            for i in range(nchunks):
                print(f"chunk {i}")
                # Calculate start and end indices for each chunk
                start_index = i * chunk_size
                end_index = min((i + 1) * chunk_size, len(self.slews))
                print(f"start: {self.slews[start_index].begin}")
                print(f"end: {self.slews[end_index - 1].end}")
                # Get the data for the current chunk and append it to the list
                chunk_data = await self.get_data(self.slews[start_index].begin, self.slews[end_index - 1].end, self.client)
                chunked_results.append(chunk_data)
            
            keys = chunked_results[0].keys()
            self.query_dict = {}
            # Combine the data for each key
            for key in keys:
                if key != "day_obs":
                    combined_data = pd.concat([result[key] for result in query_results])
                    self.query_dict[key] = combined_data
                else:
                    # Assuming 'day_obs' is a list and not a DataFrame
                    self.query_dict["day_obs"] = self.day_obs

                 
        event_list = []

        for slew in tqdm(self.slews):
            slew_dict = self.get_single_slew_data_dict(slew)
            if slew_dict is None:
                continue
            result = self.identify(slew_dict)
            if result is not None:
                event_list.append(result)
        if len(event_list) > 0:
            events_frame = pd.concat(event_list)
            return events_frame
        else:
            empty_frame=pd.DataFrame()
            for key in ['times', 'heights', 'rmean_diff', 'counts', 'actuators', 'slew_state',
       'elevation_velocity', 'azimuth_velocity', 'elevation_position',
       'seq_num', 'day_obs']:
                empty_frame[key]=[]
            return empty_frame


if __name__ == "__main__":
    # want to understand if force actuators are on
    if not os.path.exists("./data/"):
        os.makedirs("./data/")

    begin_day_obs = 20230628
    end_day_obs = 20230628

    id_oscillations = IdentifyOscillationEvents()

    current_day_obs = begin_day_obs
    while int(current_day_obs) <= int(end_day_obs):
        next_day_obs = calcNextDay(current_day_obs)
        print(current_day_obs)
        save_string = f"./data/oscillation_events_{current_day_obs}.csv"
        if os.path.exists(save_string):
            print(f"file exists: {save_string}")
            current_day_obs = next_day_obs
            continue
        oscillation_events_frame = asyncio.run(id_oscillations.run(current_day_obs))
        if oscillation_events_frame is not None:
            oscillation_events_frame.to_csv(save_string)
            print("finished")

        current_day_obs = next_day_obs
