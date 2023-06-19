"""
Currently this script tries to identify oscillation events in the 
'lsst.sal.MTM1M3.hardpointActuatorData' measuredForces

For now set the start_date, end_date and window (seconds) and the script 
will break up EFD queries search for events and save the results in a `./data` 
directory. 
"""import argparse
import asyncio
import os
import sys

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from astropy.time import Time, TimeDelta
from scipy.interpolate import UnivariateSpline

from lsst_efd_client import EfdClient

from scipy.signal import find_peaks
from scipy.interpolate import interp1d

class identify_oscillation_events():
    def __init__(self):
        self.force="two"
        # self.output_dir=output_dir
        # self.smoothingFactor=0.2 # In spline creation
        # self.kernel_size=100 # In convolution
        # self.buffer = 4
        # self.return_meas=True
        # self.day=self.start_time.iso[:10]
        # self.fit_method=fit_method # either splines or savgol
        # self.window = 400 # savgol window size
    def add_timestamp(self,data):
        # add correct timestamp column in utc
        if "snd_timestamp_utc" not in data.columns:
            data["snd_timestamp_utc"]=Time(data["private_sndStamp"], format = "unix_tai").unix
        return data
    def combine_peaks_accross_actuators(self, peak_frame, window=4):
        # takes a set of identified peaks in different actuators and associates nearby 
        # ones default window is 4 seconds, 
        # returns 1 peak per window and the max height of peaks in that window as well as the number of 
        # actuators that saw that peak, only returns peaks seen with more than 3 actuators.
        super_heights =[]
        super_times = []
        super_counts = [] 
        super_actuators= []
        for peak in peak_frame["times"].values:
            sel = (abs(peak_frame["times"]-peak) < window)
            subframe=peak_frame[sel]
            count=len(np.unique(subframe["actuators"]))
            if count > 3:
                max_height=subframe["heights"].max()
                super_counts.append(count)
                super_heights.append(max_height)
                super_times.append(subframe["times"][subframe["heights"]==max_height].values)
                super_actuators.append(subframe["actuators"][subframe["heights"]==max_height].values)
        super_times = np.concatenate(super_times)
        super_actuators = np.concatenate(super_actuators)
        super_heights=np.array(super_heights)
        super_counts=np.array(super_counts)
        super_times, super_inds=np.unique(super_times, return_index=True)
        super_heights=super_heights[super_inds]
        super_actuators=super_actuators[super_inds]
        super_counts=super_counts[super_inds]
        return pd.DataFrame({"times":super_times,"heights":super_heights,"counts":super_counts, "actuators":super_actuators})

    async def get_data(self):
            "Extract all the MTMount data from the EFD and save to parquet files"

            # Get EFD client options are usdf_efd or summit_efd
            
            
            client = EfdClient('usdf_efd')

            self.query_dict={}
            self.query_dict["el"] = await client.select_time_series('lsst.sal.MTMount.elevation', \
                                        ['*'],  
                                        self.start_time, self.end_time)
            if  ("private_sndStamp" not in self.query_dict["el"].keys()):
                print("no el data")
                self.query_dict=None
                return
            self.query_dict["el"]=self.add_timestamp(self.query_dict["el"])
            

            self.query_dict["hpmf"] = await client.select_time_series('lsst.sal.MTM1M3.hardpointActuatorData',
                                                    ['private_sndStamp',
                                                    'measuredForce0', 
                                                    'measuredForce1', 
                                                    'measuredForce2', 
                                                    'measuredForce3',
                                                    'measuredForce4', 
                                                    'measuredForce5'],  
                                                    self.start_time, self.end_time)
            if  ("private_sndStamp" not in self.query_dict["hpmf"].keys()):
                print("no hpmf data")
                self.query_dict=None
                return
            self.query_dict["hpmf"] = self.add_timestamp(self.query_dict["hpmf"])            
            
    async def run(self, start_time, end_time):
        # given hp data iterate over all and create a dict with identified peaks as well as their height
        
        self.start_time=start_time
        self.end_time=end_time
        # make query
        await self.get_data()
        if self.query_dict is None:
            return None
        
        rolling_std_window = 100  # 100 is ~ 2 second window
        association_window_1 = 2 # window in seconds to combine peaks in same actuator
        association_window_2 = 4 # window in seconds to combine peaks accross actuators
        slew_speed_min = 0.01 # used for identifiying when we are slewing

        peak_dict={}
        peak_frame=pd.DataFrame({"times":[],"heights":[],"actuators":[]})


        for i in range(6):
            # this loop identifies rolling std peaks in the measured force
            rolling_std_val=self.query_dict["hpmf"][f"measuredForce{i}"].rolling(rolling_std_window).std() # 100 is ~ 2 second window
            peak_indicies=find_peaks(rolling_std_val, height=50)[0] 
            
            # keep time and height of peaks
            peak_dict[f"hp_{i}_peak_times"]=self.query_dict["hpmf"]["snd_timestamp_utc"][peak_indicies].values
            peak_dict[f"hp_{i}_peak_heights"]= rolling_std_val[peak_indicies].values

            # for each peak combine by looking at all peaks within 
            # a window and keeping the one with the largest height then np.unique that 
            super_heights=[]
            super_times=[]
            for j,peak in enumerate(peak_dict[f"hp_{i}_peak_times"]):
                sel_peaks=(abs(peak_dict[f"hp_{i}_peak_times"]-peak) < association_window_1)
                max_height=np.max(peak_dict[f"hp_{i}_peak_heights"][sel_peaks])
                max_time=peak_dict[f"hp_{i}_peak_times"][sel_peaks][np.where(peak_dict[f"hp_{i}_peak_heights"][sel_peaks]==max_height)]
                max_index=np.where(peak_dict[f"hp_{i}_peak_times"]==max_time)[0]
                super_times.append(peak_dict[f"hp_{i}_peak_times"][max_index])
                super_heights.append(peak_dict[f"hp_{i}_peak_heights"][max_index])
            peak_dict[f"hp_{i}_peak_times"] = np.unique(super_times)
            peak_dict[f"hp_{i}_peak_heights"] = np.unique(super_heights)
            peak_frame=pd.concat([peak_frame,pd.DataFrame({"times":peak_dict[f"hp_{i}_peak_times"],
                                                        "heights":peak_dict[f"hp_{i}_peak_heights"],
                                                        "actuators":i})])
        peak_frame=peak_frame.sort_values("times")

        # next we want to combine peaks across actuators
        overall_frame=self.combine_peaks_accross_actuators(peak_frame, window=association_window_2)

        #identify when we are slewing
        overall_frame["slew_state"]=False
        slew_speed=interp1d(self.query_dict["el"]["snd_timestamp_utc"],
                            abs(self.query_dict["el"]["actualVelocity"].rolling(10).mean()),
                            bounds_error=False)
        slew_velocity=interp1d(self.query_dict["el"]["snd_timestamp_utc"],
                               (self.query_dict["el"]["actualVelocity"].rolling(10).mean()),
                               bounds_error=False)
        slew_position=interp1d(self.query_dict["el"]["snd_timestamp_utc"],
                               (self.query_dict["el"]["actualPosition"].rolling(10).mean()),
                               bounds_error=False)
        
        sel=(slew_speed(overall_frame["times"]) > slew_speed_min)
        overall_frame.loc[sel,"slew_state"]=True
        overall_frame["elevation_velocity"]=slew_velocity(overall_frame["times"])
        overall_frame["elevation_position"]=slew_position(overall_frame["times"])
        overall_frame=overall_frame.loc[overall_frame["slew_state"]==True,:]
        return overall_frame
    
if __name__ == '__main__':
    
    if not os.path.exists("./data/"):
        os.makedirs("./data/")
    
    start_date = Time("2023-06-03T14:01:00", scale='utc')
    end_date =Time("2023-06-19T14:01:00", scale='utc')
    window = TimeDelta(24*60*60, format = 'sec')
    id_oscillations=identify_oscillation_events()
    
    start_time=start_date
    while start_time < end_date:
        end_time=start_time + window
        save_string=f"./data/oscillation_events_{start_time}_to_{end_time}.csv"
        print(f"starting query for {start_time} to {end_time} ")
        
    #start_date = Time("2023-06-16T00:00:00", scale='utc')
    
        overall_frame=asyncio.run(id_oscillations.run(start_time,end_time)) 
        if overall_frame is not None:
            overall_frame.to_csv(save_string)
            print("finished")
        else:
            print(f"no data for {save_string} ")
        start_time=end_time
        
        
    #print(overall_frame)