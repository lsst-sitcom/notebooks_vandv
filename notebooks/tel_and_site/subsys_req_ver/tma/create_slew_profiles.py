# MTMount data extraction script for LSST SITCOM test
# LVV-TXXX:<https://jira.lsstcorp.org/secure/Tests.jspa#/testCase/LVV-TXXX>
# LVV-TXXX tests verification element LVV-YYY, <https://jira.lsstcorp.org/browse/LVV-YYY>
# which verifies requirement LTS-REQ-00ZZ-V-0Z: 2.2.2 Slewing Rates in LTS-103
"""
Extracts MTMount data from the EFD and computes spline profiles for identified 
-m 1 : is method 1 identifying slews based of of efd log events and is
    reccomended
-m 2: is method 2 where slews are identified soley from tma encoder data
slews
    Example usage:
    ```
    Test data range 
    python -W ignore create_slew_profiles.py -d 2023-03-23 -w 10 -m 1
    
    1 night of data (-w controls window in hours)
    python -W ignore create_slew_profiles.py -d 2023-03-23 -w 24 -m 1
    
    ```
"""
import argparse
import asyncio
import os

import astropy.time 
import astropy.units as u
from astropy.time import Time, TimeDelta
from lsst.sitcom import vandv
import numpy as np

from datetime import datetime
import pandas as pd
from matplotlib import pyplot
from lsst_efd_client import EfdClient
from scipy.interpolate import UnivariateSpline
from scipy.signal import find_peaks
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d
import json 


class create_slew_splines():
    def __init__(self,start_time,end_time,force,output_dir, fit_method="splines"):
        self.start_time=start_time
        self.end_time=end_time
        self.force=force
        self.output_dir=output_dir
        self.smoothingFactor=0.2 # In spline creation
        self.kernel_size=100 # In convolution
        self.buffer = 4
        self.return_meas=True
        self.day=self.start_time.iso[:10]
        self.fit_method=fit_method # either splines or savgol
        self.window = 400 # savgol window size
    
    def get_slew_pairs(self,starts,stops):
        """
        Given vectors of start times and stop times take the longer vector 
        and iterate over it. If that is starts for each start time select all stop
        times that are > than the start time and < than the next start time. 
        If multiple stops are detected select the minimum one. Also keeps track and
        returns the unmatched start and stop times. 
        """
        new_starts=[]
        new_stops=[]
        unmatched_stops=[]
        unmatched_starts=[]

        if len(stops) <= len(starts):
            for i in range(len(starts)):
                if i == len(starts)-1:
                    stop_sel=(stops > starts[i])
                else: 
                    stop_sel=(stops > starts[i]) & (stops < starts[i+1])
                if stop_sel.sum()==1:
                    new_stops.append(stops[stop_sel][0])
                    new_starts.append(starts[i])
                if stop_sel.sum() > 1:
                    new_stops.append(np.min(stops[stop_sel]))
                    new_starts.append(starts[i])
                    for j in stops[stop_sel]: 
                        if j != np.min(stops[stop_sel]):
                            unmatched_stops.append(j)
                if stop_sel.sum() == 0 :
                    unmatched_starts.append(starts[i])


        if len(stops) > len(starts):
            for i in range(len(stops)):
                if i == 0:
                    start_sel=(starts < stops[0]) & (starts > 0)
                else: 
                    start_sel=(starts < stops[i]) & (starts > stops[i-1])
                if start_sel.sum()==1:
                    new_stops.append(stops[i])
                    new_starts.append(starts[start_sel][0])
                if start_sel.sum() > 1:
                    new_stops.append(stops[i])
                    new_starts.append(np.max(starts[start_sel]))
                    for j in starts[start_sel]: 
                        if j != np.max(starts[start_sel]):
                            unmatched_starts.append(j)
                if start_sel.sum() == 0 :
                    unmatched_stops.append(stops[i])


        return new_starts, new_stops, unmatched_starts, unmatched_stops
    
    def savgolFilter(self, times, positions,interpPoints, window=200, deriv=1, smoothingFactor = 0.01): 
            positionSpline = UnivariateSpline(times, positions, s=smoothingFactor)(interpPoints) 
            derivativePoints = savgol_filter(positionSpline, window_length=window, mode="mirror",
                                             deriv=deriv, polyorder=3, delta=(interpPoints[1]-interpPoints[0])) 
            return derivativePoints#interp1d(times,derivativePoints)(interpPoints)

    def get_savgol_splines(self,times, positions, interpPoints):
        posSpline = UnivariateSpline(times, positions, s=0)(interpPoints)
        velSpline = self.savgolFilter(times, positions, interpPoints, window=self.window, deriv=1, smoothingFactor=self.smoothingFactor)
        accSpline = self.savgolFilter(times, positions, interpPoints, window=self.window, deriv=2, smoothingFactor=self.smoothingFactor)
        jerkSpline = self.savgolFilter(times, positions, interpPoints, window=self.window, deriv=3, smoothingFactor=self.smoothingFactor)
        return posSpline, velSpline, accSpline, jerkSpline
        
    def get_univariate_splines(self, times, positions, velocities, interpPoints, kernel):
        """
        """
        try:
            posSpline = UnivariateSpline(times, position, s=0)
        except:
            #if there are duplicate time measurements remove them  (this occured on 
            # 23/11/22-23 and 21-22)
            times, indexes=np.unique(times, return_index=True)
            positions=positions[indexes]
            velocities=velocities[indexes]
            
            posSpline = UnivariateSpline(times, positions, s=0)
            velSpline1  = UnivariateSpline(times, velocities, s=0) 
            
        # Now smooth the derivative before differentiating again
        smoothedVel = np.convolve(velSpline1(interpPoints), kernel, mode='same')
        velSpline = UnivariateSpline(interpPoints, smoothedVel, s=self.smoothingFactor)
        accSpline1 = velSpline.derivative(n=1)
        smoothedAcc = np.convolve(accSpline1(interpPoints), kernel, mode='same')
        # Now smooth the derivative before differentiating again
        accSpline = UnivariateSpline(interpPoints, smoothedAcc, s=self.smoothingFactor)
        jerkSpline = accSpline.derivative(n=1)
        return posSpline(interpPoints), velSpline(interpPoints), accSpline(interpPoints), jerkSpline(interpPoints)
         
    
    def fit_slew_profiles(self,index):
        """
        givien identified slews and encoder data, splines for Velocity,
        Acceleration and Jerk are calculated. 
        
        kernel_size: size of smoothing kernel
        buffer: buffer in seconds to add onto the slew time
        smoothingFactor: used in spline creation
        
        """
        kernel=np.ones(self.kernel_size)/self.kernel_size
        selAz=(self.query_dict["az"]['timestamp'] > (self.slew_dict["slew_start_times"][index] - self.buffer)) 
        selAz&=(self.query_dict["az"]['timestamp'] < (self.slew_dict["slew_stop_times"][index] + self.buffer)) 
        
        selEl= (self.query_dict["el"]['timestamp'] > (self.slew_dict["slew_start_times"][index] - self.buffer)) 
        selEl&= (self.query_dict["el"]['timestamp'] < (self.slew_dict["slew_stop_times"][index] + self.buffer)) 
        
        if selAz.sum() ==0 | selEl.sum() ==0 :
            print(f"slew {index} no values")
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        
        plotAz = self.query_dict["az"][selAz]
        plotEl = self.query_dict["el"][selEl]
        
        ss_time = Time(self.slew_dict["slew_start_times"][index], format='unix_tai', scale='utc')
        ip_time = Time(self.slew_dict["slew_stop_times"][index], format='unix_tai', scale='utc')
        
        if (ip_time - ss_time).sec > 99:
            print(f"slew {index} is {(ip_time - ss_time).sec:0.0f} seconds long")
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        
        # Now calculates the spline fit and differentiate it to get the acceleration and jerk
        azPs = plotAz['actualPosition'].values
        azVs = plotAz['actualVelocity'].values
        azXs = plotAz['timestamp'].values - plotAz['timestamp'].values[0]  
        elPs = plotEl['actualPosition'].values
        elVs = plotEl['actualVelocity'].values
        elXs = plotEl['timestamp'].values - plotEl['timestamp'].values[0]
        plotStart = azXs[0] + 1.0
        plotEnd = azXs[-1] - 1.0
        
        npoints=int(np.max([np.round((azXs[-1]-azXs[0])/0.01/1e3,0)*1e3, 4000]))
        plotAzXs = np.linspace(azXs[0], azXs[-1], npoints)
        plotElXs = np.linspace(elXs[0], elXs[-1], npoints)
        if self.fit_method=="splines":
            azPosSpline, azVelSpline, azAccSpline, azJerkSpline = self.get_univariate_splines(azXs, azPs, azVs, plotAzXs, kernel) # times, positions, velocities, interpPoints, kernel
            elPosSpline, elVelSpline, elAccSpline, elJerkSpline = self.get_univariate_splines(elXs, elPs, elVs, plotElXs, kernel)
        elif self.fit_method=="savgol":
            azPosSpline, azVelSpline, azAccSpline, azJerkSpline = self.get_savgol_splines(azXs, azPs, plotAzXs)
            elPosSpline, elVelSpline, elAccSpline, elJerkSpline = self.get_savgol_splines(elXs, elPs, plotElXs)
        else: 
            print(f"bad fit_method: {self.fit_method}")
            exit
    
        spline_frame=pd.DataFrame({
            "slew_index":index, 
            "day":self.day,
            "azZeroTime":plotAz['timestamp'].values[0],
            "elZeroTime":plotEl['timestamp'].values[0],
            "azTime":plotAzXs,
            "azPosition":azPosSpline,
            "azVelocity":azVelSpline,          
            "azAcceleration":azAccSpline,
            "azJerk":azJerkSpline,
            "elZeroTime":plotEl['timestamp'].values[0],
            "elTime":plotElXs,
            "elPosition":elPosSpline,
            "elVelocity":elVelSpline,          
            "elAcceleration":elAccSpline,
            "elJerk":elJerkSpline}
        )
        if self.return_meas:
            az_measurement_frame=pd.DataFrame({
                            "slew_index":index, 
                            "day":self.day,
                            "azZeroTime":plotAz['timestamp'].values[0],
                            "az_times":azXs,
                            "azPositionMeas":azPs,
                            "azVelocityMeas":azVs,
                            })
            el_measurement_frame=pd.DataFrame({
                            "slew_index":index, 
                            "day":self.day,
                            "elZeroTime":plotEl['timestamp'].values[0],
                            "el_times":elXs,
                            "elPositionMeas":elPs,
                            "elVelocityMeas":elVs,
                            })
            return spline_frame, az_measurement_frame, el_measurement_frame
        else:
            return spline_frame

class create_slew_splines_method_1(create_slew_splines):
    def get_slew_times(self):
        "Use log events to identify telescope slews"
        # Find all of the time stamps

        # Start with start_slew times
        
        azs = self.query_dict["az_track"].values[:,0]
        els = self.query_dict["el_track"].values[:,0]
        az_times = self.query_dict["az_track"].values[:,1]
        el_times = self.query_dict["el_track"].values[:,1]
        az_starts = []
        el_starts = []
        
        
        slew_times_1 = []
        
        for i in range(1,len(self.query_dict["az_track"])):
            #command_trackTarget
            az_shift = abs(azs[i] - azs[i-1])
            if (az_shift > 0.1):
                az_starts.append(az_times[i])

        for i in range(1,len(self.query_dict["el_track"])):
            #command_trackTarget
            el_shift = abs(els[i] - els[i-1])
            if (el_shift > 0.1):
                el_starts.append(el_times[i])
        az_starts=np.array(az_starts)
        el_starts=np.array(el_starts)
        # Now in position timestamps

       
        az_stops = self.query_dict["az_pos"].values[:,1]
        el_stops = self.query_dict["el_pos"].values[:,1]

        az_starts, az_stops, az_unmatched_starts, az_unmatched_stops = self.get_slew_pairs(az_starts, az_stops)
        el_starts, el_stops, el_unmatched_starts, el_unmatched_stops = self.get_slew_pairs(el_starts, el_stops)
        
        slew_starts=az_starts
        slew_stops=az_stops
                                        
        for i in range(len(el_starts)):
            # get closest indentified slew 
            min_index=np.argmin(abs(slew_starts-el_starts[i]))
            start_min=slew_starts[min_index]
            stop_min=slew_stops[min_index]
            # see if we need to update the slew list
            if (el_starts[i] < start_min) & (el_stops[i] < start_min):
                # new slew
                slew_starts=np.append(slew_starts,el_starts[i])
                slew_stops=np.append(slew_stops,el_stops[i])
            elif (el_starts[i] > stop_min) & (el_stops[i] > stop_min) & \
            (el_starts[i] < slew_starts[np.min([min_index+1, len(slew_starts)-1], )]):
                # also a new slew
                slew_starts=np.append(slew_starts,el_starts[i])
                slew_stops=np.append(slew_stops,el_stops[i])
            elif (el_starts[i] <= start_min) & (el_stops[i] <= stop_min):
                # replace start
                slew_starts[min_index] = el_starts[i]
            elif (el_starts[i] >= start_min) & (el_stops[i] >= stop_min):
                # replace stop
                slew_stops[min_index] = el_stops[i]
        
        # truncating obviously too long slews
        long_slew_start=[]
        long_slew_stop=[]
        for i in range(len(slew_starts)):
            if slew_stops[i] - slew_starts[i] > 100:
                slew_stops[i] = slew_starts[i] + 100.0
                long_slew_start.append(slew_starts[i])
                long_slew_stop.append(slew_stops[i])
                
        print(f"identified {len(slew_starts)} slews")
        print(f"{len(long_slew_start)} of these slews are unreasonably long"
            "and truncated to 100s")
        unmatched_az_len=len(az_unmatched_starts)+len(az_unmatched_stops)
        unmatched_el_len=len(el_unmatched_starts)+len(el_unmatched_stops)
        if unmatched_az_len > 0:
            print(f"{unmatched_az_len} unmatched az peaks")
        if unmatched_el_len > 0:
            print(f"{unmatched_el_len} unmatched el peaks")
        slew_dict={
            "slew_start_times": slew_starts,
            "slew_stop_times": slew_stops
        }
        unmatched_slew_dict={"az_start":az_unmatched_starts, 
                            "az_stop":az_unmatched_stops,
                            "el_start":el_unmatched_starts,
                            "el_stop":el_unmatched_stops,
                            "long_slew_start":long_slew_start,
                            "long_slew_stop":long_slew_stop}
        return slew_dict, unmatched_slew_dict
    
    async def get_data(self):
        "Extract all the MTMount data from the EFD and save to parquet files"
        
        # Get EFD client
        client = EfdClient('usdf_efd')#vandv.efd.create_efd_client()
        
        self.query_dict={}
        print("starting query")
        # Query the EFD to extract the MTMount Azimuth data and wirte to csv
        self.query_dict["az"] = await client.select_time_series('lsst.sal.MTMount.azimuth', \
                                                ['actualPosition', 'actualVelocity', "timestamp"],  self.start_time, self.end_time)
        self.query_dict["el"] = await client.select_time_series('lsst.sal.MTMount.elevation', \
                                                ['actualPosition', 'actualVelocity', "timestamp"],  self.start_time, self.end_time)    
        self.query_dict["az_track"] = await client.select_time_series('lsst.sal.MTMount.command_trackTarget', \
                                                ['azimuth', 'taiTime'],  start_time, end_time)
        self.query_dict["el_track"] = await client.select_time_series('lsst.sal.MTMount.command_trackTarget', \
                                                ['elevation', 'taiTime'],  start_time, end_time)   
    
        self.query_dict["az_pos"] = await client.select_time_series('lsst.sal.MTMount.logevent_azimuthInPosition', \
                                                ['inPosition', 'private_kafkaStamp'],  start_time, end_time)
        
        self.query_dict["el_pos"] = await client.select_time_series('lsst.sal.MTMount.logevent_elevationInPosition', \
                                                    ['inPosition', 'private_kafkaStamp'],  start_time, end_time)
        if  ("inPosition" not in self.query_dict["az_pos"].keys()) | \
        ("inPosition" not in self.query_dict["el_pos"].keys()):
            print("no slews")
            return
        
        self.query_dict["az_pos"] = self.query_dict["az_pos"][self.query_dict["az_pos"]['inPosition']] # Select only the True values
        self.query_dict["el_pos"] = self.query_dict["el_pos"][self.query_dict["el_pos"]['inPosition']] # Select only the True values
        
        
        if ('actualVelocity' not in self.query_dict["az"].keys()) | \
        ('actualVelocity' not in self.query_dict["el"].keys()):
            print("no data")
            return None
        print("query done")
               
        # get dictionary of matched slews
        self.slew_dict, unmatched_slew_dict=self.get_slew_times()
        
        # save unmatched slews
        if np.max([len(unmatched_slew_dict[i]) for i in unmatched_slew_dict.keys()]) > 0:
            with open(os.path.join(output_dir,
                    f"unmatched_slews_{self.start_time}.json"), "w") as outfile:
                        json.dump(unmatched_slew_dict, outfile)
        
        spline_frames=[] 
        az_measurement_frames=[]
        el_measurement_frames=[]
        if len(self.slew_dict["slew_start_times"]) == 0:
            return
        for index in range(len(self.slew_dict["slew_start_times"])):
            spl_frame,az_meas_frame, el_meas_frame=self.fit_slew_profiles(index)
            spline_frames.append(spl_frame)
            az_measurement_frames.append(az_meas_frame)
            el_measurement_frames.append(el_meas_frame)
        spline_frame=pd.concat(spline_frames)
        az_measurement_frame=pd.concat(az_measurement_frames)
        el_measurement_frame=pd.concat(el_measurement_frames)
        
        # Write dataframes to parquet files in data dir. 
        #Using parquet preserves the column data types
        az_measurement_frame.to_parquet(os.path.join(output_dir,
                                        f"az_measurement_frame-{start_time}--{end_time}.parquet"))
        el_measurement_frame.to_parquet(os.path.join(output_dir, 
                                        f"el_measurement_frame-{start_time}--{end_time}.parquet"))
        if self.fit_method == "splines":
            spline_frame.to_parquet(os.path.join(output_dir, 
                                        f"spline_frame-{start_time}--{end_time}.parquet"))
        elif self.fit_method == "savgol":
            spline_frame.to_parquet(os.path.join(output_dir, 
                                        f"savgol_frame-{start_time}--{end_time}.parquet"))

    
class create_slew_splines_method_2(create_slew_splines):
    def get_slew_times(self):
        """use edge detection kernel to identify telesope slews"""
        smooth_kernel=np.ones(self.kernel_size)/self.kernel_size
        edge_kernel=np.concatenate([1 * np.ones(int(self.kernel_size/2)), -1 * np.ones(int(self.kernel_size/2))])/self.kernel_size
        
        #initially smooth and get speed not velocity
        az_smooth1=abs(np.convolve(self.query_dict["az"]['actualVelocity'], smooth_kernel, mode='same'))
        el_smooth1=abs(np.convolve(self.query_dict["el"]['actualVelocity'], smooth_kernel, mode='same'))
        
        #convolve edge detection kernel
        az_edge=np.convolve(az_smooth1, edge_kernel, mode='same')
        el_edge=np.convolve(el_smooth1, edge_kernel, mode='same')
        
        az_starts=self.query_dict["az"]["timestamp"][find_peaks(az_edge, height=0.02)[0]].values 
        az_stops=self.query_dict["az"]["timestamp"][find_peaks(az_edge * -1.0, height=0.02)[0]].values
        
        el_starts=self.query_dict["el"]["timestamp"][find_peaks(el_edge, height=0.02)[0]].values
        el_stops=self.query_dict["el"]["timestamp"][find_peaks(el_edge * -1.0, height=0.02)[0]].values
        
        if (len(az_starts) == 0) | (len(el_starts) == 0):
            print("no slews")
            slew_dict={
                "slew_start_times": [],
                "slew_stop_times": []
            }
            unmatched_slew_dict={"az_start":[], 
                                        "az_stop":[],
                                        "el_start":[],
                                        "el_stop":[]}
            return slew_dict, unmatched_slew_dict
        
        
        az_starts, az_stops, az_unmatched_starts, az_unmatched_stops = self.get_slew_pairs(az_starts, az_stops)
        el_starts, el_stops, el_unmatched_starts, el_unmatched_stops = self.get_slew_pairs(el_starts, el_stops)
        
        slew_starts=az_starts
        slew_stops=az_stops
                                        
        for i in range(len(el_starts)):
            # get closest indentified slew 
            min_index=np.argmin(abs(slew_starts-el_starts[i]))
            start_min=slew_starts[min_index]
            stop_min=slew_stops[min_index]
            # see if we need to update the slew list
            if (el_starts[i] < start_min) & (el_stops[i] < start_min):
                # new slew
                slew_starts=np.append(slew_starts,el_starts[i])
                slew_stops=np.append(slew_stops,el_stops[i])
            elif (el_starts[i] > stop_min) & (el_stops[i] > stop_min) & \
            (el_starts[i] < slew_starts[np.min([min_index+1, len(slew_starts)-1], )]):
                # also a new slew
                slew_starts=np.append(slew_starts,el_starts[i])
                slew_stops=np.append(slew_stops,el_stops[i])
            elif (el_starts[i] <= start_min) & (el_stops[i] <= stop_min):
                # replace start
                slew_starts[min_index] = el_starts[i]
            elif (el_starts[i] >= start_min) & (el_stops[i] >= stop_min):
                # replace stop
                slew_stops[min_index] = el_stops[i]
        
        # truncating obviously too long slews
        long_slew_start=[]
        long_slew_stop=[]
        for i in range(len(slew_starts)):
            if slew_stops[i] - slew_starts[i] > 100:
                slew_stops[i] = slew_starts[i] + 100.0
                long_slew_start.append(slew_starts[i])
                long_slew_stop.append(slew_stops[i])
                
        print(f"identified {len(slew_starts)} slews")
        print(f"{len(long_slew_start)} of these slews are unreasonably long"
            "and truncated to 100s")
        unmatched_az_len=len(az_unmatched_starts)+len(az_unmatched_stops)
        unmatched_el_len=len(el_unmatched_starts)+len(el_unmatched_stops)
        if unmatched_az_len > 0:
            print(f"{unmatched_az_len} unmatched az peaks")
        if unmatched_el_len > 0:
            print(f"{unmatched_el_len} unmatched el peaks")
        slew_dict={
            "slew_start_times": slew_starts,
            "slew_stop_times": slew_stops
        }
        unmatched_slew_dict={"az_start":az_unmatched_starts, 
                            "az_stop":az_unmatched_stops,
                            "el_start":el_unmatched_starts,
                            "el_stop":el_unmatched_stops,
                            "long_slew_start":long_slew_start,
                            "long_slew_stop":long_slew_stop}
        return slew_dict, unmatched_slew_dict
    
    async def get_data(self):
        "Extract all the MTMount data from the EFD and save to parquet files"
        
        # Get EFD client
        client = EfdClient('usdf_efd')#vandv.efd.create_efd_client()
        
        self.query_dict={}
        print("starting query")
        # Query the EFD to extract the MTMount Azimuth data and wirte to csv
        self.query_dict["az"] = await client.select_time_series('lsst.sal.MTMount.azimuth', \
                                                ['actualPosition', 'actualVelocity', "timestamp"],  self.start_time, self.end_time)
        self.query_dict["el"] = await client.select_time_series('lsst.sal.MTMount.elevation', \
                                                ['actualPosition', 'actualVelocity', "timestamp"],  self.start_time, self.end_time)    
        print("query done")
        if ('actualVelocity' not in self.query_dict["az"].keys()) | \
        ('actualVelocity' not in self.query_dict["el"].keys()):
            print("no data")
            return None

               
        # get dictionary of matched slews
        self.slew_dict, unmatched_slew_dict=self.get_slew_times()
        
        # save unmatched slews
        if np.max([len(unmatched_slew_dict[i]) for i in unmatched_slew_dict.keys()]) > 0:
            with open(os.path.join(output_dir,
                    f"unmatched_slews_{self.start_time}.json"), "w") as outfile:
                        json.dump(unmatched_slew_dict, outfile)
        
        spline_frames=[] 
        az_measurement_frames=[]
        el_measurement_frames=[]
        if len(self.slew_dict["slew_start_times"]) == 0:
            return
        for index in range(len(self.slew_dict["slew_start_times"])):
            spl_frame,az_meas_frame, el_meas_frame=self.get_splines(index)
            spline_frames.append(spl_frame)
            az_measurement_frames.append(az_meas_frame)
            el_measurement_frames.append(el_meas_frame)
        spline_frame=pd.concat(spline_frames)
        az_measurement_frame=pd.concat(az_measurement_frames)
        el_measurement_frame=pd.concat(el_measurement_frames)
        
        # Write dataframes to parquet files in data dir. 
        #Using parquet preserves the column data types
        az_measurement_frame.to_parquet(os.path.join(output_dir,
                                        f"az_measurement_frame-{start_time}--{end_time}.parquet"))
        el_measurement_frame.to_parquet(os.path.join(output_dir, 
                                        f"el_measurement_frame-{start_time}--{end_time}.parquet"))
        if self.fit_method == "splines":
            spline_frame.to_parquet(os.path.join(output_dir, 
                                        f"spline_frame-{start_time}--{end_time}.parquet"))
        elif self.fit_method == "savgol":
            spline_frame.to_parquet(os.path.join(output_dir, 
                                        f"savgol_frame-{start_time}--{end_time}.parquet"))

def get_arguments():
        """Get user supplied arguments using the argparse library."""
        
        parser = argparse.ArgumentParser("LVV-TXXX data from the EFD") 
        parser.add_argument("-d","--date", 
                            help="2023-03-24")
                            #"start time: ISO format if time_type is tai or unix; 
                            # float if time_type is unix_tai")
        parser.add_argument("-w","--window", 
                            help="end time is start time in hours")
        parser.add_argument("-t","--time", 
                            help="exact start time of window", 
                            default="14:01:00")
        parser.add_argument("-sm","--slew_method", 
                            help="which method to use for slew identification", 
                            default=2)
        parser.add_argument("-fm","--fit_method", 
                            help="which method to use for slew fits", 
                            default="splines")
        parser.add_argument('-f','--force',action='store_true',
                            help='overwrite outputs if they exist')
        
        args = parser.parse_args()

        date=args.date
        hour=args.time
        slew_method=int(args.slew_method)
        fit_method=args.fit_method
        force=args.force
        
        start_time_str=f"{date}T{hour}"
        start_time = astropy.time.Time(start_time_str, scale='utc')
        window=TimeDelta((args.window * u.hr).to(u.s), format='sec')
        end_time = start_time + window
        
        
        
        return start_time, end_time,slew_method, fit_method,force
    
if __name__ == '__main__':
    # Get time range from inputs
    start_time, end_time, slew_method, fit_method, force = get_arguments()
    print(start_time, end_time)
    # Setup output directory for data
    output_dir = f"./data/method_{slew_method}/"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if fit_method == "splines":
        output_spline_name=os.path.join(
            output_dir, 
            f"spline_frame-{start_time}--{end_time}.parquet"
        )
    elif fit_method == "savgol":
        output_spline_name=os.path.join(
            output_dir, 
            f"savgol_frame-{start_time}--{end_time}.parquet"
        )
    else: 
        print(f"bad fit_method: {fit_method}")
        exit
    
    output_az_measurement_name=os.path.join(
        output_dir, 
        f"az_measurement_frame-{start_time}--{end_time}.parquet"
    )
    output_el_measurement_name=os.path.join(
        output_dir, 
        f"el_measurement_frame-{start_time}--{end_time}.parquet"
    )
    run_bool=os.path.exists(output_spline_name)
    run_bool&=os.path.exists(output_az_measurement_name)
    run_bool&=os.path.exists(output_el_measurement_name)
    if run_bool and not force:
        print("output already found")
    else:    
        if slew_method==2:
            createSlewSplines=create_slew_splines_method_2(start_time=start_time,
                                                       end_time=end_time,
                                                       force=force,
                                                       output_dir=output_dir,
                                                       fit_method=fit_method)
        if slew_method==1:
            createSlewSplines=create_slew_splines_method_1(start_time=start_time,
                                                       end_time=end_time,
                                                       force=force,
                                                       output_dir=output_dir,
                                                       fit_method=fit_method)
    
    
        asyncio.run(createSlewSplines.get_data())    