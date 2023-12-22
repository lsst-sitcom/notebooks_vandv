import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import pandas as pd
from astropy.time import Time
from scipy.interpolate import UnivariateSpline
from scipy.interpolate import interp1d
from scipy.signal import find_peaks
from scipy.signal import savgol_filter

from lsst.summit.utils.tmaUtils import TMAEventMaker, getSlewsFromEventList, TMAEvent
from lsst.summit.utils.efdUtils import getEfdData, calcNextDay
from lsst.sitcom import vandv
import lsst.sitcom.vandv.tma.sitcom710 as sitcom710

# define limits from science requirements document (LTS-103 2.2.2) for plotting
# units in deg/s - deg/s^2 - deg/s^3
el_limit_dict={
    "max_velocity": 5.25,
    "max_acceleration": 5.25,
    "max_jerk": 21,
    "design_velocity": 3.5,
    "design_acceleration": 3.5,
    "design_jerk": 14,
}
az_limit_dict={
    "max_velocity": 10.5,
    "max_acceleration": 10.5,
    "max_jerk": 42,
    "design_velocity": 7,
    "design_acceleration": 7,
    "design_jerk": 28,
}

# default minimum y-ranges (peak-to-peak) for slew profile plot
plot_range_dict = {
    "az_pos": 0.1,
    "el_pos": 0.1,
    "az_vel": 0.02,
    "el_vel": 0.02,
    "az_acc": 0.02,
    "el_acc": 0.02,
    "az_jerk": 0.04,
    "el_jerk": 0.04,
}


# Class for analysis
class SlewData:
    """
    A class to query, analyze, and store slew data for a range of day observations.

    Handles data processing for slewing events within a specified date range. 
    It involves fetching data, spline/savgol fitting, and extracting key parameters 
    such as velocity, acceleration, and jerk for azimuth and elevation movements.

    Parameters:
    dayStart (str): Start date for data query in 'yyyymmdd' format.
    dayEnd (str): End date for data query in 'yyyymmdd' format.
    event_maker (EventMaker): Instance of EventMaker to fetch event data.
    spline_fit (str, optional): Type of spline fitting, defaults to "spline".
    padding (int, optional): Padding amount for data fetching, defaults to 0.
    smoothing (float, optional): smoothing factor for Univariate spline fit, defaults to 0
    kernel_size (int, optional): kernel size for Univariate spline fit, defaults to 20
    block_id (int, optional): block ID to identify groups of slews, defaults to -1 (which finds all slews)

    Attributes:
    day_range (list): List of days within the specified date range.
    event_maker (EventMaker): Event maker instance for fetching data.
    spline_fit (str): Spline fitting type applied.
    padding (int): Padding amount used in data fetching.
    smoothing (float): smoothing factor for Univariate spline fit
    kernel_size (int): kernel size for Univariate spline fit
    block_id (int): block ID to identify groups of slews related to that block
    real_az_data (DataFrame): Measured azimuth data for the entire day range.
    real_el_data (DataFrame): Measured elevation data for the entire day range.
    all_data (DataFrame): Data for the entire day range after spline fitting.
    max_data (DataFrame): Max values of velocity, acceleration, and jerk.
    """
    
    def __init__(self, 
                 dayStart, 
                 dayEnd, 
                 event_maker, 
                 spline_fit = "spline", 
                 padding = 0, 
                 smoothing = 0, 
                 kernel_size = 100,
                 block_id = -1):
        self.day_range = self.get_day_range(dayStart, dayEnd)
        self.event_maker = event_maker
        self.spline_fit = spline_fit
        self.padding = padding
        self.smoothing = smoothing
        self.kernel_size = kernel_size
        self.block_id = block_id
        self.real_az_data, self.real_el_data, self.all_data = self.get_spline_data()
        self.max_data = self.get_max_frame()
        
    def get_day_range(self, dayStart, dayEnd):
        """
        Generates a list of days between start and end dates, inclusive.

        Calculates the range of days from the start date to the end date. 
        Raises an error if start date is after end date.

        Parameters:
        dayStart (str): Start date in 'yyyymmdd' format.
        dayEnd (str): End date in 'yyyymmdd' format.

        Returns:
        list: Days in the specified date range.
        """
        if dayStart > dayEnd:
            assert False, "dayStart is after dayEnd"
        
        dayRange = []
        
        while dayStart <= dayEnd:
            dayRange.append(dayStart)
            dayStart = calcNextDay(dayStart)
        
        return dayRange

    def get_spline_frame(self, dayObs, index, fullAzTimestamp, fullElTimestamp, az_position, az_velocity, el_position, el_velocity):
        """
        Create a data frame for all fitted data
        """
        npoints = int(np.max([np.round((fullAzTimestamp[-1]-fullAzTimestamp[0])/0.01/1e3,0)*1e3, 4000])) # clarify what this is doing
        plotAzXs = np.linspace(fullAzTimestamp[0], fullAzTimestamp[-1], npoints)
        plotElXs = np.linspace(fullElTimestamp[0], fullElTimestamp[-1], npoints)

        kernel_size = self.kernel_size
        kernel = np.ones(kernel_size)/kernel_size
        s = self.smoothing # smoothing factor

        if self.spline_fit == "spline":
            # input: times, positions, velocities, interpPoints, kernel, smoothing factor
            azPosSpline, azVelSpline, azAccSpline, azJerkSpline = self.get_univariate_splines(fullAzTimestamp, az_position, az_velocity, plotAzXs, kernel, s) 
            elPosSpline, elVelSpline, elAccSpline, elJerkSpline = self.get_univariate_splines(fullElTimestamp, el_position, el_velocity, plotElXs, kernel, s)
        elif self.spline_fit == "savgol":
            azPosSpline, azVelSpline, azAccSpline, azJerkSpline = self.get_savgol_splines(fullAzTimestamp, az_position, plotAzXs)
            elPosSpline, elVelSpline, elAccSpline, elJerkSpline = self.get_savgol_splines(fullElTimestamp, el_position, plotElXs)
        else:
            assert False, self.spline_method + " is not a valid spline method. Use either \"spline\" or \"savgol\"."

        spline_frame=pd.DataFrame({
                    "slew_index":index,
                    "day":dayObs,
                    "azZeroTime":fullAzTimestamp.values[0],
                    "elZeroTime":fullElTimestamp.values[0],
                    "azTime":plotAzXs,
                    "azPosition":azPosSpline,
                    "azVelocity":azVelSpline,          
                    "azAcceleration":azAccSpline,
                    "azJerk":azJerkSpline,
                    "elTime":plotElXs,
                    "elPosition":elPosSpline,
                    "elVelocity":elVelSpline,          
                    "elAcceleration":elAccSpline,
                    "elJerk":elJerkSpline
        })

        return spline_frame

    def get_univariate_splines(self, times, positions, velocities, interpPoints, kernel, smoothingFactor):
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
        velSpline = UnivariateSpline(interpPoints, smoothedVel, s=smoothingFactor)
        accSpline1 = velSpline.derivative(n=1)
        smoothedAcc = np.convolve(accSpline1(interpPoints), kernel, mode='same')
        # Now smooth the derivative before differentiating again
        accSpline = UnivariateSpline(interpPoints, smoothedAcc, s=smoothingFactor)
        jerkSpline = accSpline.derivative(n=1)
        
        return posSpline(interpPoints), velSpline(interpPoints), accSpline(interpPoints), jerkSpline(interpPoints)
    
    def savgolFilter(self, times, positions,interpPoints, window=200, deriv=1, smoothingFactor = 0.01): 
        positionSpline = UnivariateSpline(times, positions, s=smoothingFactor)(interpPoints) 
        derivativePoints = savgol_filter(positionSpline, window_length=window, mode="mirror",
                                        deriv=deriv, polyorder=3, delta=(interpPoints[1]-interpPoints[0])) 
        return derivativePoints

    def get_savgol_splines(self, times, positions, interpPoints):
        posSpline = UnivariateSpline(times, positions, s=0)(interpPoints)
        velSpline = self.savgolFilter(times, positions, interpPoints, window=200, deriv=1, smoothingFactor=0.01)
        accSpline = self.savgolFilter(times, positions, interpPoints, window=200, deriv=2, smoothingFactor=0.01)
        jerkSpline = self.savgolFilter(times, positions, interpPoints, window=200, deriv=3, smoothingFactor=0.01)
        return posSpline, velSpline, accSpline, jerkSpline
    
    def get_spline_data(self):
        """
        Queries the EFD and returns data frames of measured and fitted data
        for each slew in the day range.
        """
        topic_az = "lsst.sal.MTMount.azimuth"
        topic_el = "lsst.sal.MTMount.elevation"
        topic_columns = ["actualPosition", "actualVelocity", "timestamp"]
        
        azActual_frame = pd.DataFrame()
        elActual_frame = pd.DataFrame()
        spline_frame = pd.DataFrame()

        for day in self.day_range:
            all_events = self.event_maker.getEvents(day)

            # check if querying for slews related to specific blocks
            if self.block_id == -1:
                slew_events = getSlewsFromEventList(all_events)
                print(f'Found {len(slew_events)} slews for {day=}')
            else:
                relateTo_events = [e for e in all_events if e.relatesTo(block=self.block_id)]
                slew_events = getSlewsFromEventList(relateTo_events)
                print(f'For {day=}, related events: {len(relateTo_events)}, slews: {len(slew_events)}')

            if len(slew_events) == 0:
                continue

            for slew in range(len(slew_events)):
                data_az = getEfdData(
                    self.event_maker.client, 
                    topic_az, columns = topic_columns,
                    prePadding = self.padding,
                    postPadding = self.padding,
                    event=slew_events[slew]
                )
                data_el = getEfdData(
                    self.event_maker.client, 
                    topic_el, columns = topic_columns,
                    prePadding = self.padding,
                    postPadding = self.padding,
                    event=slew_events[slew]
                )
                
                # check if the event has enough data to fit a spline
                if (data_az.shape[0] < 4) or (data_el.shape[0] < 4):
                    continue
                
                azActual_frame_single = pd.DataFrame({
                    "day":day,
                    "slew_index":slew,
                    "azTimestamp":data_az["timestamp"],
                    "azPosition":data_az["actualPosition"],
                    "azVelocity":data_az["actualVelocity"],
                })
                
                elActual_frame_single = pd.DataFrame({
                    "day":day,
                    "slew_index":slew,
                    "elTimestamp":data_el["timestamp"],
                    "elPosition":data_el["actualPosition"],
                    "elVelocity":data_el["actualVelocity"],
                })

                spline_frame_single = self.get_spline_frame(
                    day,
                    slew,
                    data_az["timestamp"],
                    data_el["timestamp"], 
                    data_az["actualPosition"], 
                    data_az["actualVelocity"], 
                    data_el["actualPosition"], 
                    data_el["actualVelocity"]
                )
                
                azActual_frame = pd.concat([azActual_frame, azActual_frame_single], ignore_index=True)
                elActual_frame = pd.concat([elActual_frame, elActual_frame_single], ignore_index=True)
                spline_frame = pd.concat([spline_frame, spline_frame_single], ignore_index=True)
        
        return azActual_frame, elActual_frame, spline_frame
    
    def get_max_frame(self):
        if self.all_data.empty:
            print("Warning! Query succesful but there's no data.")
            empty  = pd.DataFrame()
            return empty
                  
        slew_num = []
        day_num = []
        slew_time = [] # final time

        az_vel_max = []
        el_vel_max = []

        az_acc_max = []
        el_acc_max = []

        az_jerk_max = []
        el_jerk_max = []
        
        for i in np.unique(self.all_data['day']):
            slew_day = (self.all_data['day']==i)
            for j in np.unique(self.all_data['slew_index'][slew_day]):
                slew_id = slew_day & (self.all_data['slew_index']==j)

                az_vel_max.append(abs(self.all_data.loc[slew_id,"azVelocity"]).max())
                el_vel_max.append(abs(self.all_data.loc[slew_id,"elVelocity"]).max())

                az_acc_max.append(abs(self.all_data.loc[slew_id,"azAcceleration"]).max())
                el_acc_max.append(abs(self.all_data.loc[slew_id,"elAcceleration"]).max())

                az_jerk_max.append(abs(self.all_data.loc[slew_id,"azJerk"]).max())
                el_jerk_max.append(abs(self.all_data.loc[slew_id,"elJerk"]).max())

                slew_num.append(j)
                day_num.append(i)

                slew_time.append(np.max([self.all_data.loc[slew_id,"azTime"].max(),self.all_data.loc[slew_id,"elTime"].max()]))

        max_frame=pd.DataFrame({
            "day":day_num,
            "slew":slew_num,
            "az_vel":az_vel_max,
            "az_acc":az_acc_max,
            "az_jerk":az_jerk_max,
            "el_vel":el_vel_max,
            "el_acc":el_acc_max,
            "el_jerk":el_jerk_max
        })

        return max_frame
    
# plotting functions 
# 
def plot_max_hist(max_frame, limitsBool, logBool, fit, padding):
    """
    Generate a histogram of maximum velocity, acceleration, and jerk for each
    slew in both azimuth and elevation
    """
    padding = str(padding)
    num_slews= str(max_frame.shape[0])
    design_color = "green"
    max_color = "orange"
    
    first_day = max_frame['day'].min()
    last_day = max_frame['day'].max()

    fig,axs = plt.subplots(3, 2, dpi=175, figsize=(10,5), sharex=False)
    plt.subplots_adjust(wspace=0.3, hspace=0.5)
    if len(np.unique(max_frame['day'])) > 1:
        plt.suptitle(f"Maximums for {first_day} - {last_day} -- Slews: " + num_slews + \
                     "\n Fit: " + fit + " -- Padding: " + padding, fontsize = 14)
    else:
        plt.suptitle(f"Maximums for {first_day} -- Slews: " + num_slews + \
                     "\nFit: " + fit + " -- Padding: " + padding, fontsize = 14)

    # bins for each graph
    velbins_az = np.linspace(0, max_frame["az_vel"].max(), 75) 
    accbins_az = np.linspace(0, max_frame["az_acc"].max(), 100)
    jerkbins_az = np.linspace(0, max_frame["az_jerk"].max(), 100)
    
    velbins_el = np.linspace(0, max_frame["el_vel"].max(), 75) 
    accbins_el = np.linspace(0, max_frame["el_acc"].max(), 100)
    jerkbins_el = np.linspace(0, max_frame["el_jerk"].max(), 100)
    
    plt.subplot(3,2,1)
    plt.hist(max_frame["az_vel"], log=logBool, color="tab:blue", bins=velbins_az)
    if limitsBool:
        # only require counts output from plt.hist
        counts, bins, patches = plt.hist(max_frame["az_vel"], color="tab:blue", bins=velbins_az)
        plotHistAzDesignLim("design_velocity", "max_velocity", 0, np.max(counts), design_color, max_color)
    plt.title(f"Azimuth")
    plt.xlabel("Velocity [deg/s]")

    plt.subplot(3,2,2)
    plt.hist(max_frame["el_vel"], log=logBool, color="tab:blue", bins=velbins_el)
    if limitsBool == True:
        counts, bins, patches = plt.hist(max_frame["el_vel"], color="tab:blue", bins=velbins_el)
        plotHistElDesignLim("design_velocity", "max_velocity", 0, np.max(counts), design_color, max_color)
    plt.title(f"Elevation")
    plt.xlabel("Velocity [deg/s]")

    plt.subplot(3,2,3)
    plt.hist(max_frame["az_acc"], log=logBool, color="tab:blue", bins=accbins_az)
    if limitsBool == True:
        counts, bins, patches = plt.hist(max_frame["az_acc"], color="tab:blue", bins=accbins_az)
        plotHistAzDesignLim("design_acceleration", "max_acceleration", 0, np.max(counts), design_color, max_color)
    plt.xlabel("Acceleration [deg/s$^2$]")

    plt.subplot(3,2,4)
    plt.hist(max_frame["el_acc"], log=logBool, color="tab:blue", bins=accbins_el)
    if limitsBool == True:
        counts, bins, patches = plt.hist(max_frame["el_acc"], color="tab:blue", bins=accbins_el)
        plotHistElDesignLim("design_acceleration", "max_acceleration", 0, np.max(counts), design_color, max_color)
    plt.xlabel("Acceleration [deg/s$^2$]")

    plt.subplot(3,2,5)
    plt.hist(max_frame["az_jerk"], log=logBool, color="tab:blue", bins=jerkbins_az)
    if limitsBool == True:
        counts, bins, patches = plt.hist(max_frame["az_jerk"], color="tab:blue", bins=jerkbins_az)
        plotHistAzDesignLim("design_jerk", "max_jerk", 0, np.max(counts), design_color, max_color)
    plt.xlabel("Jerk [deg/s$^3$]")

    plt.subplot(3,2,6)
    plt.hist(max_frame["el_jerk"], log=logBool, color="tab:blue", bins=jerkbins_el)
    if limitsBool == True:
        counts, bins, patches = plt.hist(max_frame["el_jerk"], color="tab:blue", bins=jerkbins_el)
        plotHistElDesignLim("design_jerk", "max_jerk", 0, np.max(counts), design_color, max_color)
    plt.xlabel("Jerk [deg/s$^3$]")

    plt.show()

# define functions to add limits to slew profile for neatness
def plotAzDesignlim(design_input, max_input, xmin, xmax, design_color, max_color):
    plt.hlines(az_limit_dict[design_input], xmin=xmin, xmax=xmax, color = design_color)
    plt.hlines(-az_limit_dict[design_input], xmin=xmin, xmax=xmax, color = design_color)
    plt.hlines(az_limit_dict[max_input], xmin=xmin, xmax=xmax, color = max_color)
    plt.hlines(-az_limit_dict[max_input], xmin=xmin, xmax=xmax, color = max_color)
    
def plotElDesignlim(design_input, max_input, xmin, xmax, design_color, max_color):
    plt.hlines(el_limit_dict[design_input], xmin=xmin, xmax=xmax, color = design_color)
    plt.hlines(-el_limit_dict[design_input], xmin=xmin, xmax=xmax, color = design_color)
    plt.hlines(el_limit_dict[max_input], xmin=xmin, xmax=xmax, color = max_color)
    plt.hlines(-el_limit_dict[max_input], xmin=xmin, xmax=xmax, color = max_color)

def plotHistAzDesignLim(design_input, max_input, ymin, ymax, design_color, max_color):
    plt.vlines(az_limit_dict[design_input], ymin=ymin, ymax=ymax, color = design_color)
    plt.vlines(az_limit_dict[max_input], ymin=ymin, ymax=ymax, color = max_color)
    
def plotHistElDesignLim(design_input, max_input, ymin, ymax, design_color, max_color):
    plt.vlines(el_limit_dict[design_input], ymin=ymin, ymax=ymax, color = design_color)
    plt.vlines(el_limit_dict[max_input], ymin=ymin, ymax=ymax, color = max_color)

def slew_profile_plot(actual_az_frame, actual_el_frame, spline_frame, dayObs, slew_index, limitsBool, min_ylims = plot_range_dict):
    """
    Generate plots for a slew profile in both azimuth and elevation 
    for position, velocity, acceleration, and jerk
    """
    # create a spline frame for a single slew
    actual_az_slew = actual_az_frame.loc[((actual_az_frame['day']==dayObs) & (actual_az_frame['slew_index']==slew_index))]
    actual_el_slew = actual_el_frame.loc[((actual_el_frame['day']==dayObs) & (actual_el_frame['slew_index']==slew_index))]
    slew_frame = spline_frame.loc[((spline_frame['day']==dayObs) & (spline_frame['slew_index']==slew_index))]
    
    if len(slew_frame) == 0:
        assert False, f"There is no data for slew {slew_index} of dayObs {dayObs}"
    
    # format time used in the title
    title_time = Time(slew_frame['azZeroTime'].iloc[[0]], format = 'unix').iso
    
    # get the relative times for the x-axis
    azActualTimes = actual_az_slew['azTimestamp'] - actual_az_slew['azTimestamp'].values[0]
    elActualTimes = actual_el_slew['elTimestamp'] - actual_el_slew['elTimestamp'].values[0]
    
    azRelativeTimes = slew_frame['azTime'] - slew_frame['azZeroTime']
    elRelativeTimes = slew_frame['elTime'] - slew_frame['elZeroTime']

    fig,axs = plt.subplots(4, 2, dpi=175, figsize=(10,5), sharex=True)
    plt.subplots_adjust(wspace=0.3, hspace=0.5)
    plt.suptitle(f"TMA Slew Number {slew_index} \n Time: {title_time}", fontsize = 12, y = 1.00)

    # make it easier to change variables across subplots
    mark = "x"
    mark_color = "purple"
    mark_size = 30
    line_width = 2
    az_color = "red"
    el_color = "blue"
    design_color = "green"
    max_color = "orange"
    opacity = 0.5
    
    plt.subplot(4,2,1)
    plt.scatter(azActualTimes, actual_az_slew['azPosition'], marker=mark, color=mark_color,alpha=opacity, s=mark_size, label='data')
    plt.plot(azRelativeTimes, slew_frame['azPosition'], lw=line_width, color=az_color, label='fit')
    plt.title(f"Azimuth")
    plt.ylabel("Position\n[deg]")
    if (slew_frame['azPosition'].max()-slew_frame['azPosition'].min() < min_ylims["az_pos"]):
        meanval = slew_frame['azPosition'].mean()
        plt.ylim(meanval-(min_ylims["az_pos"]/2), meanval+(min_ylims["az_pos"]/2))
    plt.legend()

    plt.subplot(4,2,2)
    plt.scatter(elActualTimes, actual_el_slew['elPosition'], marker=mark, color=mark_color,alpha=opacity, s=mark_size, label='data')
    plt.plot(elRelativeTimes, slew_frame['elPosition'], lw=line_width, color=el_color, label='fit')
    plt.title(f"Elevation")
    if (slew_frame['elPosition'].max()-slew_frame['elPosition'].min() < min_ylims["el_pos"]):
        meanval = slew_frame['elPosition'].mean()
        plt.ylim(meanval-(min_ylims["el_pos"]/2), meanval+(min_ylims["el_pos"]/2))
    plt.legend()

    plt.subplot(4,2,3)
    plt.scatter(azActualTimes, actual_az_slew['azVelocity'], marker=mark, color=mark_color,alpha=opacity, s=mark_size)
    plt.plot(azRelativeTimes, slew_frame['azVelocity'], lw=line_width, color=az_color)
    if limitsBool == True:
        plotAzDesignlim("design_velocity", "max_velocity", azRelativeTimes.iloc[[0]], azRelativeTimes.iloc[[-1]], design_color, max_color)
    plt.ylabel("Velocity\n[deg/sec]")
    if (slew_frame['azVelocity'].max()-slew_frame['azVelocity'].min() < min_ylims["az_vel"]) & (limitsBool == False):
        meanval = slew_frame['azVelocity'].mean()
        plt.ylim(meanval-(min_ylims["az_vel"]/2), meanval+(min_ylims["az_vel"]/2))

    plt.subplot(4,2,4)
    plt.scatter(elActualTimes, actual_el_slew['elVelocity'], marker=mark, color=mark_color,alpha=opacity, s=mark_size)
    plt.plot(elRelativeTimes, slew_frame['elVelocity'], lw=line_width, color=el_color)
    if limitsBool == True:
        plotElDesignlim("design_velocity", "max_velocity", elRelativeTimes.iloc[[0]], elRelativeTimes.iloc[[-1]], design_color, max_color)
    if (slew_frame['elVelocity'].max()-slew_frame['elVelocity'].min() < min_ylims["el_vel"]) & (limitsBool == False):
        meanval = slew_frame['elVelocity'].mean()
        plt.ylim(meanval-(min_ylims["el_vel"]/2), meanval+(min_ylims["el_vel"]/2))

    plt.subplot(4,2,5)
    plt.plot(azRelativeTimes, slew_frame['azAcceleration'], lw=line_width, color=az_color)
    if limitsBool == True:
        plotAzDesignlim("design_acceleration", "max_acceleration", azRelativeTimes.iloc[[0]], azRelativeTimes.iloc[[-1]], design_color, max_color)
    plt.ylabel("Acceleration\n[deg/sec$^2$]")
    if (slew_frame['azAcceleration'].max()-slew_frame['azAcceleration'].min() < min_ylims["az_acc"]) & (limitsBool == False):
        meanval = slew_frame['azAcceleration'].mean()
        plt.ylim(meanval-(min_ylims["az_acc"]/2), meanval+(min_ylims["az_acc"]/2))

    plt.subplot(4,2,6)
    plt.plot(elRelativeTimes, slew_frame['elAcceleration'], lw=line_width, color=el_color)
    if limitsBool == True:
        plotElDesignlim("design_acceleration", "max_acceleration", elRelativeTimes.iloc[[0]], elRelativeTimes.iloc[[-1]], design_color, max_color)
    if (slew_frame['elAcceleration'].max()-slew_frame['elAcceleration'].min() < min_ylims["el_acc"]) & (limitsBool == False):
        meanval = slew_frame['elAcceleration'].mean()
        plt.ylim(meanval-(min_ylims["el_acc"]/2), meanval+(min_ylims["el_acc"]/2))

    plt.subplot(4,2,7)
    plt.plot(azRelativeTimes, slew_frame['azJerk'], lw=line_width, color=az_color)
    if limitsBool == True:
        plotAzDesignlim("design_jerk", "max_jerk", azRelativeTimes.iloc[[0]], azRelativeTimes.iloc[[-1]], design_color, max_color)
    plt.ylabel("Jerk\n[deg/sec$^3$]")
    plt.xlabel("time since slew start [sec]")
    if (slew_frame['azJerk'].max()-slew_frame['azJerk'].min() < min_ylims["az_jerk"]) & (limitsBool == False):
        meanval = slew_frame['azJerk'].mean()
        plt.ylim(meanval-(min_ylims["az_jerk"]/2), meanval+(min_ylims["az_jerk"]/2))

    plt.subplot(4,2,8)
    plt.plot(elRelativeTimes, slew_frame['elJerk'], lw=line_width, color=el_color)
    if limitsBool == True:
        plotElDesignlim("design_jerk", "max_jerk", elRelativeTimes.iloc[[0]], elRelativeTimes.iloc[[-1]], design_color, max_color)
    plt.xlabel("time since slew start [sec]")
    if (slew_frame['elJerk'].max()-slew_frame['elJerk'].min() < min_ylims["el_jerk"]) & (limitsBool == False):
        meanval = slew_frame['elJerk'].mean()
        plt.ylim(meanval-(min_ylims["el_jerk"]/2), meanval+(min_ylims["el_jerk"]/2))
    
    plt.subplots_adjust(wspace=0.20, hspace=0.20)

    plt.show()