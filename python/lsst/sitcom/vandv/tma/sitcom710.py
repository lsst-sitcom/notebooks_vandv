from scipy.interpolate import UnivariateSpline
from scipy.interpolate import interp1d
from scipy.signal import find_peaks
from scipy.signal import savgol_filter


class SlewData:
    """
    Queries, analyzes, and holds slew data for a range of dayObs
    
    Parameters
    
    Attributes
    """
    
    def __init__(self, dayStart, dayEnd, event_maker, spline_fit = "spline", padding = 0):
        self.day_range = self.get_day_range(dayStart, dayEnd)
        self.event_maker = event_maker
        self.spline_fit = spline_fit
        self.padding = padding
        self.all_data = self.get_spline_data()
        self.max_data = self.get_max_frame()
        
    def get_day_range(self, dayStart, dayEnd):
        if dayStart > dayEnd:
            assert False, "dayStart is after dayEnd"
        
        dayRange = []
        
        while dayStart <= dayEnd:
            dayRange.append(dayStart)
            dayStart = calcNextDay(dayStart)
        
        return dayRange

    def get_spline_frame(self, dayObs, index, fullAzTimestamp, fullElTimestamp, az_position, az_velocity, el_position, el_velocity):
        """
        Create a data frame for all original and fitted data
        """
        npoints = int(np.max([np.round((fullAzTimestamp[-1]-fullAzTimestamp[0])/0.01/1e3,0)*1e3, 4000])) # clarify what this is doing
        plotAzXs = np.linspace(fullAzTimestamp[0], fullAzTimestamp[-1], npoints)
        plotElXs = np.linspace(fullElTimestamp[0], fullElTimestamp[-1], npoints)

        kernel_size = len(fullAzTimestamp)
        kernel = np.ones(kernel_size)/kernel_size
        s = 0 # smoothing factor

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
        topic_az = "lsst.sal.MTMount.azimuth"
        topic_el = "lsst.sal.MTMount.elevation"
        topic_columns = ["actualPosition", "actualVelocity", "timestamp"]

        spline_frame = pd.DataFrame()

        for day in self.day_range:
            slew_events = getSlewsFromEventList(self.event_maker.getEvents(day))
            print(f'Found {len(slew_events)} slews for {day=}')

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
                
                spline_frame = pd.concat([spline_frame, spline_frame_single], ignore_index=True)
        
        return spline_frame
    
    def get_max_frame(self):
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