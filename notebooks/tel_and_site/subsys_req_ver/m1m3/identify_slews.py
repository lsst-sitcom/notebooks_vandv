import numpy as np
from scipy.signal import find_peaks


def get_slews_edge_detection(times,velocities, kernel_size=100, height=0.005):
    smooth_kernel=np.ones(kernel_size)/kernel_size
    smoothed_speed = abs(np.convolve(velocities, smooth_kernel, mode='same'))
    
    edge_kernel=np.concatenate([1 * np.ones(int(kernel_size/2)), -1 * np.ones(int(kernel_size/2))])/kernel_size
    edge=np.convolve(smoothed_speed, edge_kernel, mode='same')
    
    starts=times[find_peaks(edge, height=height)[0]].values 
    stops=times[find_peaks(-1 * edge, height=height)[0]].values 
    maxv=[]
    for i, st in enumerate(starts):
        sel_vel = (times >= starts[i]) & (times <= stops[i])
        maxv.append(np.max(np.abs(smoothed_speed[sel_vel])))
    sel_slew=(np.array(maxv) > 0.05)
    
    starts=starts[sel_slew]
    stops=stops[sel_slew]
    
    for i,st in enumerate(starts):
        sel_starts=(times < st) & (smoothed_speed < 0.01)
        starts[i]=times[sel_starts][np.argmin(np.abs(times[sel_starts]-starts[i]))]
        
        sel_stops=(times > stops[i]) & (smoothed_speed < 0.01)
        stops[i]=times[sel_stops][np.argmin(np.abs(times[sel_stops]-stops[i]))]
                
    
    
    return starts,stops, maxv

def get_slew_pairs(starts,stops, return_unmatched=False):
        """
        Given vectors of start times and stop times take the longer vector 
        and iterate over it. If that is starts, for each start time select all stop
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

        if return_unmatched:
            return np.array(new_starts), np.array(new_stops), unmatched_starts, unmatched_stops
        else:
            return np.array(new_starts), np.array(new_stops)