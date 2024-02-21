# # Settling Time Statistics for a block
# 
# This script will create a plot of average settling time for the six IMS components  
#  for all slews in a given block
#
# Author: Nacho Sevilla
# https://jira.lsstcorp.org/browse/SITCOM-1172
# 

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from optparse import OptionParser
from datetime import datetime

from astropy.time import Time, TimezoneInfo
from scipy import stats

from lsst.summit.utils.tmaUtils import TMAEventMaker, TMAState
from lsst.summit.utils.efdUtils import getEfdData, makeEfdClient
from lsst.summit.utils.blockUtils import BlockParser

import warnings

warnings.filterwarnings("ignore")

def checkRequirement(referenceTime, df_ims, imsColumn, correctedVariable, req, f, verbose = False):
    '''
    Function to check requirement for RMS and bias 1 s after slew stop
    Parameters:
    reference_time: slew stop time in UTC
    df_ims: pandas data frame with IMS (Independent Measurement System of the M1M3) data
    imsColumn: specific IMS column to analyze in this function call
    correctedVariable: column value corrected for the value at the end of the window, determined before the function call
    req: tolerance for IMS value to be within requirement. We will apply it to the RMS and bias of the value, with respect to the value at reference_t + post_delta_t
    f: file name for logs
    verbose: flag for verbosity in outputs
    Returns:
    PF: PASS/FAIL for this column and event, at 1 second after slew stop, checking RMS and bias
    rmsAtReq: value of the RMS (jitter) of the column value, using a rolling check, after the slew stop
    meanAtReq: value of the bias (in absolute value) of the column value, using a rolling check, after the slew stop
    settleTime: computed as latest time in which the IMS value has gone above the requirement in RMS or mean
    '''
    ## recomputing RMS for the whole range since T0
    rolling = 20
    time_array = df_ims.index
    slew_stop = Time(referenceTime).unix
    iT0 = np.argmin(np.abs((Time(time_array).unix) - slew_stop)) #which index in time_array is closest to slew_stop
    iT1 = np.argmin(np.abs((Time(time_array).unix) - (Time(time_array).unix[iT0] + 1))) #which index in time_array is closest to slew_stop + 1 second
    targetVariable = df_ims[imsColumn][iT0:-1]
    #rms = (targetVariable - targetVariableReference[1]).rolling(rolling).std()
    rms = (correctedVariable).rolling(rolling).std()
    mean = abs((correctedVariable).rolling(rolling).mean())
    rmsAtReq = rms[iT1 - iT0]
    meanAtReq = mean[iT1 - iT0]
    krms = [index for index, x in enumerate(rms) if np.abs(x) >= req] 
    kmean = [index for index, x in enumerate(mean) if np.abs(x) >= req]
    if (all(x < req for x in rms[iT1 - iT0 : -1])) and (all(x < req for x in mean[iT1 - iT0 : -1])):
        #all values of RMS and mean are OK since 1 s after slew stop
        if verbose:
            logMessage(f"{imsColumn} Test PASSED",f)
        if (krms == []) and (kmean == []): #both RMS and mean comply with requirement at all values 
            settleTime = 0 #already settled at slew stop
        else:
            if krms == []: # only RMS values comply with requirement since slew stop
                settleTime = time_array[iT0 + kmean[-1]] - time_array[iT0]
            elif kmean == []: # only mean values comply with requirement
                settleTime = time_array[iT0 + krms[-1]] - time_array[iT0]
            else: # neither comply with requirement for all times since slew stop, take maximum
                settleTime = max(time_array[iT0 + krms[-1]],time_array[iT0 + kmean[-1]]) - time_array[iT0] 
            settleTime = settleTime.total_seconds()
        PF = True
    else:
        if rmsAtReq > req:
            if verbose:
                logMessage(f"{imsColumn} Test FAILED in RMS by {rmsAtReq-req}",f)
        if meanAtReq > req:
            if verbose:
                logMessage(f"{imsColumn} Test FAILED in mean by {meanAtReq-req}",f)  
        if krms == []:
            settleTime = time_array[iT0 + kmean[-1]] - time_array[iT0]
        elif kmean == []:
            settleTime = time_array[iT0 + krms[-1]] - time_array[iT0]
        else:
            settleTime = max(time_array[iT0 + krms[-1]],time_array[iT0 + kmean[-1]]) - time_array[iT0]    
        settleTime = settleTime.total_seconds()
        PF = False
    logMessage(f"settleTime:{settleTime}",f)
    return PF,rmsAtReq,meanAtReq,settleTime

def computeSettleTime(
    df_ims,  # input data frame
    referenceTime="2023-06-01T06:00:0Z",  # time for slew stop (T0)
    lo_delta_t=5,  # in seconds
    hi_delta_t=30,  # in seconds
    imsColumn="xPosition",  # IMS column
    rmsReq=2e-3,  # requirement in appropriate units
    chi2prob=0.999,  # confidence level for IMS variable wrt to long term value and variance to agree
    f="SITCOM_1172.log",
    verbose=False
):
    '''
    Function to compute settle time and PASS/FAIL for a given slew stop event
    Parameters:
    df_ims: pandas data frame with IMS (Independent Measurement System of the M1M3) data
    reference_time: slew stop time in UTC
    lo_delta_t: time window in seconds BEFORE the slew stop to retrieve data
    hi_delta_t: time window in seconds AFTER the slew stop to retrieve data
    imsColumn: specific IMS column to analyze in this function call
    rmsReq: tolerance for IMS value to be within requirement. We will apply it to the RMS and bias of the value, with respect to the value at reference_t + post_delta_t
    chi2prob: confidence level for IMS variable wrt to long term value and variance to agree
    f: file name for logs
    verbose: Verbosity flag of log outputs
    Returns:
    PF: PASS/FAIL for this column and event, at 1 second after slew stop, checking RMS and bias
    settleInterval: the time after slew stop where the algorithm determines there is stability, independently of requirement
    rmsAtReq: value of the RMS (jitter) of the column value, using a rolling check, after the slew stop
    meanAtReq: value of the bias (in absolute value) of the column value, using a rolling check, after the slew stop
    '''

    if "Position" in imsColumn:
        units = "mm"
        ylimMax = rmsReq + 0.005
    elif "Rotation" in imsColumn:
        units = "deg"
        ylimMax = rmsReq + 0.0001
    else:
        print("Unidentified column")
        return -1

    settleTime = False

    T0 = pd.to_datetime(referenceTime)  # this is slew stop
    delta_window = [
        pd.Timedelta(lo_delta_t, "seconds"),
        pd.Timedelta(hi_delta_t, "seconds"),
    ]
    # zoom around the T0 of interest
    TZoom = [T0 - delta_window[0], T0 + delta_window[1]]

    # targetVariablePlot takes the data frame for the complete plot range
    targetVariablePlot = df_ims[imsColumn][TZoom[0] : TZoom[1]]
    # targetVariableCheck takes the data from the slew stop, until the end of the plot
    targetVariableCheck = df_ims[imsColumn][T0 : TZoom[1]]
    idxT0 = df_ims.index[  # index in dataframe closest in time to slew stop
        df_ims.index.get_indexer([pd.to_datetime(T0)], method="nearest")
    ]
    idxTend = df_ims.index[  # index in dataframe closest in time to end of plot
        df_ims.index.get_indexer(
            [pd.to_datetime(T0 + delta_window[1])], method="nearest"
        )
    ]
    targetVariableReference = [
        float(df_ims[imsColumn][idxT0]),
        float(df_ims[imsColumn][idxTend]),
    ]
    if len(targetVariablePlot.index) == 0:
        print("Data frame is empty")
        return -1

    # it is important that the end of the plot (targetVariableReference[1])
    # does not hit another slew or movement, nor at any point in the middle of the window

    # correct IMS variable wrt end of plot
    correctedVariablePlot = targetVariablePlot - targetVariableReference[1]
    correctedVariableCheck = targetVariableCheck - targetVariableReference[1]
    correctedVariableCheck2 = np.square(correctedVariableCheck)

    # number of values where the chi2 will be computed
    rolling = 30  # 50 is approx. 1 s
    # chi2 right tail probability for N=rolling dof at chi2prob CL
    crit = stats.chi2.ppf(chi2prob, rolling)

    rms = correctedVariableCheck.rolling(rolling).std()
    var = correctedVariableCheck.rolling(rolling).var()
    mean = abs(correctedVariableCheck.rolling(rolling).mean())

    # compute the chi2 against the null hypothesis
    # the x_i are the measurements in a window (wrt to reference at end of plot)
    # the variance is for the same values
    # so chi2 = sum_N[(x_i - 0)**2/variance] where N = rolling
    sum2 = correctedVariableCheck2.rolling(rolling).sum()
    chi2 = sum2 / var
    # check the chi2 at each step using rollingCheck as the number of consecutive instances in which
    # chi2 has to be under the critical value
    # or rms and bias be both already 10% of requirement
    PFCheck = (chi2 < crit) | ((rms < 0.1 * rmsReq) & (mean < 0.1 * rmsReq))
    # PFCheck = (rms < 0.2 * rmsReq) & (mean < 0.5 * rmsReq)
    rollingCheck = 30
    stabilityCheck = (
        PFCheck.rolling(rollingCheck).apply(lambda s: s.all()) > 0
    )  # true if rollingCheck consecutive true values of PFcheck
    if len(stabilityCheck[stabilityCheck == True]) <= rollingCheck:  ## == 0:
        # print(f"Not settled within {postPadding} s window")
        settleTime = False
    elif rms[stabilityCheck[stabilityCheck == True].index[0]] <= rmsReq:
        settleTime = stabilityCheck[stabilityCheck == True].index[rollingCheck]
    else:
        n = 1
        while (
            rms[stabilityCheck[stabilityCheck == True].index[n + rollingCheck]] > rmsReq
        ):
            settleTime = stabilityCheck[stabilityCheck == True].index[n + rollingCheck]
            n = n + 1
        # if settleTime < referenceTime:
        #    settleTime = referenceTime
    settleInterval = -1
    if settleTime:
        settleInterval = settleTime - referenceTime
        if settleInterval.total_seconds() < 0:
            print(f"Already settled at slew stop")
            settleInterval = 0
        else:
            settleInterval = settleInterval.total_seconds()

    PF,rmsAtReq,meanAtReq,settleIntervalReq = checkRequirement(referenceTime, df_ims, imsColumn, correctedVariableCheck, rmsReq, f, verbose)    

    #if not settleTime:
    #    return True,-1,rmsAtReq,meanAtReq

    return PF, settleIntervalReq,rmsAtReq,meanAtReq
    
def logMessage(mess, f):
    print(mess)
    try:
        print(mess, file=f)
    except:
        print("Could not write to log file")
    return

def getSlewTimes(events, i_slew):

    t0 = Time(events[i_slew].begin, format='isot', scale='utc')
    t0 = pd.to_datetime(t0.value, utc=True) # astropy Time to Timestamp conversion
    t1 = Time(events[i_slew].end, format='isot', scale='utc')
    t1 = pd.to_datetime(t1.value, utc=True) # astropy Time to Timestamp conversion

    return (t0,t1)

def getIMSdata(events, i_slew, postPadding):
    client = makeEfdClient()
    df_ims = getEfdData(client, 'lsst.sal.MTM1M3.imsData', 
                        event=events[i_slew], 
                        postPadding = postPadding)
    return df_ims
    
def runTestSettlingTime(dayObs, postPadding, block, outdir, f, verbose):
    '''
    Function to run the settling time statistics test over a complete block on a given night
    Parameters:
    dayObs: corresponding observation day to analyze
    postPadding: number of seconds after slew stop in which to perform the analysis
    block: observation block number
    outdir: directory to store plots and log outputs
    f: file handle for log output 
    verbose: flag for output verbosity
    '''
    
    req_rms_position = 2e-3 ## mm, tolerance from repeatability requirement for IMS positional
    req_rms_rotation = 3e-5 ## degrees (1arcsec is 27e-5), tolerance from repeatability requirement for IMS rotational

    # Select data from a given date
    eventMaker = TMAEventMaker()
    events = eventMaker.getEvents(dayObs)
    
    # Define columns
    all_columns = ["xPosition","yPosition","zPosition","xRotation","yRotation","zRotation"] 
    pos_columns = [c for c in all_columns if "Position" in c]
    rot_columns = [c for c in all_columns if "Rotation" in c]

    # Get lists of slew and track events
    slews = [e for e in events if e.type==TMAState.SLEWING]
    tracks = [e for e in events if e.type==TMAState.TRACKING]
    mess = f'Found {len(slews)} slews and {len(tracks)} tracks'
    logMessage(mess, f)
    
    blockParser = BlockParser(dayObs)
    print(f"Found blocks for {dayObs}: {blockParser.getBlockNums()}")
    blockNums = blockParser.getBlockNums()
    
    blockEvents = []
    for event in events:
        blockInfos = event.blockInfos
        if blockInfos is None:
            continue  # no block info attached to event at all
        # check if any of the attached blockInfos are for blockEvents
        blockNums = {b.blockNumber for b in blockInfos}
        if block in blockNums:
            blockEvents.append(event)

    logMessage(f"Of the {len(events)} events, {len(blockEvents)} relate to block {block}", f)

    rmsPosAtReqAgg = []
    meanPosAtReqAgg = []
    meanXPosAtReqAgg = []
    meanYPosAtReqAgg = []
    meanZPosAtReqAgg = []
    rmsRotAtReqAgg = []
    meanRotAtReqAgg = []
    settleTimePosAgg = []
    settleTimeRotAgg = []
    failsAgg = []
    settleTimeXPosAgg = []
    settleTimeYPosAgg = []
    settleTimeZPosAgg = []
    settleTimeXRotAgg = []
    settleTimeYRotAgg = []
    settleTimeZRotAgg = []

    ignoreList = [92, 120, 274] #these are specific seqNums to ignore 
   
    for i in range(len(blockEvents)):
        #print(TMAState.TRACKING, TMAState.SLEWING, blockEvents[i].endReason, blockEvents[i].type)
        if (blockEvents[i].endReason == TMAState.TRACKING and blockEvents[i].type == TMAState.SLEWING):
            single_slew = blockEvents[i].seqNum  
            if single_slew in ignoreList:
                logMessage(f"Skipping {single_slew}", f)
                continue #e.g. 92 is badly identified in 20231220
            logMessage(f'Will look at slew {single_slew}',f)
            t0,t1 = getSlewTimes(events, single_slew)
            df_ims = getIMSdata(events, single_slew, postPadding)
            df_ims = df_ims[all_columns]
            # Convert meter to milimeter 
            df_ims[pos_columns] = df_ims[pos_columns] * 1e3
            allcolPF = True #flag to detect whether any column has failed the test
            fails = 0
            for col in all_columns:
                if col in pos_columns:
                    req = req_rms_position
                else:
                    req = req_rms_rotation
                PF, settleInterval, rmsAtReq, meanAtReq = computeSettleTime(df_ims=df_ims,                                                                referenceTime=t1, 
                                                lo_delta_t=5,hi_delta_t=postPadding, 
                                                imsColumn=col, rmsReq=req, 
                                                chi2prob=0.99, f = f, verbose = verbose)
                if settleInterval >= 0:
                    logMessage(f"{col} settled in {settleInterval:.2f} s",f)
                else:
                    logMessage(f"{col} not settled in {postPadding} s",f)
                if PF == False:
                    allcolPF = False
                    fails = fails + 1
                if col in pos_columns:
                    rmsPosAtReqAgg.append(rmsAtReq)
                    meanPosAtReqAgg.append(meanAtReq)
                    settleTimePosAgg.append(settleInterval)
                    if col == 'xPosition':
                        meanXPosAtReqAgg.append(meanAtReq)
                        settleTimeXPosAgg.append(settleInterval)
                    if col == 'yPosition':
                        meanYPosAtReqAgg.append(meanAtReq)
                        settleTimeYPosAgg.append(settleInterval)
                    if col == 'zPosition':
                        meanZPosAtReqAgg.append(meanAtReq)
                        settleTimeZPosAgg.append(settleInterval)
                else:
                    rmsRotAtReqAgg.append(rmsAtReq)
                    meanRotAtReqAgg.append(meanAtReq)
                    settleTimeRotAgg.append(settleInterval)
                    if col == 'xRotation':
                        settleTimeXRotAgg.append(settleInterval)
                    if col == 'yRotation':
                        settleTimeYRotAgg.append(settleInterval)
                    if col == 'zRotation':
                        settleTimeZRotAgg.append(settleInterval)
            if allcolPF == False:
                logMessage(f"Event {single_slew} has {fails} failure(s)",f)
            failsAgg.append(fails)

        #if i > 2:
        #    break

    title = f"Settle test for block {block} on {dayObs}"
    plt.hist(failsAgg,bins=[0,1,2,3,4,5,6,7])
    plt.title(title+" Number of failures per event")
    plt.ylabel('Number of events')
    plt.xlabel('Number of failures')
    plt.legend()
    plt.savefig(outdir+'/nb_failures.png')
    
    plt.clf()
    plt.hist(settleTimePosAgg,bins=50)
    plt.title(title+" settle time for position")
    plt.ylabel('Number of events (all axes)')
    plt.xlabel('Settling time (s)')
    plt.legend()
    plt.savefig(outdir+'/settletime_position.png')

    plt.clf()
    plt.hist(settleTimeXPosAgg,bins=50,alpha=0.5,color='red',ls='dashed',label='xPosition')
    plt.hist(settleTimeYPosAgg,bins=50,alpha=0.5,color='green',ls='dashed',label='yPosition')
    plt.hist(settleTimeZPosAgg,bins=50,alpha=0.5,color='black',ls='dashed',label='zPosition') 
    plt.axvline(1, lw="1.25", c="k", ls="dashed", label="Requirement")
    plt.title(title+" settle time (per axis)")
    plt.xlabel('Settle time (s)')
    plt.legend()
    plt.savefig(outdir+'/settletime_position_xyz.png')

    plt.clf()
    plt.hist(settleTimeXRotAgg,bins=50,alpha=0.5,color='red',ls='dashed',label='xRotation')
    plt.hist(settleTimeYRotAgg,bins=50,alpha=0.5,color='green',ls='dashed',label='yRotation')
    plt.hist(settleTimeZRotAgg,bins=50,alpha=0.5,color='black',ls='dashed',label='zRotation') 
    plt.axvline(1, lw="1.25", c="k", ls="dashed", label="Requirement")
    plt.title(title+" settle time (per axis)")
    plt.xlabel('Settle time (s)')
    plt.legend()
    plt.savefig(outdir+'/settletime_rotation_xyz.png')

    plt.clf()    
    plt.hist(settleTimeRotAgg,bins=50)
    plt.title(title+" settle time for rotation")
    plt.ylabel('Number of events (all axes)')
    plt.xlabel('Settling time (s)')
    plt.legend()
    plt.savefig(outdir+'/settletime_rotation.png')

    plt.clf()
    plt.hist(rmsPosAtReqAgg,bins=50)
    plt.axvline(req_rms_position, lw="1.25", c="k", ls="dashed", label="Requirement")
    plt.title(title+" position RMS")
    plt.xlabel('IMS position RMS wrt settled, at 1 s after stop (mm)')
    plt.legend()
    plt.savefig(outdir+'/rms_position.png')

    plt.clf()
    plt.hist(rmsRotAtReqAgg,bins=50)
    plt.axvline(req_rms_rotation, lw="1.25", c="k", ls="dashed", label="Requirement")
    plt.title(title+" rotation RMS")
    plt.xlabel('IMS rotation RMS wrt settled, at 1 s after stop (deg)')
    plt.legend()
    plt.savefig(outdir+'/rms_rotation.png')

    plt.clf()
    plt.hist(meanPosAtReqAgg,bins=50)
    plt.axvline(req_rms_position, lw="1.25", c="k", ls="dashed", label="Requirement")
    plt.title(title+" position bias")
    plt.xlabel('IMS position BIAS wrt settled, at 1 s after stop (mm)')
    plt.legend()
    plt.savefig(outdir+'/mean_position.png')

    plt.clf()
    plt.hist(meanXPosAtReqAgg,bins=50,alpha=0.5,color='red',ls='dashed',label='xPosition')
    plt.hist(meanYPosAtReqAgg,bins=50,alpha=0.5,color='green',ls='dashed',label='yPosition')
    plt.hist(meanZPosAtReqAgg,bins=50,alpha=0.5,color='black',ls='dashed',label='zPosition')    
    plt.axvline(req_rms_position, lw="1.25", c="k", ls="dashed", label="Requirement")
    plt.title(title+" position bias (per axis)")
    plt.xlabel('IMS position BIAS wrt settled, at 1 s after stop (mm)')
    plt.legend()
    plt.savefig(outdir+'/mean_position_xyz.png')

    plt.clf()
    plt.hist(meanRotAtReqAgg,bins=50)
    plt.axvline(req_rms_rotation, lw="1.25", c="k", ls="dashed", label="Requirement")
    plt.title(title+" rotation bias")
    plt.xlabel('IMS rotation BIAS wrt settled, at 1 s after stop (deg)')
    plt.legend()
    plt.savefig(outdir+'/mean_rotation.png')

    return 0

def main():
    '''
    Run code with options
    '''
    usage = "%prog [options]"
    parser = OptionParser(usage=usage)
    parser.add_option("--dayObs",dest="dayObs",help="Date for observations or measurements",default=20231222, type= "int")
    parser.add_option("--block",dest="block",help="Set block value",default=146, type="int")
    parser.add_option("--padding",dest="postPadding",help="Seconds to analyze after slew stop",default=15)
    parser.add_option("--outdir",dest="outdir",help="Output directory for results",default='./SITCOM-1172_out')
    (options, args) = parser.parse_args()
    
    if not os.path.exists(options.outdir):
        os.makedirs(options.outdir)

    f = open(options.outdir+"/SITCOM_1172.log","a")

    c = datetime.now()
    timeStamp = c.strftime('%H:%M:%S')
    logMessage(f"Running runTestSettlingTime at {timeStamp}",f)

    result = runTestSettlingTime(options.dayObs, options.postPadding, options.block, options.outdir, f, verbose = True)
        
    logMessage(f"Test result {result}. Check outputs in {options.outdir}",f)
        
if __name__ == "__main__":
    main()



