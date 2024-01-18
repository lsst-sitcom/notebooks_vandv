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

def computeSettleTime(
    df_ims,  # input data frame
    referenceTime="2023-06-01T06:00:0Z",  # time for slew stop (T0)
    lo_delta_t=5,  # in seconds
    hi_delta_t=30,  # in seconds
    imsColumn="xPosition",  # IMS column
    rmsReq=2e-3,  # requirement in appropriate units
    req_delta_t=3,  # time for settling, in seconds
    chi2prob=0.999,  # confidence level for IMS variable wrt to long term value and variance to agree
):
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

    # T0 and T1 defines the window for requirement to be met
    T0 = pd.to_datetime(referenceTime)  # this is slew stop
    T1 = T0 + pd.to_timedelta(
        req_delta_t, unit="s"
    )  # this is the end of maximum req. window
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
    # targetVariableWindow takes the data from the slew stop, until the end of requirement window
    targetVariableWindow = df_ims[imsColumn][T0:T1]
    idxT0 = df_ims.index[  # index in dataframe closest in time to slew stop
        df_ims.index.get_indexer([pd.to_datetime(T0)], method="nearest")
    ]
    idxT1 = (
        df_ims.index[  # index in dataframe closest in time to end of requirement window
            df_ims.index.get_indexer([pd.to_datetime(T1)], method="nearest")
        ]
    )
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
    rolling = 10  # 50 is approx. 1 s
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
    rollingCheck = 10
    stabilityCheck = (
        PFCheck.rolling(rollingCheck).apply(lambda s: s.all()) > 0
    )  # true if rollingCheck consecutive true values of PFcheck
    if len(stabilityCheck[stabilityCheck == True]) < rollingCheck:  ## == 0:
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

    title = imsColumn
    fig = plt.figure()
    label = "Corrected " + imsColumn + "(" + units + ") difference wrt end of plot"
    plt.plot(
        correctedVariablePlot,
        color="red",
        ls="dashed",
        lw="0.5",
        label=label,
    )
    plt.plot(rms, label=f"RMS of corrected value, using {rolling} rolling values")
    plt.plot(
        mean,
        ls="dashed",
        label=f"Bias of corrected value, using {rolling} rolling values",
    )
    if settleTime:
        plt.arrow(
            settleTime,
            1.1 * rmsReq,
            0.0,
            -0.5 * rmsReq,
            width=2e-6,
            head_length=0.5 * rmsReq,
            label = 'Settling achieved'
        )
        plt.text(
            0.5,
            0.92,
            "Settle time =" + " {:.2f} ".format(settleInterval) + " s",
            transform=fig.transFigure,
        )
    plt.axvline(T0, lw="1.25", c="k", ls="dashed", label="Slew stop")
    plt.axhline(-rmsReq, lw="0.75", c="k", ls="dashed", label="IMS repeatability req.")
    plt.axhline(rmsReq, lw="0.75", c="k", ls="dashed")
    plt.xlabel("Time [UTC]")
    plt.ylabel(f"{imsColumn} {units}")
    plt.ylim(-ylimMax, ylimMax)
    fig.autofmt_xdate()
    plt.legend(loc="upper right", fontsize="8")
    fig.tight_layout()

    if not settleTime:
        return -1

    return settleInterval
    
def logMessage(mess, f):
    print(mess)
    try:
        print(mess, file=f)
    except:
        print("Could not write to log file")
    return

def getSlewTimes(slews, i_slew):

    t0 = Time(slews[i_slew].begin, format='isot', scale='utc')
    t0 = pd.to_datetime(t0.value, utc=True) # astropy Time to Timestamp conversion
    t1 = Time(slews[i_slew].end, format='isot', scale='utc')
    t1 = pd.to_datetime(t1.value, utc=True) # astropy Time to Timestamp conversion

    return (t0,t1)

def getIMSdata(slews, i_slew, postPadding):
    client = makeEfdClient()
    df_ims = getEfdData(client, 'lsst.sal.MTM1M3.imsData', 
                        event=slews[i_slew], 
                        postPadding = postPadding)
    return df_ims
    
def runTestSettlingTime(dayObs, postPadding, block, outdir):
    
    client = makeEfdClient()
    f = open("SITCOM_1172.log","a")

    req_delta_t = 3 ## seconds after slew
    req_rms_position = 2e-3 ## mm, tolerance from repeatability requirement for IMS positional
    req_rms_rotation = 3e-5 ## degrees, tolerance from repeatability requirement for IMS rotational

    # Select data from a given date
    eventMaker = TMAEventMaker()
    events = eventMaker.getEvents(dayObs)
    
    # Define columns
    all_columns = ["xPosition", "xRotation", "yPosition", "yRotation", "zPosition", "zRotation"]
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

    print(f"Of the {len(events)} events, {len(blockEvents)} relate to block {block}")

    for i in range(len(blockEvents)):
        single_slew = blockEvents[i].seqNum        
        logMessage(f'Will look at slew {single_slew}',f)
        t0,t1 = getSlewTimes(slews, single_slew)
        df_ims = getIMSdata(slews, single_slew, postPadding)
        df_ims = df_ims[all_columns]
        # Convert meter to milimeter 
        df_ims[pos_columns] = df_ims[pos_columns] * 1e3        
        for col in all_columns:
            if col in pos_columns:
                req = req_rms_position
            else:
                req = req_rms_rotation
            settle_interval = computeSettleTime(df_ims=df_ims, referenceTime=t1,
                                                lo_delta_t=5,hi_delta_t=postPadding, 
                                                imsColumn=col, rmsReq=req, 
                                                req_delta_t=req_delta_t, chi2prob=0.999)
            if settle_interval >= 0:
                logMessage(f"{col} settled in {settle_interval:.2f} s")
            else:
                logMessage(f"{col} not settled in {postPadding} s")

        if i > 10:
            break

    return 0

def main():
    '''
    Run code with options
    '''
    usage = "%prog [options]"
    parser = OptionParser(usage=usage)
    parser.add_option("--dayObs",dest="dayObs",help="Date for observations or measurements",default=20231220)
    parser.add_option("--block",dest="block",help="Set block value",default=146)
    parser.add_option("--padding",dest="postPadding",help="Seconds to analyze after slew stop",default=15)
    parser.add_option("--outdir",dest="outdir",help="Output directory for results",default='./SITCOM-1172_out')
    (options, args) = parser.parse_args()
    
    if not os.path.exists(options.outdir):
        os.makedirs(options.outdir)

    c = datetime.now()
    timeStamp = c.strftime('%H:%M:%S')
    
    logMessage(f"Running runTestSettlingTime at {timeStamp}")
    result = runTestSettlingTime(options.dayObs, options.postPadding, options.block, options.outdir)
        
    logMessage(f"Test result {result}. Check outputs in {options.outdir}")
        
if __name__ == "__main__":
    main()



