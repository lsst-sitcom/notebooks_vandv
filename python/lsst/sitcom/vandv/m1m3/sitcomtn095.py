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

def check_requirement_threshold(
    reference_time, 
    settling_margin, 
    df_ims, 
    ims_column, 
    include_bias,
    corrected_variable, 
    req, 
    log_file, 
    verbose,
):
    """Function to check requirement for RMS and bias after allowed margin until the end of the check interval

    Parameters
    ----------
    reference_time: str
        reference time in UTC
    settling_margin: float
        number of seconds after reference time allowed for settling
    df_ims: pandas data frame
        IMS (Independent Measurement System of the M1M3) data
    ims_column: str
        specific IMS column to analyze in this function call
    include_bias: bool
        include abs(mean) (not only RMS) in check of requirement during check_interval
    corrected_variable:
        column value corrected for the value at the end of the window, 
        determined before the function call
    req: float
        tolerance for IMS value to be within requirement. We will apply it 
        to the RMS and bias of the value, with respect to the value at 
        reference_t + post_delta_t
    log_file: str
        file name for logs
    verbose: bool
        flag for verbosity in outputs

    Returns:
    --------
    PF: bool
        PASS/FAIL for this column and event, with a window of check_interval seconds after reference time, 
        checking RMS and bias
    settle_time: float
        computed as latest time in which the IMS value has gone above the 
        requirement in RMS or mean
    """
    ## recomputing RMS for the whole range since slew_ref_t
    rolling = 20
    time_array = df_ims.index
    slew_ref = Time(reference_time).unix
    index_slew_ref = np.argmin(
        np.abs((Time(time_array).unix) - slew_ref)
    )  # which index in time_array is closest to slew_ref
    index_slew_ref_margin = np.argmin(
        np.abs((Time(time_array).unix) - (Time(time_array).unix[index_slew_ref] + settling_margin))
    )  # which index in time_array is closest to slew_ref + 1 second
    rms = (corrected_variable).rolling(rolling).std()
    mean = abs((corrected_variable).rolling(rolling).mean())
    krms = [index for index, x in enumerate(rms) if np.abs(x) >= req] # all bad events in RMS
    kmean = [index for index, x in enumerate(mean) if np.abs(x) >= req] # all bad events in bias
    condition_rms = all(x < req for x in rms[index_slew_ref_margin - index_slew_ref : -1])
    condition_bias = all(x < req for x in mean[index_slew_ref_margin - index_slew_ref : -1])
    if (condition_rms) and (condition_bias):
        if (krms == []) and (kmean == []):  # both RMS and mean comply with requirement since ref time
            settle_time = 0  # already settled at reference time
        else:
            if krms == []:  # only RMS values comply with requirement since reference time
                if include_bias:
                    settle_time = time_array[index_slew_ref + kmean[-1]] - time_array[index_slew_ref]
                    settle_time = settle_time.total_seconds()
                else:
                    settle_time = 0
            elif kmean == []:  # only mean values comply with requirement
                settle_time = time_array[index_slew_ref + krms[-1]] - time_array[index_slew_ref]
                settle_time = settle_time.total_seconds()
            else:  # neither comply with requirement for all times since reference time, take maximum
                if include_bias:
                    settle_time = (
                        max(time_array[index_slew_ref + krms[-1]], time_array[index_slew_ref + kmean[-1]])
                        - time_array[index_slew_ref]
                    )
                else:
                    settle_time = time_array[index_slew_ref + krms[-1]] - time_array[index_slew_ref]
                settle_time = settle_time.total_seconds()

        PF = True
        if verbose:
            log_message(f"Test PASSED. Settle_time:{settle_time} s", log_file, verbose)
    else:
        if krms == []:
            if include_bias:
                settle_time = time_array[index_slew_ref + kmean[-1]] - time_array[index_slew_ref]
                settle_time = settle_time.total_seconds()
                PF = False
            else:
                settle_time = 0 #because we disregard any bias and RMS is below requirement at all times
                PF = True
        elif kmean == []:
            settle_time = time_array[index_slew_ref + krms[-1]] - time_array[index_slew_ref]
            settle_time = settle_time.total_seconds()
            PF = False
        else:
            if include_bias:
                settle_time = (
                    max(time_array[index_slew_ref + krms[-1]], time_array[index_slew_ref + kmean[-1]])
                    - time_array[index_slew_ref]
                )
                PF = False
            else:
                settle_time = time_array[index_slew_ref + krms[-1]] - time_array[index_slew_ref]
                if condition_rms:
                    PF = True
                else:
                    PF = False
            settle_time = settle_time.total_seconds()
        if PF == False:
            log_message(f"Test FAILED for column {ims_column} with settling time {settle_time} s", log_file, verbose)
        else:
            log_message(f"Test PASSED. Settle_time:{settle_time} s", log_file, verbose)
    return PF, settle_time


def compute_settle_time(
    df_ims,  
    reference_time="2023-06-01T06:00:0Z",  
    settling_margin=5,
    lo_delta_t=5,  
    hi_delta_t=30,  
    ims_column="xPosition",  
    include_bias=True,
    rms_req=2e-3, 
    chi2_prob=0.999,  
    log_file="SITCOMTN_095.log",
    verbose=False,
):
    """Function to compute settle time and PASS/FAIL for a given event

    Parameters:
    -----------
    df_ims: pandas data frame
        IMS (Independent Measurement System of the M1M3) data
    reference_time: str
        reference time in UTC
    settling_margin: float
        seconds after reference time in which system is allowed to settle
    lo_delta_t: int
        time window in seconds BEFORE the reference time to retrieve data
    hi_delta_t: int
        time window in seconds AFTER the reference time to retrieve data
    ims_column: str
        specific IMS column to analyze in this function call
    include_bias: bool
        include abs(mean) (not only RMS) in check of requirement during check_interval
    rms_req: float
        tolerance for IMS value to be within requirement. We will apply it 
        to the RMS and bias of the value, with respect to the value at 
        reference_t + post_delta_t
    chi2_prob: float
        confidence level for IMS variable wrt to long term value and variance 
        to agree. If None, settle time will be computed by checking last time requirement fails for any column
    log_file: str
        file name for logs
    verbose: bool
        Verbosity flag of log outputs

    Returns:
    --------
    PF: bool
        PASS/FAIL for this column and event
    settle_time: float
        the time for settling counting from reference time
    """

    if "Position" in ims_column:
        units = "mm"
        ylimMax = rms_req + 0.005
    elif "Rotation" in ims_column:
        units = "deg"
        ylimMax = rms_req + 0.0001
    else:
        print("Unidentified column")
        return -1

    settle_time = False

    slew_ref_t = pd.to_datetime(reference_time)  
    delta_window = [
        pd.Timedelta(lo_delta_t, "seconds"),
        pd.Timedelta(hi_delta_t, "seconds"),
    ]
    # target_variable_check takes the data from the reference time, until the end of the plot
    target_variable_check = df_ims[ims_column][slew_ref_t : slew_ref_t + delta_window[1]]
    idx_t0 = df_ims.index[  # index in dataframe closest in time to reference time
        df_ims.index.get_indexer([pd.to_datetime(slew_ref_t)], method="nearest")
    ]
    idx_tend = df_ims.index[  # index in dataframe closest in time to end of plot
        df_ims.index.get_indexer(
            [pd.to_datetime(slew_ref_t + delta_window[1])], method="nearest"
        )
    ]
    if len(target_variable_check.index) == 0:
        print("Data frame is empty")
        return -1
    target_variable_reference = float(df_ims[ims_column][idx_tend])

    # it is important that the end of the plot (target_variable_reference)
    # does not hit another slew or movement, nor at any point in the middle of the window

    # correct IMS variable wrt end of plot
    corrected_variable_check = target_variable_check - target_variable_reference

    if chi2_prob is None:
        PF, settle_time = check_requirement_threshold(
            reference_time, 
            settling_margin,
            df_ims, 
            ims_column,
            include_bias,
            corrected_variable_check, 
            rms_req, 
            log_file, 
            verbose
            )
    else:
        #PF, settle_time = check_requirement_chi2test()
        
        # number of values where the chi2 will be computed
        rolling = 30  # 50 is approx. 1 s
        # chi2 right tail probability for N=rolling dof at chi2_prob CL
        crit = stats.chi2.ppf(chi2_prob, rolling)

        rms = corrected_variable_check.rolling(rolling, center = True).std()
        var = corrected_variable_check.rolling(rolling, center = True).var()
        mean = abs(corrected_variable_check.rolling(rolling, center = True).mean())

        # compute the chi2 against the null hypothesis
        # the x_i are the measurements in a window (wrt to reference at end of plot)
        # the variance is for the same values
        # so chi2 = sum_N[(x_i - 0)**2/variance] where N = rolling
        sum2 = corrected_variable_check2.rolling(rolling, center = True).sum()
        chi2 = sum2 / var

        # check the chi2 at each step using rolling_check as the number of consecutive instances in which
        # chi2 has to be under the critical value
        # or rms and bias be both already 10% of requirement
        PFCheck = (chi2 < crit) | ((rms < 0.1 * rms_req) & (mean < 0.1 * rms_req))
 
        rolling_check = 30
        stability_check = (
            PFCheck.rolling(rolling_check, center = True).apply(lambda s: s.all()) > 0
        )  # true if rolling_check consecutive true values of PFcheck

        if len(stability_check[stability_check == True]) <= rolling_check:  ## == 0:
            # print(f"Not settled within {postPadding} s window")
            settle_time = False
        elif rms[stability_check[stability_check == True].index[0]] <= rms_req:
            settle_time = stability_check[stability_check == True].index[rolling_check]
        else:
            n = 1
            while (
                rms[stability_check[stability_check == True].index[n + rolling_check]] > rms_req
            ):
                settle_time = stability_check[stability_check == True].index[n + rolling_check]
                n = n + 1
        
        settle_interval = -1
        if settle_time:
            settle_interval = settle_time - reference_time
            if settle_interval.total_seconds() < 0:
                print(f"Already settled at reference time")
                settle_interval = 0
            else:
                settle_interval = settle_interval.total_seconds()
        PF = True

    return PF, settle_time


def log_message(mess, log_file,verbose):
    """Function to write a message in a log file

    Parameters:
    -----------
    mess: str
        message to be written
    log_file: str
        path to the log file
    verbose: bool
        screen output verbosity
    """
    if verbose:
        print(mess)
    try:
        print(mess, file=log_file)
    except:
        print("Could not write to log file")
    return


def get_slew_times(events, i_slew):
    """Function to return the beginning and the end time of a slew
    
    Parameters:
    -----------
    events: 
        list of slews
    i_slew: int
        index of the slew for which the beginning and end time will be determined

    Returns:
    --------
    t_start: 
        timestamp corresponding to the beginning of the slew
    t_end:
        timestamp corresponding to the end of the slew
    """
    t_start = Time(events[i_slew].begin, format="isot", scale="utc")
    t_start= pd.to_datetime(t_start.value, utc=True)  # astropy Time to Timestamp conversion
    t_end = Time(events[i_slew].end, format="isot", scale="utc")
    t_end = pd.to_datetime(t_end.value, utc=True)  # astropy Time to Timestamp conversion

    return (t_start, t_end)


def get_IMS_data(client, events, single_slew, check_interval):
    """ Function to return the IMS dataframe corresponding to a slew duration 
    plus an extra time

    Parameters:
    -----------
    client:
        EFD client instance
    events: 
        list of slews
    single_slew: int
        index of the slew for which the beginning and end time will be determined
    check_interval: float
        extra time to be added after the reference time to get IMS data

    Returns:
    --------
    df_ims: pandas dataframe
        dataframe containing the IMS data
    """
    #client = makeEfdClient()
    df_ims = getEfdData(
        client, "lsst.sal.MTM1M3.imsData", event=events[single_slew], postPadding=check_interval
    )
    return df_ims

def get_azel_data(client, events, single_slew, check_interval):
    """ Function to return the azimuth, elevation corresponding to an interval of time around a slew

    Parameters:
    -----------
    client:
        EFD client instance
    events: 
        list of slews
    single_slew: int
        index of the slew for which the beginning and end time will be determined
    check_interval: float
        extra time to be added after the reference time to get IMS data

    Returns:
    --------
    az, el: pandas dataframe
        dataframe containing the IMS data
    """
    df_mtmount_ele = getEfdData(
        client,
        "lsst.sal.MTMount.elevation",
        event=events[single_slew],
        postPadding=check_interval,
    )
    df_mtmount_azi = getEfdData(
        client,
        "lsst.sal.MTMount.azimuth",
        event=events[single_slew],
        postPadding=check_interval,
    )
    return df_mtmount_azi, df_mtmount_ele

def run_test_settling_time(dayObs, check_interval, block, outdir, log_file, verbose):
    """Function to run the settling time statistics test over a complete 
    block on a given night

    Parameters:
    -----------
    dayObs: int
        corresponding observation day to analyze
    check_interval: int
        number of seconds after reference time in which to perform the analysis
    block: int
        observation block number
    outdir: str
        directory to store plots and log outputs
    log_file: str
        file handle for log output
    verbose: bool
        flag for screen output verbosity
    """

    req_rms_position = (
        2e-3  ## mm, tolerance from repeatability requirement for IMS positional
    )
    req_rms_rotation = 3e-5  ## degrees (1 arcsec is 27e-5), tolerance from repeatability requirement for IMS rotational
    settling_margin = 5 # seconds after slew start

    # Select data from a given date
    eventMaker = TMAEventMaker()
    events = eventMaker.getEvents(dayObs)

    # Define columns
    all_columns = [
        "xPosition",
        "yPosition",
        "zPosition",
        "xRotation",
        "yRotation",
        "zRotation",
    ]
    pos_columns = [c for c in all_columns if "Position" in c]
    rot_columns = [c for c in all_columns if "Rotation" in c]

    # Get lists of slew and track events
    slews = [e for e in events if e.type == TMAState.SLEWING]
    tracks = [e for e in events if e.type == TMAState.TRACKING]
    mess = f"Found {len(slews)} slews and {len(tracks)} tracks"
    log_message(mess, log_file, verbose)

    block_parser = BlockParser(dayObs)
    print(f"Found blocks for {dayObs}: {block_parser.getBlockNums()}")
    blockNums = block_parser.getBlockNums()

    block_events = []
    for event in events:
        blockInfos = event.blockInfos
        if blockInfos is None:
            continue  # no block info attached to event at all
        # check if any of the attached blockInfos are for block_events
        blockNums = {b.blockNumber for b in blockInfos}
        if block in blockNums:
            block_events.append(event)

    log_message(
        f"Of the {len(events)} events, {len(block_events)} relate to block {block}", log_file, verbose
    )

    fails_agg = []
    settle_time_xpos_agg = []
    settle_time_ypos_agg = []
    settle_time_zpos_agg = []
    settle_time_xrot_agg = []
    settle_time_yrot_agg = []
    settle_time_zrot_agg = []
    az_agg = []
    el_agg = []
    az_ypos_fails_agg = []
    az_yrot_fails_agg = []
    el_ypos_fails_agg = []
    el_yrot_fails_agg = []

    ignoreList = [92, 120, 274]  # these are specific seqNums to ignore

    client = makeEfdClient()

    for i in range(len(block_events)):
#    for i in range(10):
        if (
            block_events[i].endReason == TMAState.TRACKING
            and block_events[i].type == TMAState.SLEWING
        ):
            single_slew = block_events[i].seqNum
            if single_slew in ignoreList:
                log_message(f"Skipping {single_slew}", log_file, verbose)
                continue  # e.g. 92 is badly identified in 20231220
            log_message(f"Will look at slew {single_slew}", log_file, verbose)
            slew_start, slew_stop = get_slew_times(events, single_slew)
            df_mtmount_azi, df_mtmount_ele = get_azel_data(client, events, single_slew, check_interval)
            az_agg.append(df_mtmount_azi["actualPosition"].mean())
            el_agg.append(df_mtmount_ele["actualPosition"].mean())
            df_ims = get_IMS_data(client, events, single_slew, check_interval)
            df_ims = df_ims[all_columns]
            # Convert meter to milimeter
            df_ims[pos_columns] = df_ims[pos_columns] * 1e3
            all_col_pass = True  # flag to detect whether any column has failed the test
            fails = 0
            for col in all_columns:
                if col in pos_columns:
                    req = req_rms_position
                else:
                    req = req_rms_rotation
                PF, settle_interval = compute_settle_time(
                    df_ims=df_ims,
                    reference_time=slew_start,
                    settling_margin=settling_margin,
                    lo_delta_t=5,
                    hi_delta_t=check_interval,
                    ims_column=col,
                    include_bias=False,
                    rms_req=req,
                    chi2_prob=None, #fixing to None for now
                    log_file=log_file,
                    verbose=verbose,
                )

                if settle_interval >= 0:
                    log_message(f"{col} settled in {settle_interval:.2f} s", log_file, verbose)
                else:
                    log_message(f"{col} not settled in {postPadding} s", log_file, verbose)

                if PF == False:
                    all_col_pass = False
                    fails = fails + 1
                    if col == "yPosition":
                        az_ypos_fails_agg.append(df_mtmount_azi["actualPosition"].mean())
                        el_ypos_fails_agg.append(df_mtmount_ele["actualPosition"].mean())
                    if col == "yRotation":
                        az_yrot_fails_agg.append(df_mtmount_azi["actualPosition"].mean())
                        el_yrot_fails_agg.append(df_mtmount_ele["actualPosition"].mean())

                if col in pos_columns:
                    if col == "xPosition":
                        settle_time_xpos_agg.append(settle_interval)
                    if col == "yPosition":
                        settle_time_ypos_agg.append(settle_interval)
                    if col == "zPosition":
                        settle_time_zpos_agg.append(settle_interval)
                else:
                    if col == "xRotation":
                        settle_time_xrot_agg.append(settle_interval)
                    if col == "yRotation":
                        settle_time_yrot_agg.append(settle_interval)
                    if col == "zRotation":
                        settle_time_zrot_agg.append(settle_interval)

            if all_col_pass == False:
                log_message(f"Event {single_slew} has {fails} failure(s)", log_file, verbose)
            fails_agg.append(fails)

    title = f"Settle test for block {block} on {dayObs}"

    plt.hist(fails_agg, bins=[0, 1, 2, 3, 4, 5, 6, 7])
    plt.title(title + " Number of failures per event")
    plt.ylabel("Number of events")
    plt.xlabel("Number of failures")
    plt.legend()
    plt.savefig(outdir + "/nb_failures.png")

    plt.clf()
    plt.hist(
        settle_time_xpos_agg,
        bins=50,
        alpha=0.5,
        color="red",
        ls="dashed",
        label="xPosition",
    )
    plt.hist(
        settle_time_ypos_agg,
        bins=50,
        alpha=0.5,
        color="green",
        ls="dashed",
        label="yPosition",
    )
    plt.hist(
        settle_time_zpos_agg,
        bins=50,
        alpha=0.5,
        color="black",
        ls="dashed",
        label="zPosition",
    )
    plt.axvline(settling_margin, lw="1.25", c="k", ls="dashed", label="Requirement")
    plt.title(title + " settle time (per axis)")
    plt.xlabel("Settle time (s)")
    plt.legend()
    plt.savefig(outdir + "/settletime_position_xyz.png")

    plt.clf()
    plt.hist(
        settle_time_xrot_agg,
        bins=50,
        alpha=0.5,
        color="red",
        ls="dashed",
        label="xRotation",
    )
    plt.hist(
        settle_time_yrot_agg,
        bins=50,
        alpha=0.5,
        color="green",
        ls="dashed",
        label="yRotation",
    )
    plt.hist(
        settle_time_zrot_agg,
        bins=50,
        alpha=0.5,
        color="black",
        ls="dashed",
        label="zRotation",
    )
    plt.axvline(settling_margin, lw="1.25", c="k", ls="dashed", label="Requirement")
    plt.title(title + " settle time (per axis)")
    plt.xlabel("Settle time (s)")
    plt.legend()
    plt.savefig(outdir + "/settletime_rotation_xyz.png")

    plt.clf()
    plt.scatter(az_agg,el_agg,marker='.',color='black',label='Slew event')
    plt.scatter(az_ypos_fails_agg,el_ypos_fails_agg,marker='o',color='blue',label='yPosition fails')
    plt.scatter(az_yrot_fails_agg,el_yrot_fails_agg,marker='x',color='red',label='yRotation fails')
    plt.title("Azimuth/Elevation distribution of soak events")
    plt.xlabel("Azimuth (degrees)")
    plt.ylabel("Elevation (degrees)")
    plt.legend(loc="lower right")
    plt.savefig(outdir + "/azel.png")

    return 0


def main():
    """Run code with options"""
    usage = "%prog [options]"
    parser = OptionParser(usage=usage)
    parser.add_option(
        "--dayObs",
        dest="dayObs",
        help="Date for observations or measurements",
        default=20231220,
        type="int",
    )
    parser.add_option(
        "--block", 
        dest="block", 
        help="Set block value", 
        default=146, 
        type="int",
    )
    parser.add_option(
        "--check_interval",
        dest="check_interval",
        help="Seconds to analyze after reference time",
        default=20,
    )
    parser.add_option(
        "--outdir",
        dest="outdir",
        help="Output directory for results",
        default="./SITCOMTN-095_out",
    )
    parser.add_option(
        "--verbose",
        dest="verbose",
        help="Log message verbosity in screen output",
        default=True,
    )

    (options, args) = parser.parse_args()

    if not os.path.exists(options.outdir):
        os.makedirs(options.outdir)

    log_file = open(options.outdir + "/SITCOMTN-095.log", "a")

    c = datetime.now()
    timeStamp = c.strftime("%H:%M:%S")
    log_message(f"Running run_test_settling_time at {timeStamp}", log_file, True)

    result = run_test_settling_time(
        options.dayObs,
        options.check_interval,
        options.block,
        options.outdir,
        log_file,
        verbose=options.verbose,
    )

    log_message(f"Test result {result}. Check outputs in {options.outdir}", log_file, True)


if __name__ == "__main__":
    main()
