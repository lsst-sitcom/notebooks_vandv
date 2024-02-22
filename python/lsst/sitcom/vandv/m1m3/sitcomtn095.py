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


def check_requirement(
    reference_time, df_ims, ims_column, corrected_variable, req, f, verbose=False
):
    """Function to check requirement for RMS and bias 1 s after slew stop

    Parameters
    ----------
    reference_time: str
        slew stop time in UTC
    df_ims: pandas data frame
        IMS (Independent Measurement System of the M1M3) data
    ims_column: str
        specific IMS column to analyze in this function call
    corrected_variable:
        column value corrected for the value at the end of the window, 
        determined before the function call
    req: float
        tolerance for IMS value to be within requirement. We will apply it 
        to the RMS and bias of the value, with respect to the value at 
        reference_t + post_delta_t
    f: str
        file name for logs
    verbose: bool
        flag for verbosity in outputs

    Returns:
    --------
    PF: bool
        PASS/FAIL for this column and event, at 1 second after slew stop, 
        checking RMS and bias
    rms_at_req: float
        value of the RMS (jitter) of the column value, using a rolling check, 
        after the slew stop
    mean_at_req: float
        value of the bias (in absolute value) of the column value, using a 
        rolling check, after the slew stop
    settle_time: float
        computed as latest time in which the IMS value has gone above the 
        requirement in RMS or mean
    """
    ## recomputing RMS for the whole range since T0
    rolling = 20
    time_array = df_ims.index
    slew_stop = Time(reference_time).unix
    index_slew_stop = np.argmin(
        np.abs((Time(time_array).unix) - slew_stop)
    )  # which index in time_array is closest to slew_stop
    index_slew_stop_delayed = np.argmin(
        np.abs((Time(time_array).unix) - (Time(time_array).unix[index_slew_stop] + 1))
    )  # which index in time_array is closest to slew_stop + 1 second
    target_variable = df_ims[ims_column][index_slew_stop:-1]
    # rms = (target_variable - target_variable_reference[1]).rolling(rolling).std()
    rms = (corrected_variable).rolling(rolling).std()
    mean = abs((corrected_variable).rolling(rolling).mean())
    rms_at_req = rms[index_slew_stop_delayed - index_slew_stop]
    mean_at_req = mean[index_slew_stop_delayed - index_slew_stop]
    krms = [index for index, x in enumerate(rms) if np.abs(x) >= req]
    kmean = [index for index, x in enumerate(mean) if np.abs(x) >= req]
    if (all(x < req for x in rms[index_slew_stop_delayed - index_slew_stop : -1])) and (
        all(x < req for x in mean[index_slew_stop_delayed - index_slew_stop : -1])
    ):
        # all values of RMS and mean are OK since 1 s after slew stop
        if verbose:
            log_message(f"{ims_column} Test PASSED", f)

        if (krms == []) and (
            kmean == []
        ):  # both RMS and mean comply with requirement at all values
            settle_time = 0  # already settled at slew stop
        else:
            if krms == []:  # only RMS values comply with requirement since slew stop
                settle_time = time_array[index_slew_stop + kmean[-1]] - time_array[index_slew_stop]
            elif kmean == []:  # only mean values comply with requirement
                settle_time = time_array[index_slew_stop + krms[-1]] - time_array[index_slew_stop]
            else:  # neither comply with requirement for all times since slew stop, take maximum
                settle_time = (
                    max(time_array[index_slew_stop + krms[-1]], time_array[index_slew_stop + kmean[-1]])
                    - time_array[index_slew_stop]
                )
            settle_time = settle_time.total_seconds()
        PF = True
    else:
        if rms_at_req > req:
            if verbose:
                log_message(f"{ims_column} Test FAILED in RMS by {rms_at_req-req}", f)

        if mean_at_req > req:
            if verbose:
                log_message(f"{ims_column} Test FAILED in mean by {mean_at_req-req}", f)

        if krms == []:
            settle_time = time_array[index_slew_stop + kmean[-1]] - time_array[index_slew_stop]
        elif kmean == []:
            settle_time = time_array[index_slew_stop + krms[-1]] - time_array[index_slew_stop]
        else:
            settle_time = (
                max(time_array[index_slew_stop + krms[-1]], time_array[index_slew_stop + kmean[-1]])
                - time_array[index_slew_stop]
            )
        settle_time = settle_time.total_seconds()
        PF = False
    log_message(f"settle_time:{settle_time}", f)
    return PF, rms_at_req, mean_at_req, settle_time


def compute_settle_time(
    df_ims,  
    reference_time="2023-06-01T06:00:0Z",  
    lo_delta_t=5,  
    hi_delta_t=30,  
    ims_column="xPosition",  
    rms_req=2e-3, 
    chi2_prob=0.999,  
    f="SITCOM_1172.log",
    verbose=False,
):
    """Function to compute settle time and PASS/FAIL for a given slew stop event

    Parameters:
    -----------
    df_ims: pandas data frame
        IMS (Independent Measurement System of the M1M3) data
    reference_time: str
        slew stop time in UTC
    lo_delta_t: int
        time window in seconds BEFORE the slew stop to retrieve data
    hi_delta_t: int
        time window in seconds AFTER the slew stop to retrieve data
    ims_column: str
        specific IMS column to analyze in this function call
    rms_req: float
        tolerance for IMS value to be within requirement. We will apply it 
        to the RMS and bias of the value, with respect to the value at 
        reference_t + post_delta_t
    chi2_prob: float
        confidence level for IMS variable wrt to long term value and variance 
        to agree
    f: str
        file name for logs
    verbose: bool
        Verbosity flag of log outputs

    Returns:
    --------
    PF: bool
        PASS/FAIL for this column and event, at 1 second after slew stop, 
        checking RMS and bias
    settle_interval: float
        the time after slew stop where the algorithm determines there is 
        stability, independently of requirement
    rms_at_req: float
        value of the RMS (jitter) of the column value, using a rolling check, 
        after the slew stop
    mean_at_req: float
        value of the bias (in absolute value) of the column value, using a 
        rolling check, after the slew stop
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

    T0 = pd.to_datetime(reference_time)  # this is slew stop
    delta_window = [
        pd.Timedelta(lo_delta_t, "seconds"),
        pd.Timedelta(hi_delta_t, "seconds"),
    ]
    # zoom around the T0 of interest
    TZoom = [T0 - delta_window[0], T0 + delta_window[1]]

    # target_variable_plot takes the data frame for the complete plot range
    target_variable_plot = df_ims[ims_column][TZoom[0] : TZoom[1]]
    # target_variable_check takes the data from the slew stop, until the end of the plot
    target_variable_check = df_ims[ims_column][T0 : TZoom[1]]
    idx_t0 = df_ims.index[  # index in dataframe closest in time to slew stop
        df_ims.index.get_indexer([pd.to_datetime(T0)], method="nearest")
    ]
    idx_tend = df_ims.index[  # index in dataframe closest in time to end of plot
        df_ims.index.get_indexer(
            [pd.to_datetime(T0 + delta_window[1])], method="nearest"
        )
    ]
    target_variable_reference = [
        float(df_ims[ims_column][idx_t0]),
        float(df_ims[ims_column][idx_tend]),
    ]
    if len(target_variable_plot.index) == 0:
        print("Data frame is empty")
        return -1

    # it is important that the end of the plot (target_variable_reference[1])
    # does not hit another slew or movement, nor at any point in the middle of the window

    # correct IMS variable wrt end of plot
    corrected_variablePlot = target_variable_plot - target_variable_reference[1]
    corrected_variable_check = target_variable_check - target_variable_reference[1]
    corrected_variable_check2 = np.square(corrected_variable_check)

    # number of values where the chi2 will be computed
    rolling = 30  # 50 is approx. 1 s
    # chi2 right tail probability for N=rolling dof at chi2_prob CL
    crit = stats.chi2.ppf(chi2_prob, rolling)

    rms = corrected_variable_check.rolling(rolling).std()
    var = corrected_variable_check.rolling(rolling).var()
    mean = abs(corrected_variable_check.rolling(rolling).mean())

    # compute the chi2 against the null hypothesis
    # the x_i are the measurements in a window (wrt to reference at end of plot)
    # the variance is for the same values
    # so chi2 = sum_N[(x_i - 0)**2/variance] where N = rolling
    sum2 = corrected_variable_check2.rolling(rolling).sum()
    chi2 = sum2 / var
    # check the chi2 at each step using rolling_check as the number of consecutive instances in which
    # chi2 has to be under the critical value
    # or rms and bias be both already 10% of requirement
    PFCheck = (chi2 < crit) | ((rms < 0.1 * rms_req) & (mean < 0.1 * rms_req))
    # PFCheck = (rms < 0.2 * rms_req) & (mean < 0.5 * rms_req)
    rolling_check = 30
    stability_check = (
        PFCheck.rolling(rolling_check).apply(lambda s: s.all()) > 0
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
        # if settle_time < reference_time:
        #    settle_time = reference_time
    settle_interval = -1
    if settle_time:
        settle_interval = settle_time - reference_time
        if settle_interval.total_seconds() < 0:
            print(f"Already settled at slew stop")
            settle_interval = 0
        else:
            settle_interval = settle_interval.total_seconds()

    PF, rms_at_req, mean_at_req, settle_intervalReq = check_requirement(
        reference_time, df_ims, ims_column, corrected_variable_check, rms_req, f, verbose
    )

    # if not settle_time:
    #    return True,-1,rms_at_req,mean_at_req

    return PF, settle_intervalReq, rms_at_req, mean_at_req


def log_message(mess, f):
    print(mess)
    try:
        print(mess, file=f)
    except:
        print("Could not write to log file")
    return


def get_slew_times(events, i_slew):
    t0 = Time(events[i_slew].begin, format="isot", scale="utc")
    t0 = pd.to_datetime(t0.value, utc=True)  # astropy Time to Timestamp conversion
    t1 = Time(events[i_slew].end, format="isot", scale="utc")
    t1 = pd.to_datetime(t1.value, utc=True)  # astropy Time to Timestamp conversion

    return (t0, t1)


def get_IMS_data(events, i_slew, postPadding):
    client = makeEfdClient()
    df_ims = getEfdData(
        client, "lsst.sal.MTM1M3.imsData", event=events[i_slew], postPadding=postPadding
    )
    return df_ims


def run_test_settling_time(dayObs, postPadding, block, outdir, f, verbose):
    """Function to run the settling time statistics test over a complete 
    block on a given night

    Parameters:
    -----------
    dayObs: int
        corresponding observation day to analyze
    postPadding: int
        number of seconds after slew stop in which to perform the analysis
    block: int
        observation block number
    outdir: str
        directory to store plots and log outputs
    f: str
        file handle for log output
    verbose: bool
        flag for output verbosity
    """

    req_rms_position = (
        2e-3  ## mm, tolerance from repeatability requirement for IMS positional
    )
    req_rms_rotation = 3e-5  ## degrees (1 arcsec is 27e-5), tolerance from repeatability requirement for IMS rotational

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
    log_message(mess, f)

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
        f"Of the {len(events)} events, {len(block_events)} relate to block {block}", f
    )

    rms_pos_at_req_agg = []
    mean_pos_at_req_agg = []
    mean_xpos_at_req_agg = []
    mean_ypos_at_req_agg = []
    mean_zpos_at_req_agg = []
    rms_rot_at_req_agg = []
    mean_rot_at_req_agg = []
    settle_time_pos_agg = []
    settle_time_rot_agg = []
    fails_agg = []
    settle_time_xpos_agg = []
    settle_time_ypos_agg = []
    settle_time_zpos_agg = []
    settle_time_xrot_agg = []
    settle_time_yrot_agg = []
    settle_time_zrot_agg = []

    ignoreList = [92, 120, 274]  # these are specific seqNums to ignore

    for i in range(len(block_events)):
        # print(TMAState.TRACKING, TMAState.SLEWING, block_events[i].endReason, block_events[i].type)
        if (
            block_events[i].endReason == TMAState.TRACKING
            and block_events[i].type == TMAState.SLEWING
        ):
            single_slew = block_events[i].seqNum
            if single_slew in ignoreList:
                log_message(f"Skipping {single_slew}", f)
                continue  # e.g. 92 is badly identified in 20231220
            log_message(f"Will look at slew {single_slew}", f)
            t0, t1 = get_slew_times(events, single_slew)
            df_ims = get_IMS_data(events, single_slew, postPadding)
            df_ims = df_ims[all_columns]
            # Convert meter to milimeter
            df_ims[pos_columns] = df_ims[pos_columns] * 1e3
            all_col_PF = True  # flag to detect whether any column has failed the test
            fails = 0
            for col in all_columns:
                if col in pos_columns:
                    req = req_rms_position
                else:
                    req = req_rms_rotation
                PF, settle_interval, rms_at_req, mean_at_req = compute_settle_time(
                    df_ims=df_ims,
                    reference_time=t1,
                    lo_delta_t=5,
                    hi_delta_t=postPadding,
                    ims_column=col,
                    rms_req=req,
                    chi2_prob=0.99,
                    f=f,
                    verbose=verbose,
                )
                if settle_interval >= 0:
                    log_message(f"{col} settled in {settle_interval:.2f} s", f)
                else:
                    log_message(f"{col} not settled in {postPadding} s", f)
                if PF == False:
                    all_col_PF = False
                    fails = fails + 1
                if col in pos_columns:
                    rms_pos_at_req_agg.append(rms_at_req)
                    mean_pos_at_req_agg.append(mean_at_req)
                    settle_time_pos_agg.append(settle_interval)
                    if col == "xPosition":
                        mean_xpos_at_req_agg.append(mean_at_req)
                        settle_time_xpos_agg.append(settle_interval)
                    if col == "yPosition":
                        mean_ypos_at_req_agg.append(mean_at_req)
                        settle_time_ypos_agg.append(settle_interval)
                    if col == "zPosition":
                        mean_zpos_at_req_agg.append(mean_at_req)
                        settle_time_zpos_agg.append(settle_interval)
                else:
                    rms_rot_at_req_agg.append(rms_at_req)
                    mean_rot_at_req_agg.append(mean_at_req)
                    settle_time_rot_agg.append(settle_interval)
                    if col == "xRotation":
                        settle_time_xrot_agg.append(settle_interval)
                    if col == "yRotation":
                        settle_time_yrot_agg.append(settle_interval)
                    if col == "zRotation":
                        settle_time_zrot_agg.append(settle_interval)
            if all_col_PF == False:
                log_message(f"Event {single_slew} has {fails} failure(s)", f)
            fails_agg.append(fails)

    title = f"Settle test for block {block} on {dayObs}"
    plt.hist(fails_agg, bins=[0, 1, 2, 3, 4, 5, 6, 7])
    plt.title(title + " Number of failures per event")
    plt.ylabel("Number of events")
    plt.xlabel("Number of failures")
    plt.legend()
    plt.savefig(outdir + "/nb_failures.png")

    plt.clf()
    plt.hist(settle_time_pos_agg, bins=50)
    plt.title(title + " settle time for position")
    plt.ylabel("Number of events (all axes)")
    plt.xlabel("Settling time (s)")
    plt.legend()
    plt.savefig(outdir + "/settletime_position.png")

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
    plt.axvline(1, lw="1.25", c="k", ls="dashed", label="Requirement")
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
    plt.axvline(1, lw="1.25", c="k", ls="dashed", label="Requirement")
    plt.title(title + " settle time (per axis)")
    plt.xlabel("Settle time (s)")
    plt.legend()
    plt.savefig(outdir + "/settletime_rotation_xyz.png")

    plt.clf()
    plt.hist(settle_time_rot_agg, bins=50)
    plt.title(title + " settle time for rotation")
    plt.ylabel("Number of events (all axes)")
    plt.xlabel("Settling time (s)")
    plt.legend()
    plt.savefig(outdir + "/settletime_rotation.png")

    plt.clf()
    plt.hist(rms_pos_at_req_agg, bins=50)
    plt.axvline(req_rms_position, lw="1.25", c="k", ls="dashed", label="Requirement")
    plt.title(title + " position RMS")
    plt.xlabel("IMS position RMS wrt settled, at 1 s after stop (mm)")
    plt.legend()
    plt.savefig(outdir + "/rms_position.png")

    plt.clf()
    plt.hist(rms_rot_at_req_agg, bins=50)
    plt.axvline(req_rms_rotation, lw="1.25", c="k", ls="dashed", label="Requirement")
    plt.title(title + " rotation RMS")
    plt.xlabel("IMS rotation RMS wrt settled, at 1 s after stop (deg)")
    plt.legend()
    plt.savefig(outdir + "/rms_rotation.png")

    plt.clf()
    plt.hist(mean_pos_at_req_agg, bins=50)
    plt.axvline(req_rms_position, lw="1.25", c="k", ls="dashed", label="Requirement")
    plt.title(title + " position bias")
    plt.xlabel("IMS position BIAS wrt settled, at 1 s after stop (mm)")
    plt.legend()
    plt.savefig(outdir + "/mean_position.png")

    plt.clf()
    plt.hist(
        mean_xpos_at_req_agg,
        bins=50,
        alpha=0.5,
        color="red",
        ls="dashed",
        label="xPosition",
    )
    plt.hist(
        mean_ypos_at_req_agg,
        bins=50,
        alpha=0.5,
        color="green",
        ls="dashed",
        label="yPosition",
    )
    plt.hist(
        mean_zpos_at_req_agg,
        bins=50,
        alpha=0.5,
        color="black",
        ls="dashed",
        label="zPosition",
    )
    plt.axvline(req_rms_position, lw="1.25", c="k", ls="dashed", label="Requirement")
    plt.title(title + " position bias (per axis)")
    plt.xlabel("IMS position BIAS wrt settled, at 1 s after stop (mm)")
    plt.legend()
    plt.savefig(outdir + "/mean_position_xyz.png")

    plt.clf()
    plt.hist(mean_rot_at_req_agg, bins=50)
    plt.axvline(req_rms_rotation, lw="1.25", c="k", ls="dashed", label="Requirement")
    plt.title(title + " rotation bias")
    plt.xlabel("IMS rotation BIAS wrt settled, at 1 s after stop (deg)")
    plt.legend()
    plt.savefig(outdir + "/mean_rotation.png")

    return 0


def main():
    """Run code with options"""
    usage = "%prog [options]"
    parser = OptionParser(usage=usage)
    parser.add_option(
        "--dayObs",
        dest="dayObs",
        help="Date for observations or measurements",
        default=20231222,
        type="int",
    )
    parser.add_option(
        "--block", dest="block", help="Set block value", default=146, type="int"
    )
    parser.add_option(
        "--padding",
        dest="postPadding",
        help="Seconds to analyze after slew stop",
        default=15,
    )
    parser.add_option(
        "--outdir",
        dest="outdir",
        help="Output directory for results",
        default="./SITCOM-1172_out",
    )
    (options, args) = parser.parse_args()

    if not os.path.exists(options.outdir):
        os.makedirs(options.outdir)

    f = open(options.outdir + "/SITCOM_1172.log", "a")

    c = datetime.now()
    timeStamp = c.strftime("%H:%M:%S")
    log_message(f"Running run_test_settling_time at {timeStamp}", f)

    result = run_test_settling_time(
        options.dayObs,
        options.postPadding,
        options.block,
        options.outdir,
        f,
        verbose=True,
    )

    log_message(f"Test result {result}. Check outputs in {options.outdir}", f)


if __name__ == "__main__":
    main()
