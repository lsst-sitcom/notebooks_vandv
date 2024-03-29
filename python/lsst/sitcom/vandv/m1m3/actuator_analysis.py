import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from astropy.time import Time, TimeDelta
from scipy.optimize import minimize

from lsst.summit.utils.efdUtils import getEfdData
from lsst.summit.utils.tmaUtils import TMAEvent
from lsst.ts.xml.enums.MTM1M3 import BumpTest
from lsst.ts.xml.tables.m1m3 import force_actuator_from_id


BUMP_TEST_DURATION = 14.0  # seconds


def plot_actuator_delay(
    fig: plt.Figure,
    client: object,
    fa_id: int,
    event: TMAEvent = None,    
    bt_results: pd.DataFrame = None,
    bt_index: int = 0,
) -> (float, float):
    """
    Plot the actuator delay.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The matplotlib figure object to plot the results on.
    client : object
        The EFD client object to retrieve data from.
    bt_results : pandas.DataFrame
        The bump test results data. Used if input is a bump test.
        Default is None
    event : lsst.summit.utils.tmaUtils.TMAEvent
         A TMAEvent. Used if input is a slew event.
         default is None
    fa_id : int
        The ID of the force actuator.
    bt_index : int, optional
        The index of the bump test. Default is 0.

    Returns
    -------
    primary_delay : float
        The delay of the primary actuator.
    secondary_delay : float
        The delay of the secondary actuator (if applicable).
    """
    # Grab the Force Actuator Data from its ID
    fa_data = force_actuator_from_id(fa_id)
    if bt_results is not None and event is not None:
        raise ValueError("You can't specify both a bump test and an event.")
    if bt_results is not None:
        # This branch is followed if the input is a bump test
        # Extract bump test results for a given force actuator
        bt_result = bt_results[bt_results["actuatorId"] == fa_id]
        primary_bump = f"primaryTest{fa_data.index}"
        plot_name = "Bump Test"
        t_start = Time(
            bt_result[bt_result[primary_bump] == BumpTest.TESTINGPOSITIVE][
                "timestamp"
            ].values[bt_index]
            - 1.0,
            format="unix_tai",
            scale="tai",
        )

        t_end = Time(
            t_start + TimeDelta(BUMP_TEST_DURATION, format="sec"),
            format="unix_tai",
            scale="tai",
        )
    if event is not None:
        # This branch is followed if the input is a TMAEvent
        plot_name = f"Event {event.dayObs} - {event.seqNum}"
        t_start = event.begin
        t_end = event.end

    # Plot preparation
    primary_force = f"zForce{fa_data.z_index}"
    primary_applied = f"zForces{fa_data.z_index}"

    measured_forces = getEfdData(
        client,
        "lsst.sal.MTM1M3.forceActuatorData",
        columns=[primary_force, "timestamp"],
        begin=t_start,
        end=t_end,
    )

    applied_forces = getEfdData(
        client,
        "lsst.sal.MTM1M3.appliedForces",
        columns=[primary_applied, "timestamp"],
        begin=t_start,
        end=t_end,
    )

    t0 = measured_forces["timestamp"].values[0]
    measured_forces["timestamp"] -= t0
    applied_forces["timestamp"] -= t0

    # It is easier/faster to work with arrays
    measured_forces_time = measured_forces["timestamp"].values
    measured_forces_values = measured_forces[primary_force].values
    applied_forces_time = applied_forces["timestamp"].values
    applied_forces_values = applied_forces[primary_applied].values

    # Find best shift
    primary_delay, best_shift_forces = get_best_shift(
        applied_forces_time,
        applied_forces_values,
        measured_forces_time,
        measured_forces_values,
    )

    # Start plots
    timestamp = measured_forces.index[0].isoformat().split(".")[0]
    fig.subplots_adjust(wspace=0.3)
    fig.suptitle(
        f"{plot_name} Response Delay. Actuator ID {fa_id}\n {timestamp}", fontsize=18
    )

    ax1 = fig.add_subplot(2, 2, 1)
    ax2 = fig.add_subplot(2, 2, 3, sharex=ax1)
    ax3 = fig.add_subplot(2, 2, 2)
    ax4 = fig.add_subplot(2, 2, 4, sharex=ax3)

    plot_primary_forces(
        ax1,
        applied_forces_time,
        applied_forces_values,
        measured_forces_time,
        measured_forces_values,
    )

    plot_shifted_primary_forces(
        ax2,
        applied_forces_time,
        applied_forces_values,
        primary_delay,
        best_shift_forces,
    )

    if fa_data.actuator_type.name == "DAA":
        if bt_results is not None:
            # This branch is followed if the input is a bump test
            # Extract bump test results for a given force actuator
            bt_result = bt_results[bt_results["actuatorId"] == fa_id]
            secondary_bump = f"secondaryTest{fa_data.s_index}"

            t_start = Time(
                bt_result[bt_result[secondary_bump] == 2]["timestamp"].values[bt_index]
                - 1.0,
                format="unix_tai",
                scale="tai",
            )

            t_end = Time(
                t_start + TimeDelta(BUMP_TEST_DURATION, format="sec"),
                format="unix_tai",
                scale="tai",
            )

        if event is not None:
            # This branch is followed if the input is a TMAEvent
            t_start = event.begin
            t_end = event.end

        secondary_name = fa_data.orientation.name

        if secondary_name in ["X_PLUS", "X_MINUS"]:
            secondary_force = f"xForce{fa_data.x_index}"
            secondary_applied = f"xForces{fa_data.x_index}"
        elif secondary_name in ["Y_PLUS", "Y_MINUS"]:
            secondary_force = f"yForce{fa_data.y_index}"
            secondary_applied = f"yForces{fa_data.y_index}"
        else:
            raise ValueError(f"Unknown secondary name {secondary_name}")


        secondary_measured_forces = getEfdData(
            client,
            "lsst.sal.MTM1M3.forceActuatorData",
            columns=[secondary_force, "timestamp"],
            begin=t_start,
            end=t_end,
        )

        secondary_applied_forces = getEfdData(
            client,
            "lsst.sal.MTM1M3.appliedForces",
            columns=[secondary_applied, "timestamp"],
            begin=t_start,
            end=t_end,
        )

        t0 = secondary_measured_forces["timestamp"].values[0]
        sec_app_times = secondary_applied_forces["timestamp"].values - t0
        sec_meas_times = secondary_measured_forces["timestamp"].values - t0
        sec_app_forces = secondary_applied_forces[secondary_applied].values
        sec_meas_forces = secondary_measured_forces[secondary_force].values * np.sqrt(2.0)

        # Find best shift - the sqrt(2) you see below is to account for the
        # fact that the secondary forces are measured in the diagonal direction
        # as described in Section 6 of https://sitcomtn-083.lsst.io/
        secondary_delay, sec_best_shift_forces = get_best_shift(
            sec_app_times,
            sec_app_forces,
            sec_meas_times,
            sec_meas_forces / np.sqrt(2.0),
        )

        plot_secondary_forces(
            ax3,
            sec_app_times,
            sec_app_forces,
            sec_meas_times,
            sec_meas_forces,
            secondary_name,
        )

        plot_actuator_delay_secondary(
            ax4,
            sec_app_times,
            sec_app_forces,
            sec_best_shift_forces,
            secondary_name,
            secondary_delay,
        )

    else:
        secondary_delay = None
        ax3.set_title("Secondary - None")
        ax4.set_title("Secondary - None")

    return primary_delay, secondary_delay


def match_function(params: list, args: list) -> float:
    """
    Determines best shift to match up applied and measured forces

    Parameters
    ----------
    params : list
        List of parameters for the shift
    args : list
        List of arguments containing the measured and applied forces

    Returns
    -------
    float: Sum of squared differences between the applied and shifted forces
    """
    [measured_t, measured_f, applied_t, applied_f] = args
    shifted_f = np.interp(applied_t + params[0], measured_t, measured_f)
    diff = applied_f - shifted_f
    return np.sum(diff * diff)


def get_best_shift(
    atime: np.array,
    aforces: np.array,
    mtime: np.array,
    mforces: np.array,
) -> (float, np.array):
    """
    Calculates the best shift value and shifted primary forces for a given set
    of input arrays.

    Parameters
    ----------
    atime : np.array
        Array of timestamps for the actual forces.
    aforces : np.array
        Array of actual forces.
    mtime : np.array
        Array of timestamps for the measured forces.
    mforces : np.array
        Array of measured forces.

    Returns
    -------
    delay : float
        The delay in milliseconds.
    shifted_forces : np.array
        The shifted primary forces.
    """
    param_0 = [0.10]
    args_0 = [mtime, mforces, atime, aforces]
    bounds = [(-0.5, 0.5)] # Limit shift to +/- 500ms
    best_shift = minimize(match_function, param_0, bounds=bounds, args=args_0, method="Powell")
    delay = best_shift.x[0] * 1000.0
    shifted_forces = np.interp(atime + best_shift.x[0], mtime, mforces)

    return delay, shifted_forces


def plot_primary_forces(
    ax: plt.Axes,
    app_times: np.array,
    app_forces: np.array,
    meas_times: np.array,
    meas_forces: np.array,
):
    """
    Plot the primary forces.

    Parameters
    ----------
    ax : plt.Axes
        The matplotlib axes object to plot on.
    app_times : np.array
        An array of applied force times.
    app_forces : np.array
        An array of applied forces.
    meas_times : np.array
        An array of measured force times.
    meas_forces : np.array
        An array of measured forces.
    """
    ax.set_title("Primary - Z")
    ax.plot(app_times, app_forces, label="Applied")
    ax.plot(meas_times, meas_forces, label="Measured")
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Force (N)")
    ax.grid(":", alpha=0.25)
    ax.legend()


def plot_shifted_primary_forces(
    ax: plt.Axes,
    app_times: np.array,
    app_forces: np.array,
    force_delay: float,
    shift_forces: np.array,
) -> None:
    """
    Plot the shifted primary forces.

    Parameters
    ----------
    ax : plt.Axes
        The matplotlib axes object to plot on.
    app_times : np.array
        The array of applied times.
    app_forces : np.array
        The array of applied forces.
    force_delay : float
        The delay in milliseconds.
    shift_forces : np.array
        The array of shifted measured forces.
    """
    ax.plot(app_times, app_forces, label="Applied")
    ax.plot(app_times, shift_forces, label="Shifted Measured")
    shift_y_coord = np.min(shift_forces) + 0.2 * (np.max(shift_forces) - np.min(shift_forces))
    ax.text(1, shift_y_coord, f"Delay = {force_delay:.1f} ms")
    ax.set_title("Primary - Z")
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Force (N)")
    ax.grid(":", alpha=0.25)
    ax.legend()


def plot_secondary_forces(
    ax: plt.Axes,
    sec_app_times: np.array,
    sec_app_forces: np.array,
    sec_meas_times: np.array,
    sec_meas_forces: np.array,
    secondary_name: str,
) -> None:
    """
    Plot the applied and measured forces for the secondary mirror.

    Parameters
    ----------
    ax : plt.Axes
        The matplotlib Axes object to plot on.
    sec_app_times : np.array
        The time values for the applied forces.
    sec_app_forces : np.array
        The applied forces for the secondary mirror.
    sec_meas_times : np.array
        The time values for the measured forces.
    sec_meas_forces : np.array
        The measured forces for the secondary mirror.
    secondary_name : np.array
        The name of the secondary mirror.
    """
    ax.plot(sec_app_times, sec_app_forces, label="Applied")
    ax.plot(sec_meas_times, sec_meas_forces / np.sqrt(2.0), label="Measured")

    ax.set_title(f"Secondary - {secondary_name}")
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Force (N)")
    ax.legend()
    ax.grid(":", alpha=0.2)


def plot_actuator_delay_secondary(
    ax: plt.Axes,
    sec_app_times: np.array,
    sec_app_forces: np.array,
    sec_sft_forces: np.array,
    secondary_name: str,
    secondary_delay: float,
) -> None:
    """
    Plots the bump test results for the secondary actuator delay.

    Parameters
    __________
    ax : matplotlib.axes.Axes
        The axes object to plot on.
    sec_app_times : array-like
        The time values for the applied forces.
    sec_app_forces : array-like
        The applied forces.
    sec_sft_forces : array-like
        The shifted measured forces.
    secondary_name : str
        The name of the secondary actuator.
    secondary_delay : float
        The delay of the secondary actuator in milliseconds.
    """
    ax.plot(sec_app_times, sec_app_forces, label="Applied")
    ax.plot(
        sec_app_times,
        sec_sft_forces,
        label="Shifted Measured",
    )
    shift_y_coord = np.min(sec_sft_forces) + 0.2 * (np.max(sec_sft_forces) - np.min(sec_sft_forces))
    ax.text(1, shift_y_coord, f"Delay = {secondary_delay:.1f} ms")

    ax.set_title(f"Secondary - {secondary_name}")
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Force (N)")
    ax.legend()
    ax.grid(":", alpha=0.2)


def plot_delay_histograms(fig, timestamp, primary_delays, secondary_delays):
    """
    Plot the histograms of the actuator delays.

    Parameters
    ----------
    fig : plt.Figure
        The matplotlib figure object to plot the results on.
    timestamp : str
        The timestamp of the data.
    primary_delays : numpy.array
        The primary actuator delays.
    secondary_delays : numpy.array
        The secondary actuator delays.
    """
    ax1, ax2 = fig.subplots(1, 2)
    
    ax1.set_title(
        f"Primary_delays {timestamp}\n"
        f"Mean = {np.mean(primary_delays):.1f} ms"
        )
    ax1.hist(primary_delays, bins = 20)
    ax1.set_xlim(50,150)
    ax1.set_xlabel("Delay (ms)")
    ax1.grid(":", alpha=0.25)
    
    ax2.set_title(
        f"Secondary_delays {timestamp}\n"
        f"Mean = {np.mean(secondary_delays):.1f} ms"
        )
    ax2.hist(secondary_delays, bins = 20)
    ax2.set_xlim(50,150)
    ax2.set_xlabel("Delay (ms)")
    ax2.grid(":", alpha=0.25)
    
