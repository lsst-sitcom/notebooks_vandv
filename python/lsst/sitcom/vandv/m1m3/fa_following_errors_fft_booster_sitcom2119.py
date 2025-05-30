import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
from astropy.time import Time, TimeDelta

from lsst.summit.utils.tmaUtils import TMAEventMaker
from lsst.summit.utils.efdUtils import EfdClient, getEfdData, makeEfdClient
from lsst.sitcom.vandv import m1m3
from lsst.ts.xml.tables.m1m3 import FATable

def fft_plot(fa, client, topic, column, t_start, t_end, actuator_id, plot_directory):

    FA_error = f"{column}{fa}"

    df = getEfdData(
        client,
        topic,
        columns=FA_error,
        begin=t_start,
        end=t_end,
    )

    dt = (df[FA_error].index[1] - df[FA_error].index[0]).total_seconds()
    freqs = np.fft.fftfreq(len(df), d=dt)
    positive_mask = freqs > 0
    fft_frequency = freqs[positive_mask]
    fft_result = np.fft.fft(df.values, axis=0)
    fft_magnitudes = np.abs(fft_result[positive_mask, :])
    plt.figure(figsize=(10, 5))
    if "primary" in column:
        label = f"Primary actuator {actuator_id}"
    else:
        label = f"Secondary actuator {actuator_id}"
    plt.plot(
        fft_frequency, fft_magnitudes, label=label
    )
    plt.title(f"Power spectrum for {t_start} - {t_end}")
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Magnitude")
    plt.legend()
    if "primary" in column:
        plt.savefig(f"{plot_directory}PA_{actuator_id}.png")
    else:
        plt.savefig(f"{plot_directory}SA_{actuator_id}.png")
    plt.close()

def loop_over_actuators(client, column, nb_actuators, t_start, t_end, plot_directory):

    print(f"Processing {column}")

    if not os.path.exists(plot_directory):
        os.makedirs(plot_directory)

    if "secondary" in column:
        secondary_actuator_id = np.array([])
        for i in range(len(FATable)):
            if FATable[i].s_index is not None:
                secondary_actuator_id = np.append(
                    secondary_actuator_id, FATable[i].actuator_id
                )
        secondary_actuator_id = secondary_actuator_id.astype(int)

    for fa in range(nb_actuators):
        if fa % 10 == 0:
            print(f"{fa}/{nb_actuators}")
        if "primary" in column: 
            fft_plot(fa,client,"lsst.sal.MTM1M3.forceActuatorData",column,t_start,t_end,FATable[fa].actuator_id,plot_directory)
        else:
            fft_plot(fa,client,"lsst.sal.MTM1M3.forceActuatorData",column,t_start,t_end,secondary_actuator_id[fa],plot_directory)

def search_booster_valve_events(client, t_start, t_end):

    df = getEfdData(
            client,
            "lsst.sal.MTM1M3.logevent_boosterValveStatus",
            columns="opened",
            begin=t_start,
            end=t_end,
    )

    nb_activations = len(df[df["opened"]==True])
    print(f"Found {nb_activations} activations of booster valves")

    count = 1
    for i,idx in enumerate(df["opened"]):
        if i < len(df["opened"]) - 1:
            if (df["opened"].iloc[i]==True) and (df["opened"].iloc[i+1]==False):
                print(f"Processing activation {count}/{nb_activations}")
                t_0 = Time(df["opened"].index[i])
                t_1 = Time(df["opened"].index[i+1])
                loop_over_actuators(
                    client, 
                    "primaryCylinderFollowingError", 
                    len(FATable), 
                    t_0, 
                    t_1, 
                    f"./plots/plots_booster_valve_{t_0.strftime('%Y_%m_%d_%H_%M_%S')}/"
                )
                loop_over_actuators(
                    client, 
                    "secondaryCylinderFollowingError", 
                    112, 
                    t_0, 
                    t_1, 
                    f"./plots/plots_booster_valve_{t_0.strftime('%Y_%m_%d_%H_%M_%S')}/"
                )
                count = count + 1
def main():
    client = makeEfdClient()
    parser = argparse.ArgumentParser(
        description="M1M3 force actuator booster valve activation FFT analysis"
    )
    parser.add_argument(
        "t_start",
        type=Time,
        default="2025-05-27T18:30:15",
        help="Start time in a valid format: 'YYYY-MM-DD HH:MM:SSZ'",
    )
    parser.add_argument(
        "t_end",
        type=Time,
        default="2025-05-27T18:30:24",
        help="End time in a valid format: 'YYYY-MM-DD HH:MM:SSZ'",
    )
    args = parser.parse_args()

    search_booster_valve_events(client,  args.t_start, args.t_end)


if __name__ == "__main__":
    main()
