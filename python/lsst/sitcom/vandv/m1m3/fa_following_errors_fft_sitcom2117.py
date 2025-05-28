import matplotlib.pyplot as plt
import numpy as np
import os
from astropy.time import Time, TimeDelta

from lsst.summit.utils.tmaUtils import TMAEventMaker
from lsst.summit.utils.efdUtils import EfdClient, getEfdData, makeEfdClient
from lsst.sitcom.vandv import m1m3
from lsst.ts.xml.tables.m1m3 import FATable

def loop_over_actuators(topic, nb_actuators, t_start, t_end):
    print(f"Processing {topic}")

    plot_directory = "./plots/"
    if not os.path.exists(plot_directory):
        os.makedirs(plot_directory)
    FA_error = [f"{topic}{i}" for i in range(nb_actuators)]

    if "secondary" in topic: 
        secondary_actuator_id = np.array([])
        for i in range(len(FATable)):
            if FATable[i].s_index is not None:
                secondary_actuator_id = np.append(secondary_actuator_id,FATable[i].actuator_id)
        secondary_actuator_id = secondary_actuator_id.astype(int)
    for fa in range(nb_actuators):
        if fa%10==0:
            print(f"{fa}/{nb_actuators}")
        df = getEfdData(
            client,"lsst.sal.MTM1M3.forceActuatorData", 
            columns=FA_error, 
            begin=t_start, 
            end=t_end,
        )
        dt = (df[f"{topic}0"].index[1] - df[f"{topic}0"].index[0]).total_seconds()
        freqs = np.fft.fftfreq(len(df), d=dt)
        positive_mask = freqs > 0
        fft_frequency = freqs[positive_mask]
        fft_result = np.fft.fft(df.values, axis=0)
        fft_magnitudes = np.abs(fft_result[positive_mask, :])
        plt.figure(figsize=(10, 5))
        if "primary" in topic:
            label = FATable[fa].actuator_id
        else:
            label = secondary_actuator_id[fa]
        plt.plot(fft_frequency, fft_magnitudes[:,fa], label=f"Primary actuator {label}")
        plt.title(f"Power spectrum for {t_start} - {t_end}")
        plt.xlabel("Frequency [Hz]")
        plt.ylabel("Magnitude")
        plt.legend()
        if "primary" in topic:
            plt.savefig(f"{plot_directory}PA_{FATable[fa].actuator_id}.png")
        else:
            plt.savefig(f"{plot_directory}SA_{secondary_actuator_id[fa]}.png")
 
        plt.close() 

client = makeEfdClient()

t_start = Time("2025-05-27 18:30:15",scale="utc")
t_end = Time("2025-05-27 18:30:24",scale="utc")

loop_over_actuators("primaryCylinderFollowingError", len(FATable), t_start, t_end)
loop_over_actuators("secondaryCylinderFollowingError", 112, t_start, t_end)




