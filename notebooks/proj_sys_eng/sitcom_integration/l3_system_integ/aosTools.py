import os
import asyncio
import numpy as np
import time

import pandas as pd
from matplotlib import pyplot as plt

from M1M3_FATABLE import *

from lsst.ts import salobj
from lsst.ts.idl.enums import MTM1M3
from lsst.ts.idl.enums import MTHexapod

m3ORC = 2.508
m3IRC = 0.550
m1ORC = 4.18
m1IRC = 2.558

fat = np.array(FATABLE)
m1m3_actID = np.int16(fat[:, FATABLE_ID])
m1m3_nActuator = m1m3_actID.shape[0]
m1m3_xact = np.float64(fat[:, FATABLE_XPOSITION])
m1m3_yact = np.float64(fat[:, FATABLE_YPOSITION])

aa = np.loadtxt("%s/notebooks/M2_FEA/data/M2_1um_72_force.txt" % (os.environ["HOME"]))
# to have +x going to right, and +y going up, we need to transpose and reverse x and y
m2_xact = -aa[:, 2]
m2_yact = -aa[:, 1]


class MyLogHandler:
    def __init__(self, nmsg):
        self.nmsg = nmsg
        self.nprint = 0

    def printLogMessage(self, data):
        if self.nprint < self.nmsg:
            print(f"{data.level}: {data.message}")
        self.nprint += 1


async def checkAOSSummaryStates(aos, m1m3, m2, camhex, m2hex):
    # aos
    sstate = await aos.evt_summaryState.aget(timeout=5)
    print(
        "starting with: MTAOS state",
        salobj.State(sstate.summaryState),
        pd.to_datetime(sstate.private_sndStamp, unit="s"),
    )

    await checkAOSCompStates(m1m3, m2, camhex, m2hex)


async def checkSlewCompStates(ptg, mount, rot):
    sstate = await ptg.evt_summaryState.aget(timeout=5)
    print(
        "staring with: ptg state",
        salobj.State(sstate.summaryState),
        pd.to_datetime(sstate.private_sndStamp, unit="s"),
    )

    sstate = await mount.evt_summaryState.aget(timeout=5)
    print(
        "staring with: mount state",
        salobj.State(sstate.summaryState),
        pd.to_datetime(sstate.private_sndStamp, unit="s"),
    )

    sstate = await rot.evt_summaryState.aget(timeout=5)
    print(
        "staring with: rot state",
        salobj.State(sstate.summaryState),
        pd.to_datetime(sstate.private_sndStamp, unit="s"),
    )


async def plotM1M3Forces(m1m3):
    fel = await m1m3.evt_appliedElevationForces.aget(timeout=10.0)
    fba = await m1m3.evt_appliedBalanceForces.aget(timeout=10.0)
    fst = await m1m3.evt_appliedStaticForces.aget(timeout=10.0)
    fao = await m1m3.evt_appliedActiveOpticForces.aget(timeout=10.0)

    ftel = await m1m3.tel_forceActuatorData.aget(timeout=10.0)

    fig, ax = plt.subplots(3, 1, figsize=(15, 8))
    ax[0].plot(fel.xForces, "-o", label="elevation")
    ax[0].plot(fba.xForces, label="FB")
    ax[0].plot(fst.xForces, label="static")
    ax[0].plot(ftel.xForce, "-v", label="measured")
    ax[0].legend()
    ax[0].set_title("XForces")
    ax[1].plot(fel.yForces, "-o", label="elevation")
    # ax[1].plot(fba.yForces, label='FB')
    # ax[1].plot(fst.yForces, label='static')
    ax[1].plot(ftel.yForce, "-v", label="measured")
    ax[1].legend()
    ax[1].set_title("YForces")
    ax[2].plot(fel.zForces, "-o", label="elevation")
    ax[2].plot(fba.zForces, label="FB")
    ax[2].plot(fst.zForces, label="static")
    ax[2].plot(fao.zForces, label="AOS")
    ax[2].plot(ftel.zForce, "-v", label="measured")
    ax[2].set_title("ZForces")
    ax[2].legend()

    fig2, ax = plt.subplots(1, 3, figsize=[15, 4])
    aa = np.array(fao.zForces)
    img = ax[0].scatter(m1m3_xact, m1m3_yact, c=aa, s=abs(aa) * 2)
    # plt.jet()
    ax[0].axis("equal")
    ax[0].set_title("AOS forces")
    fig.colorbar(img, ax=ax[0])

    aa = np.array(fel.zForces)
    img = ax[1].scatter(m1m3_xact, m1m3_yact, c=aa, s=abs(aa) * 0.1)
    # plt.jet()
    ax[1].axis("equal")
    ax[1].set_title("elevation forces")
    fig.colorbar(img, ax=ax[1])

    aa = np.array(fst.zForces)
    img = ax[2].scatter(m1m3_xact, m1m3_yact, c=aa, s=abs(aa) * 10)
    # plt.jet()
    ax[2].axis("equal")
    ax[2].set_title("static forces")
    fig.colorbar(img, ax=ax[2])


async def plotM2Forces(m2):
    axialForces = await m2.tel_axialForce.aget(timeout=2)
    tangentForces = await m2.tel_tangentForce.aget(timeout=2)

    fig, ax = plt.subplots(2, 1, figsize=(15, 8))
    ax[0].plot(axialForces.measured, label="measured")
    ax[0].plot(axialForces.applied, label="applied")
    ax[0].plot(axialForces.hardpointCorrection, ".", label="FB")
    ax[0].plot(axialForces.lutGravity, label="LUT G")
    ax[0].legend()
    ax[1].plot(tangentForces.measured, label="measured")
    ax[1].plot(tangentForces.applied, label="applied")
    ax[1].plot(tangentForces.hardpointCorrection, "o", label="FB")
    ax[1].plot(tangentForces.lutGravity, label="LUT G")
    ax[1].legend()

    fig2, ax = plt.subplots(1, 2, figsize=[10, 4])
    aa = np.array(axialForces.measured)
    img = ax[0].scatter(m2_xact, m2_yact, c=aa, s=abs(aa) * 2)
    # plt.jet()
    ax[0].axis("equal")
    ax[0].set_title("measured forces")
    fig.colorbar(img, ax=ax[0])

    aa = np.array(axialForces.applied)
    img = ax[1].scatter(m2_xact, m2_yact, c=aa, s=abs(aa) * 2)
    # plt.jet()
    ax[1].axis("equal")
    ax[1].set_title("applied forces")
    fig.colorbar(img, ax=ax[1])


async def moveHexaTo0(hexa):
    ### command it to collimated position (based on LUT)

    need_to_move = False
    try:
        posU = await hexa.evt_uncompensatedPosition.aget(timeout=10.0)
        if abs(max([getattr(posU, i) for i in "xyzuvw"])) < 1e-8:
            print("hexapod already at LUT position")
        else:
            need_to_move = True
    except TimeoutError:
        need_to_move = True
    if need_to_move:
        hexa.evt_inPosition.flush()
        # according to XML, units are micron and degree
        await hexa.cmd_move.set_start(x=0, y=0, z=0, u=0, v=0, w=0, sync=True)
        while True:
            state = await hexa.evt_inPosition.next(flush=False, timeout=10)
            print(
                "hexa in position?",
                state.inPosition,
                pd.to_datetime(state.private_sndStamp, unit="s"),
            )
            if state.inPosition:
                break
        await printHexaPosition(hexa)


async def printHexaPosition(hexa):
    pos = await hexa.tel_application.next(flush=True, timeout=10.0)
    print("Current Hexapod position")
    print(" ".join(f"{p:10.2f}" for p in pos.position[:3]), end=" ")
    print(" ".join(f"{p:10.6f}" for p in pos.position[3:]))


async def printHexaUncompensatedAndCompensated(hexa):
    posU = await hexa.evt_uncompensatedPosition.aget(timeout=10.0)
    print("Uncompensated position")
    print(" ".join(f"{p:10.2f}" for p in [getattr(posU, i) for i in "xyz"]), end="    ")
    print(
        " ".join(f"{p:10.6f}" for p in [getattr(posU, i) for i in "uvw"]),
        "  ",
        pd.to_datetime(posU.private_sndStamp, unit="s"),
    )
    posC = await hexa.evt_compensatedPosition.aget(timeout=10.0)
    print("Compensated position")
    print(
        " ".join(f"{p:10.2f}" for p in [getattr(posC, i) for i in "xyz"]), end="     "
    )
    print(
        " ".join(f"{p:10.6f}" for p in [getattr(posC, i) for i in "uvw"]),
        "  ",
        pd.to_datetime(posC.private_sndStamp, unit="s"),
    )


async def readyHexaForAOS(hexa):
    settings = await hexa.evt_settingsApplied.aget(timeout=10.0)
    hasSettings = 0
    if hasattr(settings, "settingsVersion"):
        print(
            "settingsVersion = ",
            settings.settingsVersion,
            pd.to_datetime(settings.private_sndStamp, unit="s"),
        )
        hasSettings = 1
    if (not hasSettings) or (not settings.settingsVersion[:12] == "default.yaml"):
        print("YOU NEED TO SEND THIS HEXAPOD TO STANDBY, THEN LOAD THE PROPER CONFIG")
    else:
        hexaConfig = await hexa.evt_configuration.aget(timeout=10.0)
        print(
            "pivot at (%.0f, %.0f, %.0f) microns "
            % (hexaConfig.pivotX, hexaConfig.pivotY, hexaConfig.pivotZ)
        )
        print(
            "maxXY = ", hexaConfig.maxXY, "microns, maxZ= ", hexaConfig.maxZ, " microns"
        )
        print("maxUV = ", hexaConfig.maxUV, "deg, maxW= ", hexaConfig.maxW, " deg")

        lutMode = await hexa.evt_compensationMode.aget(timeout=10)
        if not lutMode.enabled:
            hexa.evt_compensationMode.flush()
            await hexa.cmd_setCompensationMode.set_start(enable=1, timeout=10)
            lutMode = await hexa.evt_compensationMode.next(flush=False, timeout=10)
        print(
            "compsensation mode enabled?",
            lutMode.enabled,
            pd.to_datetime(lutMode.private_sndStamp, unit="s"),
        )
        await moveHexaTo0(hexa)
        await printHexaUncompensatedAndCompensated(hexa)
        print(
            "Does the hexapod has enough inputs to do LUT compensation? (If the below times out, we do not.)"
        )
        # Note: the target events are what the hexa CSC checks; if one is missing, the entire LUT will not be applied
        # it also needs to see an uncompensatedPosition (a move would trigger that) in order to move to the compensatedPosition
        a = await hexa.evt_compensationOffset.aget(timeout=10.0)
        print("mount elevation = ", a.elevation)
        print("mount azimth = ", a.azimuth)
        print("rotator angle = ", a.rotation)
        print("? temperature = ", a.temperature)
        print("x,y,z,u,v,w = ", a.x, a.y, a.z, a.u, a.v, a.w)


async def ofcSentApplied(aos, m1m3, m2, camhex, m2hex, make_plot=False):
    dof = await aos.evt_degreeOfFreedom.aget(timeout=5.0)
    m1m3C = await aos.evt_m1m3Correction.aget(timeout=5.0)
    m2C = await aos.evt_m2Correction.aget(timeout=5.0)
    camhexC = await aos.evt_cameraHexapodCorrection.aget(timeout=5.0)
    m2hexC = await aos.evt_m2HexapodCorrection.aget(timeout=5.0)

    aggregated_dof = np.array(dof.aggregatedDoF)
    visit_dof = np.array(dof.visitDoF)
    m1m3C = m1m3C.zForces
    m2C = m2C.zForces
    camhexC = np.array([getattr(camhexC, i) for i in ["x", "y", "z", "u", "v", "w"]])
    m2hexC = np.array([getattr(m2hexC, i) for i in ["x", "y", "z", "u", "v", "w"]])

    print("DOF event time = ", pd.to_datetime(dof.private_sndStamp, unit="s"))

    m1m3F = await m1m3.evt_appliedActiveOpticForces.aget(timeout=5.0)
    m2F = await m2.tel_axialForce.next(flush=True, timeout=5.0)
    camhexP = await camhex.evt_uncompensatedPosition.aget(timeout=5.0)
    m2hexP = await m2hex.evt_uncompensatedPosition.aget(timeout=5.0)

    m1m3F = m1m3F.zForces
    m2F = m2F.applied
    camhexP = np.array([getattr(camhexP, i) for i in ["x", "y", "z", "u", "v", "w"]])
    m2hexP = np.array([getattr(m2hexP, i) for i in ["x", "y", "z", "u", "v", "w"]])

    if make_plot:
        fig, ax = plt.subplots(2, 3, figsize=(19, 8))
        ##--------------------------------------
        ax[0][0].plot(aggregated_dof[:10], "-bo", label="aggregatedDoF")
        ax[0][0].plot(visit_dof[:10], "-rx", label="visitDoF")
        ax[0][0].set_title("hexapod DoF")
        ax[0][0].legend()

        ax[0][1].plot(aggregated_dof[10:], "-bo", label="aggregatedDoF")
        ax[0][1].plot(visit_dof[10:], "-rx", label="visitDoF")
        ax[0][1].set_title("Mirrors DoF")
        ax[0][1].legend()

        ##--------------------------------------
        ax[0][2].plot(m1m3C, "-o", label="forces sent")
        ax[0][2].plot(m1m3F, "-rx", label="forces applied")
        ax[0][2].set_title("M1M3 Forces")
        ax[0][2].legend()

        ax[1][0].plot(m2C, "-o", label="forces sent")
        ax[1][0].plot(m2F, "-x", label="forces applied")
        ax[1][0].set_title("M2 Forces")
        ax[1][0].legend()

        ##--------------------------------------
        ax[1][1].plot(camhexC[:3], "-ro", label="cam hex xyz Sent", markersize=8)
        ax[1][1].plot(m2hexC[:3], "-bx", label="m2 hex xyz Sent")
        ax[1][1].plot(camhexP[:3], "-o", label="cam hex xyz Applied")
        ax[1][1].plot(m2hexP[:3], "-v", label="m2 hex xyz Applied")
        ax[1][1].set_title("Hex xyz")
        ax[1][1].legend()

        ax[1][2].plot(camhexC[3:], "-ro", label="cam hex uvw Sent")
        ax[1][2].plot(m2hexC[3:], "-bx", label="m2 hex uvw Sent")
        ax[1][2].plot(camhexP[3:], "-o", label="cam hex uvw Applied")
        ax[1][2].plot(m2hexP[3:], "-v", label="m2 hex uvw Applied")
        ax[1][2].set_title("M2 Hex xyzuvw")
        ax[1][2].legend()

    ofc_dict = {}
    ofc_dict["aggregated_dof"] = aggregated_dof
    ofc_dict["visit_dof"] = visit_dof
    ofc_dict["m1m3C"] = m1m3C
    ofc_dict["m2C"] = m2C
    ofc_dict["camhexC"] = camhexC
    ofc_dict["m2hexC"] = m2hexC
    ofc_dict["m1m3F"] = m1m3F
    ofc_dict["m2F"] = m2F
    ofc_dict["camhexP"] = camhexP
    ofc_dict["m2hexP"] = m2hexP

    print(
        "If corrections have been issued, we should always expect sent (xxC) to match applied (xxF & xxP)"
    )
    return ofc_dict


async def moveMountConstantV(mount, startAngle, stopAngle):
    # change the elevation angle step by step

    freq = 1  # Hz
    vAngle = 2  # 1 deg change per minute
    holdMinutes = 0.1  # how long to hold at integeter values of the elevation angle
    angleStepSize = 1  # each time we change by 1 deg, before we hold in place

    rampMinutes = angleStepSize / vAngle
    print(
        "This will run for %.0f minutes"
        % ((startAngle - stopAngle) * (rampMinutes + holdMinutes))
    )
    start_time = Time(datetime.now())
    startTime = time.time()
    end_time = start_time + timedelta(minutes=80)
    demandAngle = startAngle
    while demandAngle > stopAngle - 0.01:
        await asyncio.sleep(1.0 / freq)

        timeNow = time.time()
        minutesEllapsed = (timeNow - startTime) / 60
        cyclePassed = np.floor(minutesEllapsed / (rampMinutes + holdMinutes))
        minutesIntoThisCycle = min(
            rampMinutes, minutesEllapsed - cyclePassed * (rampMinutes + holdMinutes)
        )
        demandAngle = (
            startAngle - cyclePassed * angleStepSize - minutesIntoThisCycle * vAngle
        )
        # print(demandAngle, cyclePassed, minutesIntoThisCycle)
        await mount.cmd_moveToTarget.set_start(azimuth=0, elevation=demandAngle)
