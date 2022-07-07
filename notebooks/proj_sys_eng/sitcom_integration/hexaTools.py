import os
import asyncio
import numpy as np
import time

import pandas as pd
from matplotlib import pyplot as plt

from lsst.ts import salobj
from lsst.ts.idl.enums import MTHexapod



async def readingTemperatures(hexa):
    dir(temp)
    [getattr(temp,'temperatureC%02d'%i) for i in range(1,8+1)]
    end = Time(datetime.now(), scale='tai')
    start = end - timedelta(seconds=1000)
    df = await client.select_time_series('lsst.sal.ESS.temperature8Ch', '*', start, end, csc_index)
    fig, ax = plt.subplots(1,1, figsize=(15,4))
    for i in range(1,8+1):
        plt.plot(getattr(df, 'temperatureC%02d'%i))
    plt.grid()
    
async def prepareEnvironmentforHexa(hexa):
    #Start the telemetry 
    #mount telemetry:
    mount_angle = await mount.tel_elevation.next(flush=False, timeout=10.)
    print("mount elevation angle", mount_angle.actualPosition)
    elev = mount_angle.actualPosition    
    
async def printHexaPosition(hexa):
    pos = await hexa.tel_application.next(flush=True, timeout=10.)
    print("Current Hexapod position")
    print(" ".join(f"{p:10.2f}" for p in pos.position[:3]), end = ' ') 
    print(" ".join(f"{p:10.6f}" for p in pos.position[3:]) )
    
async def printHexaUncompensatedAndCompensated(hexa):
    posU = await hexa.evt_uncompensatedPosition.aget(timeout=10.)
    print('Uncompensated position')
    print(" ".join(f"{p:10.2f}" for p in [getattr(posU, i) for i in 'xyz']), end = '    ')
    print(" ".join(f"{p:10.6f}" for p in [getattr(posU, i) for i in 'uvw']),'  ',
         pd.to_datetime(posU.private_sndStamp, unit='s'))    
    posC = await hexa.evt_compensatedPosition.aget(timeout=10.)
    print('Compensated position = (uncompensated + LUT)')
    print(" ".join(f"{p:10.2f}" for p in [getattr(posC, i) for i in 'xyz']), end = '     ')
    print(" ".join(f"{p:10.6f}" for p in [getattr(posC, i) for i in 'uvw']),'  ',
         pd.to_datetime(posC.private_sndStamp, unit='s'))
    

    
async def printHexaUncompensatedAndCompensated(hexa):
    posU = await hexa.evt_uncompensatedPosition.aget(timeout=10.)
    print('Uncompensated position')
    print(" ".join(f"{p:10.2f}" for p in [getattr(posU, i) for i in 'xyz']), end = '    ')
    print(" ".join(f"{p:10.6f}" for p in [getattr(posU, i) for i in 'uvw']),'  ',
         pd.to_datetime(posU.private_sndStamp, unit='s'))    
    posC = await hexa.evt_compensatedPosition.aget(timeout=10.)
    print('Compensated position = (uncompensated + LUT)')
    print(" ".join(f"{p:10.2f}" for p in [getattr(posC, i) for i in 'xyz']), end = '     ')
    print(" ".join(f"{p:10.6f}" for p in [getattr(posC, i) for i in 'uvw']),'  ',
         pd.to_datetime(posC.private_sndStamp, unit='s'))
    
async def moveHexaTo0(hexa, actual_z = 0):
    ### command it to collimated position (based on LUT)
    
    need_to_move = False
    try:
        posU = await hexa.evt_uncompensatedPosition.aget(timeout=10.)
        if abs(max([getattr(posU, i) for i in 'xyzuvw']))<1e-8:
            print('hexapod already at LUT position')
        else:
            need_to_move = True
    except asyncio.exceptions.TimeoutError:
        need_to_move = True
    if need_to_move:
        hexa.evt_inPosition.flush()
        #according to XML, units are micron and degree
        await hexa.cmd_move.set_start(x=0,y=0,z=actual_z, u=0,v=0,w=0,sync=True)
        while True:
            state = await hexa.evt_inPosition.next(flush=False, timeout=10)
            print("hexa in position?",state.inPosition, pd.to_datetime(state.private_sndStamp, unit='s'))
            if state.inPosition:
                break
        await printHexaPosition(hexa)
    
async def readyHexaForAOS(hexa):
    settings = await hexa.evt_settingsApplied.aget(timeout = 10.)
    hasSettings = 0
    if hasattr(settings, 'settingsVersion'):
        print('settingsVersion = ', settings.settingsVersion, pd.to_datetime(settings.private_sndStamp, unit='s'))
        hasSettings = 1
    if (not hasSettings) or (not settings.settingsVersion[:12] == 'default.yaml'):
        print('YOU NEED TO SEND THIS HEXAPOD TO STANDBY, THEN LOAD THE PROPER CONFIG')
    else:
        hexaConfig = await hexa.evt_configuration.aget(timeout=10.)
        print("pivot at (%.0f, %.0f, %.0f) microns "%(hexaConfig.pivotX, hexaConfig.pivotY, hexaConfig.pivotZ))
        print("maxXY = ", hexaConfig.maxXY, "microns, maxZ= ", hexaConfig.maxZ, " microns")
        print("maxUV = ", hexaConfig.maxUV, "deg, maxW= ", hexaConfig.maxW, " deg")

        lutMode = await hexa.evt_compensationMode.aget(timeout=10)
        if not lutMode.enabled:
            hexa.evt_compensationMode.flush()
            await hexa.cmd_setCompensationMode.set_start(enable=1, timeout=10)
            lutMode = await hexa.evt_compensationMode.next(flush=False, timeout=10)
        print("compsensation mode enabled?",lutMode.enabled, pd.to_datetime(lutMode.private_sndStamp, unit='s'))
        await moveHexaTo0(hexa, actual_z = 100)
        await moveHexaTo0(hexa)
        await printHexaUncompensatedAndCompensated(hexa)
        print("Does the hexapod has enough inputs to do LUT compensation? (If the below times out, we do not.)")
        #Note: the target events are what the hexa CSC checks; if one is missing, the entire LUT will not be applied
        #it also needs to see an uncompensatedPosition (a move would trigger that) in order to move to the compensatedPosition
        a = await hexa.evt_compensationOffset.aget(timeout=10.)
        print('mount elevation = ', a.elevation)
        print('mount azimth = ', a.azimuth)
        print('rotator angle = ', a.rotation)
        print('? temperature = ', a.temperature)
        print('x,y,z,u,v,w = ', a.x, a.y, a.z, a.u, a.v, a.w, pd.to_datetime(a.private_sndStamp, unit='s'))