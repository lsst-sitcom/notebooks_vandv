""" 
Tools used to run multiple tests involving the Main Telescope Active Optics 
System. 

These tools were originally written by Bo Xin and they were extracted from:

https://github.com/lsst-ts/ts_notebooks/blob/develop/bxin/aos2comp/aosTools.py
"""

__all__ = [
    "checkAOSCompStates"
]

import pandas as pd

from lsst.ts import salobj
from lsst.ts.idl.enums import MTM1M3


async def checkAOSCompStates(m1m3, m2, camhex, m2hex):
    """
    Prints out the state and sub-state of the components commanded by the 
    MTAOS.
    
    Parameters
    ----------
    m1m3 : lsst.ts.salobj.remote.Remote
        Remote for `mtm1m3`
    m2 : lsst.ts.salobj.remote.Remote
        Remote for `mtm2`
    camhex : lsst.ts.salobj.remote.Remote
        Remote for `mthexapod_1`
    m2hex : lsst.ts.salobj.remote.Remote
        Remote for `mthexapod_2`
    """
    # M1M3
    sstate = await m1m3.evt_summaryState.aget(timeout=5)   
    dstate = await m1m3.evt_detailedState.aget(timeout=200)
    print(
        f"mtm1m3 state: {salobj.State(sstate.summaryState).name}"
        f" {pd.to_datetime(sstate.private_sndStamp, unit='s')}\n"
        f"       detailed state: {MTM1M3.DetailedState(dstate.detailedState).name}"
        f" {pd.to_datetime(dstate.private_sndStamp, unit='s')}\n"
    )
    
    # M2
    sstate = await m2.evt_summaryState.aget(timeout=5)
    print(
        f"mtm2 state: {salobj.State(sstate.summaryState).name}"
        f" {pd.to_datetime(sstate.private_sndStamp, unit='s')}\n"
    )

    # CamHex
    sstate = await camhex.evt_summaryState.aget(timeout=5)
    print(
        f"mthexapod_1 state: {salobj.State(sstate.summaryState).name}"
        f" {pd.to_datetime(sstate.private_sndStamp, unit='s')}\n"
    )
       
    # M2hex
    state = await m2hex.evt_summaryState.aget(timeout=5)
    print(
        f"mthexapod_2 state: {salobj.State(sstate.summaryState).name}"
        f" {pd.to_datetime(sstate.private_sndStamp, unit='s')}\n"
    )
    
    
