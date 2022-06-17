import asyncio
import os

import pandas as pd

from lsst.ts import utils
from lsst.rsp import get_node


__all__ = [
    "ExecutionInfo",
]


class ExecutionInfo:
    """
    A placeholder for the information about runtime of the notebooks.

    Attributes
    ----------
    user : string
        Who is executing this notebook? It actually grabs the GitHub user.
    date :
        When is this notebook being executed?
    loc : string
        Where are you running this notebook? It might say "summit" if running at the summit,
        or "tucson" is running on the Tucson Test-Stand.
    """

    def __init__(self):
        # Extract your name from the Jupyter Hub
        self.user = os.environ["JUPYTERHUB_USER"]

        # Extract execution date
        self.date = utils.astropy_time_from_tai_unix(utils.current_tai())
        self.date.format = "isot"

        # This is used later to define where Butler stores the images or
        # to define which EFD client to use
        self.loc = os.environ["LSST_DDS_PARTITION_PREFIX"]
        self.node = get_node()

        # Create folder for plots
        os.makedirs("./plots", exist_ok=True)

    def __str__(self):
        return (
            f"\nExecuted by {self.user} on {self.date}."
            f"\n  Running in {self.node} at {self.loc}\n"
        )

    @staticmethod
    def get_index(test_case, test_execution):
        """Returns an integer obtained from the four last digits of the test
        case concatenated with the last four digits of the test execution
        so it can be used as an index for the `salobj.Controller("Script")`,
        normally used to send custom messages to the EFD.

        Parameters
        ----------
        test_case : str
            ID of the test case being run.
        test_execution : str
            ID of the test execution being run.

        Returns
        -------
        int : index to be used in a SAL Script.
        """
        return int(test_case[-4:] + test_execution[-4:])
