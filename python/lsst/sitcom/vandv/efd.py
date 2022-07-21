import os

from astropy import units as u
from astropy.time import Time
from datetime import timedelta

import warnings

try:
    from lsst_efd_client import EfdClient

    efd = True
except ModuleNotFoundError:
    warnings.warn(
        "Package not found: lsst_efd_client - EFD related operations will not be available"
    )
    efd = False


__all__ = [
    "create_efd_client" "query_last_from_efd",
]


def create_efd_client():
    """Create an EFD client for different test-stand locations"""
    if not efd:
        warnings.warn("EFD Client not available")
        return None

    location = os.environ["LSST_DDS_PARTITION_PREFIX"]

    if location == "summit":
        client = EfdClient("summit_efd")
    elif location == "tucson":
        client = EfdClient("tucson_teststand_efd")
    else:
        raise ValueError("Location does not match any valid options {summit|tucson}")

    return client


async def query_last_n(
    client,
    topic_name,
    fields,
    num=1,
    index=None,
    lower_t=None,
    upper_t=None,
    debug=False,
):
    """Query the last `num` data from the EFD.

    This function will probably be removed in the near future when
    its functionality is implemented in the `EfdClient`.

    Parameters
    ----------
    topic : str
        Name of topic to query.
    fields : str or list
        Name of field(s) to query.
    num : int, optional
        Number of rows to return (default: 1).
    index : `int`, optional
        When index is used, add an 'AND {CSCName}ID = index' to the query.
        (default is `None`).
    upper_t : `astropy.time.Time`, optional
        Upper time cut in UTC. If not provided, it defaults to `Time.now()`.
    lower_t : `astropy.time.Time`, optional
        Lower time cut in UTC. If not provided, it defaults to `upper_time_cut - 15m`.
    debug : bool, optional
        Prints the query used in the EFD for debugging purposes.
    """
    if not efd:
        warnings.warn("EFD Client not available")
        return None

    if isinstance(fields, list):
        fields = fields.join(",")

    delta_t = timedelta(minutes=15)

    if upper_t is None and lower_t is None:
        upper_t = Time.now()
        lower_t = upper_t - delta_t
    elif upper_t is None and lower_t is not None:
        assert isinstance(
            lower_t, Time
        ), "`lower_t` is expected to be an astropy.time.Time instance"
        upper_t = lower_t + delta_t
    elif upper_t is not None and lower_t is None:
        assert isinstance(
            upper_t, Time
        ), "`upper_t` is expected to be an astropy.time.Time instance"
        lower_t = upper_t - delta_t

    if upper_t < lower_t:
        warnings.warn(
            "lower_t is greater than upper_t. Inverting values so we"
            " can get a valid query interval."
        )
        temp_t = upper_t
        upper_t = lower_t
        lower_t = temp_t

    query = client.build_time_range_query(topic_name, fields, lower_t, upper_t)
    query = f"{query} ORDER BY DESC LIMIT {num}"

    df = await client.influx_client.query(query)

    return df


async def query_script_message_contains(
    client, contains, lower_t=None, upper_t=None, num=1
):
    """Query the `lsst.sal.Script.logevent_logMessage` while applying a filter
    criteria.

    Parameters
    ----------
    topic : str
        Name of topic to query.
    contains : str or list
        String(s) present within the message.
    num : int, optional
        Number of rows to return (default: 1).
    upper_t : `astropy.time.Time`, optional
        Upper time cut in UTC. If not provided, it defaults to `Time.now()`.
    lower_t : `astropy.time.Time`, optional
        Lower time cut in UTC. If not provided, it defaults to `upper_time_cut - 15m`.
    debug : bool, optional
        Prints the query used in the EFD for debugging purposes.
    """
    from datetime import timedelta

    topic_name = "lsst.sal.Script.logevent_logMessage"
    fields = "message"

    if isinstance(contains, list):
        for i, c in enumerate(contains):
            contains[i] = f'"message" =~ /{c}/'
        contains = " AND ".join(contains)
    else:
        contains = f'"message" =~ /{contains}/'

    delta_t = timedelta(minutes=15)

    if upper_t is None and lower_t is None:
        upper_t = Time.now()
        lower_t = upper_t - delta_t
    elif upper_t is None and lower_t is not None:
        assert isinstance(
            lower_t, Time
        ), "`lower_t` is expected to be an astropy.time.Time instance"
        upper_t = lower_t + delta_t
    elif upper_t is not None and lower_t is None:
        assert isinstance(
            upper_t, Time
        ), "`upper_t` is expected to be an astropy.time.Time instance"
        lower_t = upper_t - delta_t

    if upper_t < lower_t:
        warnings.warn(
            "lower_t is greater than upper_t. Inverting values so we"
            " can get a valid query interval."
        )
        temp_t = upper_t
        upper_t = lower_t
        lower_t = temp_t

    query = client.build_time_range_query(topic_name, fields, lower_t, upper_t)
    query = f"{query} AND ({contains})"
    query = f"{query} ORDER BY DESC LIMIT {num}"

    df = await client.influx_client.query(query)

    return df


def time_pd_to_astropy(timestamp):
    """Converts a pandas timestamp to astropy Time"""
    return Time(timestamp.isoformat().split("+")[0], scale="utc", format="isot")
