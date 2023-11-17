import logging

__all__ = [
    "add_filter_to_mtcs",
    "create_logger",
    "filter_dds_read_queue_is_full",
    "filter_dds_read_queue_is_filling",
]


def add_filter_to_mtcs() -> None:
    """
    Adds two filters to the main telescope CSCSs to avoid the multiple messages 
    related to the DDS.

    This function iterates over all the loggers in the `logging` module's root
    manager's logger dictionary, selecting those whose names start with "MT" or
    "CC" (main telescope and ComCam loggers).
    It then applies two predefined filters to each of these loggers to suppress
    certain DDS-related messages.

    Filters applied are:
    - `filter_dds_read_queue_is_filling`:
      To filter out messages indicating the DDS read queue is filling.
    - `filter_dds_read_queue_is_full`:
      To filter out messages indicating the DDS read queue is full.

    Parameters
    ----------
    None

    Returns
    -------
    None

    Notes
    -----
    This function assumes that the filters `filter_dds_read_queue_is_filling` 
    and `filter_dds_read_queue_is_full` are defined elsewhere in the code and 
    are accessible in the scope where this function is called.

    The function modifies the logger instances in-place by adding filters to
    them and does not return any value.
    """
    loggers = [
        logging.getLogger(name)
        for name in logging.root.manager.loggerDict
        if name.startswith("MT") or name.startswith("CC")
    ]

    for logger in loggers:
        logger.addFilter(filter_dds_read_queue_is_filling)
        logger.addFilter(filter_dds_read_queue_is_full)



def create_logger(name: str) -> logging.Logger:
    """
    Create a logger object with the specified name and returns it.
    Parameters
    ----------
    name : str
        The name of the logger object.
    Returns
    -------
    logger : logging.Logger
        The logger object with the specified name.
    """
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    formatter.datefmt = "%Y-%m-%d %H:%M:%S"

    handler = logging.StreamHandler()
    handler.setLevel(logging.ERROR)
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    return logger


def filter_dds_read_queue_is_filling(record: logging.LogRecord) -> bool:
    """Determines if log record message indicates DDS read queue is filling.

    Parameters
    ----------
    record : logging.LogRecord
        Log record to be filtered.

    Returns
    -------
    bool
        True if message contains 'DDS read queue is filling', else False.
    """
    return "DDS read queue is filling" in record.msg


def filter_dds_read_queue_is_full(record: logging.LogRecord) -> bool:
    """Checks if the log record message states that the DDS read queue is full.

    Parameters
    ----------
    record : logging.LogRecord
        The log record to be checked.

    Returns
    -------
    bool
        True if 'DDS read queue is full' is in the message; False otherwise.
    """
    return "DDS read queue is full" in record.msg
