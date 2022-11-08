import logging

__all__ = [
    "add_filter_to_mtcs",
    "filter_dds_read_queue_is_full",
    "filter_dds_read_queue_is_filling",
]


def add_filter_to_mtcs():
    """Adds two filters to the main telescope CSCSs to avoid the
    multiple messages related to the DDS"""
    loggers = [
        logging.getLogger(name)
        for name in logging.root.manager.loggerDict
        if name.startswith("MT") or name.startswith("CC")
    ]

    for logger in loggers:
        logger.addFilter(filter_dds_read_queue_is_filling)
        logger.addFilter(filter_dds_read_queue_is_full)


def filter_dds_read_queue_is_filling(record):
    return "DDS read queue is filling" in record.msg


def filter_dds_read_queue_is_full(record):
    return "DDS read queue is full" in record.msg
