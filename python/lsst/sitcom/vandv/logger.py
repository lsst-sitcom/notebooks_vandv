__all__ = [
    "filter_dds_read_queue_is_full",
    "filter_dds_read_queue_is_filling",
]


def filter_dds_read_queue_is_filling(record):
    return ("DDS read queue is filling" in record.msg)


def filter_dds_read_queue_is_full(record):
    return ("DDS read queue is full" in record.msg)

