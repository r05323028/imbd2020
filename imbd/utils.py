import logging


def get_logger():
    logger = logging.getLogger(name='imbd2020')
    stream_handler = logging.StreamHandler()
    fmt = logging.Formatter(
        fmt='%(asctime)s | %(name)s | %(levelname)s | %(message)s')
    stream_handler.setFormatter(fmt)
    logger.addHandler(stream_handler)
    logger.setLevel(logging.INFO)
    return logger