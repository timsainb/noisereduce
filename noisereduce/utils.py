import numpy as np


def int16_to_float32(data):
    """ Converts from uint16 wav to float32 wav
    """
    if np.max(np.abs(data)) > 32768:
        raise ValueError("Data has values above 32768")
    return (data / 32768.0).astype("float32")


def float32_to_int16(data):
    if np.max(data) > 1:
        data = data / np.max(np.abs(data))
    return np.array(data * 32767).astype("int16")
