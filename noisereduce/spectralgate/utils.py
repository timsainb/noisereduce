import numpy as np


def sigmoid(x, shift, mult):
    """
    Using this sigmoid to discourage one network overpowering the other
    """
    return 1 / (1 + np.exp(-(x + shift) * mult))


def _amp_to_db(x, top_db=80.0, eps=np.finfo(np.float64).eps):
    """
    Convert the input tensor from amplitude to decibel scale.
    """
    x_db = 20 * np.log10(np.abs(x) + eps)
    return np.maximum(x_db, np.max(x_db, axis=-1, keepdims=True) - top_db)
