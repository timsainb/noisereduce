import numpy as np
from librosa.core import amplitude_to_db, db_to_amplitude


def sigmoid(x, shift, mult):
    """
    Using this sigmoid to discourage one network overpowering the other
    """
    return 1 / (1 + np.exp(-(x + shift) * mult))


def _amp_to_db(x):
    """
    Convert the input tensor from amplitude to decibel scale.
    """
    return amplitude_to_db(x, ref=1.0, amin=1e-20, top_db=80.0)


def _db_to_amp(x, ):
    """
    Convert the input tensor from decibel scale to amplitude.
    """
    return db_to_amplitude(x, ref=1.0)
