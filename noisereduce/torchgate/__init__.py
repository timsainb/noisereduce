"""
TorchGating is a PyTorch-based implementation of Spectral Gating
================================================
Author: Asaf Zorea

Contents
--------
torchgate imports all the functions from PyTorch, and in addition provides:
 TorchGating       --- A PyTorch module that applies a spectral gate to an input signal

"""

from .run_with_noisereduce import run_tg_with_noisereduce
from .torchgate import TorchGating
