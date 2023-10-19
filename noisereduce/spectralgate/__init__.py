from .nonstationary import SpectralGateNonStationary
from .stationary import SpectralGateStationary
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
if TORCH_AVAILABLE:
    from .streamed_torch_gate import StreamedTorchGate
