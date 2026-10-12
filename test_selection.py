import numpy as np
import pytest

from noisereduce.spectralgate.stationary import SpectralGateStationary


@pytest.mark.parametrize("channels", [1, 2])
@pytest.mark.parametrize("chunk_size", [None, 2048, 4096])
@pytest.mark.parametrize("start,end", [(0, 3072), (512, 2560), (512, 3584)])
def test_get_traces_selects_requested_interval(
    channels, chunk_size, start, end
):
    rng = np.random.default_rng(7)
    y = rng.normal(size=(channels, 8192))
    if channels == 1:
        y = y[0]
    gate = SpectralGateStationary(
        y=y, sr=16000, y_noise=None, n_std_thresh_stationary=1.5,
        chunk_size=chunk_size, clip_noise_stationary=True, padding=1024,
        n_fft=512, win_length=None, hop_length=None, time_constant_s=2,
        freq_mask_smooth_hz=None, time_mask_smooth_ms=None,
        tmp_folder=None, prop_decrease=0, use_tqdm=False, n_jobs=1,
    )

    result = gate.get_traces(start_frame=start, end_frame=end)

    assert result.shape == y[..., start:end].shape
    np.testing.assert_allclose(result, y[..., start:end], atol=1e-12)
