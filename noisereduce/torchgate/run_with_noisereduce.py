import torch
import numpy as np


def run_tg_with_noisereduce(y,
                            sr,
                            stationary,
                            device,
                            y_noise,
                            prop_decrease,
                            n_std_thresh_stationary,
                            n_fft,
                            win_length,
                            hop_length,
                            time_constant_s,
                            freq_mask_smooth_hz,
                            time_mask_smooth_ms,
                            thresh_n_mult_nonstationary,
                            sigmoid_slope_nonstationary,
                            clip_noise_stationary
                            ):
    '''
    Run interface with noisereduce so that torch is not a necessary requirement for noisereduce.
    '''
    from . import TorchGate as TG

    device = torch.device(device) if torch.cuda.is_available() else torch.device(device)

    y_type = type(y)
    assert y_type is np.ndarray or torch.Tensor
    assert type(y_noise) is y_type or y_noise is None

    if y_type is np.ndarray:
        y = torch.from_numpy(y).to(device)
    y_ndim = y.ndim
    if y_ndim == 1:
        y = y.unsqueeze(0)

    if y_noise is not None:
        if y_noise.shape[-1] > y.shape[-1] and clip_noise_stationary:
            y_noise = y_noise[:y.shape[-1]]
        y_noise = torch.from_numpy(y_noise).to(device)

    if hop_length is None:
        hop_length = n_fft // 4

    tg = TG(sr=sr,
            nonstationary=not stationary,
            n_std_thresh_stationary=n_std_thresh_stationary,
            n_thresh_nonstationary=thresh_n_mult_nonstationary,
            temp_coeff_nonstationary=1 / sigmoid_slope_nonstationary,
            n_movemean_nonstationary=int(time_constant_s / hop_length * sr),
            prop_decrease=prop_decrease,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            freq_mask_smooth_hz=freq_mask_smooth_hz,
            time_mask_smooth_ms=time_mask_smooth_ms
            ).to(device)

    y_denoised = tg(y, y_noise)
    if y_ndim == 1:
        y_denoised = y_denoised.squeeze(0)

    if y_type is np.ndarray:
        return y_denoised.cpu().numpy()
    else:
        return y_denoised
