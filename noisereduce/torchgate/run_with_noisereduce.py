from . import TorchGating as TG
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
    device = torch.device(device) if torch.cuda.is_available() else torch.device(device)

    y_type = y.dtype
    assert y_type.dtype is np.ndarray or torch.Tensor
    assert y_noise.dtype is y_type or y_type is None

    if y_type is np.ndarray:
        y = torch.from_numpy(y).to(device)

    if y_noise.dtype is np.ndarray and y_noise is not None:
        if y_noise.shape[-1] > y.shape[-1] and clip_noise_stationary:
            y_noise = y_noise[:y.shape[-1]]
        y_noise = torch.from_numpy(y_noise).to(device)

    tg = TG(sr=sr,
            nonstationary=not stationary,
            n_std_thresh_stationary=n_std_thresh_stationary,
            n_thresh_nonstationary=thresh_n_mult_nonstationary,
            temp_coeff_nonstationary=1 / sigmoid_slope_nonstationary,
            n_movemean_nonstationary=int(time_constant_s * sr),
            prop_decrease=prop_decrease,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            freq_mask_smooth_hz=freq_mask_smooth_hz,
            time_mask_smooth_ms=time_mask_smooth_ms
            ).to(device)

    y_denoised = tg(y, y_noise)

    if y_type is np.ndarray:
        return y_denoised.cpu().numpy()
    else:
        return y_denoised
