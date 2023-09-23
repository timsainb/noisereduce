import torch
from noisereduce.spectralgate.base import SpectralGate
from noisereduce.torchgate import TorchGate as TG
import numpy as np


class StreamedTorchGate(SpectralGate):
    '''
    Run interface with noisereduce.
    '''

    def __init__(
            self,
            y,
            sr,
            stationary=False,
            y_noise=None,
            prop_decrease=1.0,
            time_constant_s=2.0,
            freq_mask_smooth_hz=500,
            time_mask_smooth_ms=50,
            thresh_n_mult_nonstationary=2,
            sigmoid_slope_nonstationary=10,
            n_std_thresh_stationary=1.5,
            tmp_folder=None,
            chunk_size=600000,
            padding=30000,
            n_fft=1024,
            win_length=None,
            hop_length=None,
            clip_noise_stationary=True,
            use_tqdm=False,
            n_jobs=1,
            device="cuda",
    ):
        super().__init__(
            y=y,
            sr=sr,
            chunk_size=chunk_size,
            padding=padding,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            time_constant_s=time_constant_s,
            freq_mask_smooth_hz=freq_mask_smooth_hz,
            time_mask_smooth_ms=time_mask_smooth_ms,
            tmp_folder=tmp_folder,
            prop_decrease=prop_decrease,
            use_tqdm=use_tqdm,
            n_jobs=n_jobs,
        )

        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        # noise convert to torch if needed
        if y_noise is not None:
            if y_noise.shape[-1] > y.shape[-1] and clip_noise_stationary:
                y_noise = y_noise[: y.shape[-1]]
            y_noise = torch.from_numpy(y_noise).to(device)
            # ensure that y_noise is in shape (#channels, #frames)
            if len(y_noise.shape) == 1:
                y_noise = y_noise.unsqueeze(0)
        self.y_noise = y_noise

        # create a torch object
        self.tg = TG(
            sr=sr,
            nonstationary=not stationary,
            n_std_thresh_stationary=n_std_thresh_stationary,
            n_thresh_nonstationary=thresh_n_mult_nonstationary,
            temp_coeff_nonstationary=1 / sigmoid_slope_nonstationary,
            n_movemean_nonstationary=int(time_constant_s / self._hop_length * sr),
            prop_decrease=prop_decrease,
            n_fft=self._n_fft,
            win_length=self._win_length,
            hop_length=self._hop_length,
            freq_mask_smooth_hz=freq_mask_smooth_hz,
            time_mask_smooth_ms=time_mask_smooth_ms,
        ).to(device)

    def _do_filter(self, chunk):
        """Do the actual filtering"""
        # convert to torch if needed
        if type(chunk) is np.ndarray:
            chunk = torch.from_numpy(chunk).to(self.device)
        chunk_filtered = self.tg(x=chunk, xn=self.y_noise)
        return chunk_filtered.cpu().detach().numpy()
