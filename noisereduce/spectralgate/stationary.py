from noisereduce.spectralgate.base import SpectralGate
import numpy as np
from librosa import stft, istft
from scipy.signal import fftconvolve
from .utils import _amp_to_db


class SpectralGateStationary(SpectralGate):
    def __init__(
        self,
        y,
        sr,
        y_noise,
        n_std_thresh_stationary,
        chunk_size,
        clip_noise_stationary,
        padding,
        n_fft,
        win_length,
        hop_length,
        time_constant_s,
        freq_mask_smooth_hz,
        time_mask_smooth_ms,
        tmp_folder,
        prop_decrease,
        use_tqdm,
        n_jobs,
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

        self.n_std_thresh_stationary = n_std_thresh_stationary

        if y_noise is None:
            self.y_noise = self.y

        else:
            y_noise = np.array(y_noise)
            # reshape data to (#channels, #frames)
            if len(y_noise.shape) == 1:
                self.y_noise = np.expand_dims(y_noise, 0)
            elif len(y.shape) > 2:
                raise ValueError("Waveform must be in shape (# frames, # channels)")
            else:
                self.y_noise = y_noise

        # collapse y_noise to one channel
        self.y_noise = np.mean(self.y_noise, axis=0)

        if clip_noise_stationary:
            self.y_noise = self.y_noise[:chunk_size]

        # calculate statistics over y_noise
        abs_noise_stft = np.abs(
            stft(
                (self.y_noise),
                n_fft=self._n_fft,
                hop_length=self._hop_length,
                win_length=self._win_length,
            )
        )
        noise_stft_db = _amp_to_db(abs_noise_stft)
        self.mean_freq_noise = np.mean(noise_stft_db, axis=1)
        self.std_freq_noise = np.std(noise_stft_db, axis=1)

        self.noise_thresh = (
            self.mean_freq_noise + self.std_freq_noise * self.n_std_thresh_stationary
        )

    def spectral_gating_stationary(self, chunk):
        """non-stationary version of spectral gating"""
        denoised_channels = np.zeros(chunk.shape, chunk.dtype)
        for ci, channel in enumerate(chunk):
            sig_stft = stft(
                (channel),
                n_fft=self._n_fft,
                hop_length=self._hop_length,
                win_length=self._win_length,
            )

            # spectrogram of signal in dB
            sig_stft_db = _amp_to_db(np.abs(sig_stft))

            # calculate the threshold for each frequency/time bin
            db_thresh = np.repeat(
                np.reshape(self.noise_thresh, [1, len(self.mean_freq_noise)]),
                np.shape(sig_stft_db)[1],
                axis=0,
            ).T

            # mask if the signal is above the threshold
            sig_mask = sig_stft_db > db_thresh

            sig_mask = sig_mask * self._prop_decrease + np.ones(np.shape(sig_mask)) * (
                1.0 - self._prop_decrease
            )

            if self.smooth_mask:
                # convolve the mask with a smoothing filter
                sig_mask = fftconvolve(sig_mask, self._smoothing_filter, mode="same")

            # multiply signal with mask
            sig_stft_denoised = sig_stft * sig_mask

            # invert/recover the signal
            denoised_signal = istft(
                sig_stft_denoised,
                hop_length=self._hop_length,
                win_length=self._win_length,
            )
            denoised_channels[ci, : len(denoised_signal)] = denoised_signal
        return denoised_channels

    def _do_filter(self, chunk):
        """Do the actual filtering"""
        chunk_filtered = self.spectral_gating_stationary(chunk)

        return chunk_filtered
