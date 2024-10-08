import numpy as np
from joblib import Parallel, delayed
import tempfile
from tqdm.auto import tqdm
from scipy.signal import fftconvolve, stft, istft
from .utils import _amp_to_db
from scipy.ndimage import uniform_filter1d
from warnings import warn


def _smoothing_filter(n_grad_freq, n_grad_time):
    """Generates a filter to smooth the mask for the spectrogram

    Arguments:
        n_grad_freq {[type]} -- [how many frequency channels to smooth over with the mask.]
        n_grad_time {[type]} -- [how many time channels to smooth over with the mask.]
    """
    smoothing_filter = np.outer(
        np.concatenate(
            [
                np.linspace(0, 1, n_grad_freq + 1, endpoint=False),
                np.linspace(1, 0, n_grad_freq + 2),
            ]
        )[1:-1],
        np.concatenate(
            [
                np.linspace(0, 1, n_grad_time + 1, endpoint=False),
                np.linspace(1, 0, n_grad_time + 2),
            ]
        )[1:-1],
    )
    smoothing_filter = smoothing_filter / np.sum(smoothing_filter)
    return smoothing_filter


class SpectralGate:
    def __init__(
        self,
        y,
        sr,
        stationary,
        y_noise,
        prop_decrease,
        noise_window_size_nonstationary_ms,
        n_std_thresh,
        chunk_size,
        clip_noise_stationary,
        padding,
        n_fft,
        win_length,
        hop_length,
        freq_mask_smooth_hz,
        time_mask_smooth_ms,
        tmp_folder,
        use_tqdm,
        n_jobs,
    ):
        self.sr = sr
        # if this is a 1D single channel recording
        self.flat = False
        self.stationary = stationary
        self._chunk_size = chunk_size
        self.padding = padding
        self.n_jobs = n_jobs
        self.n_std_thresh = n_std_thresh
        self.use_tqdm = use_tqdm

        ### Parameters for spectral gating
        self._prop_decrease = prop_decrease
        self.noise_window_size_nonstationary_ms = noise_window_size_nonstationary_ms

        # convert noise_window_size_nonstationary_ms to samples
        if self.stationary == False:
            if self.noise_window_size_nonstationary_ms is None:
                # warn the user that a noise_window_size_nonstationary_ms must be provided, and default to 5s
                warn(
                    "noise_window_size_nonstationary_ms must be provided for non-stationary noise, defaulting to 5s"
                )
                noise_window_size_nonstationary_ms = int(5 * self.sr)
            self.noise_window_size_nonstationary = int(
                noise_window_size_nonstationary_ms * self.sr / 1000
            )
            if self.noise_window_size_nonstationary > self._chunk_size:
                # warn that the window size is smaller than the chunk size, and set it to the chunk size*2
                warn(
                    "noise_window_size_nonstationary must be smaller than the chunk size, defaulting to noise_window_size_nonstationary x 2"
                )
                self._chunk_size = int(self.noise_window_size_nonstationary * 2)
            # if padding is less than half the window size, set padding to half the window size
            if self.padding < self.noise_window_size_nonstationary / 2:
                warn(
                    "padding must be at least half the noise_window_size_nonstationary, defaulting to half the window size"
                )
                self.padding = int(self.noise_window_size_nonstationary / 2) + 1

        # prepare signal clip
        y = np.array(y)
        # reshape data to (#channels, #frames)
        if len(y.shape) == 1:
            self.y = np.expand_dims(y, 0)
            self.flat = True
        elif len(y.shape) > 2:
            raise ValueError("Waveform must be in shape (# frames, # channels)")
        else:
            self.y = y
        self._dtype = y.dtype

        # get the number of channels and frames in data
        self.n_channels, self.n_frames = self.y.shape

        # where to create a temp file for parallel writing
        self._tmp_folder = tmp_folder

        # set window and hop length for stft
        self._n_fft = n_fft
        if win_length is None:
            self._win_length = self._n_fft
        else:
            self._win_length = win_length
        if hop_length is None:
            self._hop_length = self._win_length // 4
        else:
            self._hop_length = hop_length

        # prepare the smoothing mask
        if (freq_mask_smooth_hz is None) & (time_mask_smooth_ms is None):
            self.smooth_mask = False
        else:
            self._generate_mask_smoothing_filter(
                freq_mask_smooth_hz, time_mask_smooth_ms
            )

        # prepare noise clip & compute statistics
        if self.stationary:
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
            # TODO: update this to offer the option to compute noise for each channel independantly
            self.y_noise = np.mean(self.y_noise, axis=0)

            if clip_noise_stationary:
                self.y_noise = self.y_noise[:chunk_size]

            # compute statistics over noise
            self.compute_noise_statistics()

    def compute_noise_statistics(self):
        # calculate statistics over y_noise
        _, _, noise_stft = stft(
            self.y_noise,
            nfft=self._n_fft,
            noverlap=self._win_length - self._hop_length,
            nperseg=self._win_length,
            padded=False,
        )
        noise_stft_db = _amp_to_db(noise_stft)
        self.mean_freq_noise = np.mean(noise_stft_db, axis=1)
        self.std_freq_noise = np.std(noise_stft_db, axis=1)

        self.noise_thresh = (
            self.mean_freq_noise + self.std_freq_noise * self.n_std_thresh
        )

    def spectral_gating(self, chunk):
        """run spectral gating"""
        denoised_channels = np.zeros(chunk.shape, chunk.dtype)
        for ci, channel in enumerate(chunk):
            _, _, sig_stft = stft(
                channel,
                nfft=self._n_fft,
                noverlap=self._win_length - self._hop_length,
                nperseg=self._win_length,
                padded=False,
            )

            # spectrogram of signal in dB
            sig_stft_db = _amp_to_db(sig_stft)

            if self.stationary:
                # calculate the threshold for each frequency/time bin
                db_thresh = np.repeat(
                    np.reshape(self.noise_thresh, [1, len(self.mean_freq_noise)]),
                    np.shape(sig_stft_db)[1],
                    axis=0,
                ).T
            else:
                # get the mean and std over the frequency axis for the current block_size
                mean_freq_noise = uniform_filter1d(
                    sig_stft_db,
                    size=self.noise_window_size_nonstationary,
                    axis=1,
                    mode="reflect",
                )
                squared_diff = (sig_stft_db - mean_freq_noise) ** 2
                variance = uniform_filter1d(
                    squared_diff,
                    size=self.noise_window_size_nonstationary,
                    mode="reflect",
                )
                std_freq_noise = np.sqrt(variance)
                db_thresh = mean_freq_noise + std_freq_noise * self.n_std_thresh

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
            _, denoised_signal = istft(
                sig_stft_denoised,
                nfft=self._n_fft,
                noverlap=self._win_length - self._hop_length,
                nperseg=self._win_length,
            )
            denoised_channels[ci, : len(denoised_signal)] = denoised_signal
        return denoised_channels

    def _do_filter(self, chunk):
        """Do the actual filtering"""
        chunk_filtered = self.spectral_gating(chunk)

        return chunk_filtered

    def _generate_mask_smoothing_filter(self, freq_mask_smooth_hz, time_mask_smooth_ms):
        if freq_mask_smooth_hz is None:
            n_grad_freq = 1
        else:
            # filter to smooth the mask
            n_grad_freq = int(freq_mask_smooth_hz / (self.sr / (self._n_fft / 2)))
            if n_grad_freq < 1:
                raise ValueError(
                    "freq_mask_smooth_hz needs to be at least {}Hz".format(
                        int((self.sr / (self._n_fft / 2)))
                    )
                )

        if time_mask_smooth_ms is None:
            n_grad_time = 1
        else:
            n_grad_time = int(
                time_mask_smooth_ms / ((self._hop_length / self.sr) * 1000)
            )
            if n_grad_time < 1:
                raise ValueError(
                    "time_mask_smooth_ms needs to be at least {}ms".format(
                        int((self._hop_length / self.sr) * 1000)
                    )
                )
        if (n_grad_time == 1) & (n_grad_freq == 1):
            self.smooth_mask = False
        else:
            self.smooth_mask = True
            self._smoothing_filter = _smoothing_filter(n_grad_freq, n_grad_time)

    def _read_chunk(self, i1, i2):
        """read chunk and pad with zerros"""
        if i1 < 0:
            i1b = 0
        else:
            i1b = i1
        if i2 > self.n_frames:
            i2b = self.n_frames
        else:
            i2b = i2
        chunk = np.zeros((self.n_channels, i2 - i1))
        chunk[:, i1b - i1 : i2b - i1] = self.y[:, i1b:i2b]
        return chunk

    def filter_chunk(self, start_frame, end_frame):
        """Pad and perform filtering"""
        i1 = start_frame - self.padding
        i2 = end_frame + self.padding
        padded_chunk = self._read_chunk(i1, i2)
        filtered_padded_chunk = self._do_filter(padded_chunk)
        return filtered_padded_chunk[:, start_frame - i1 : end_frame - i1]

    def _get_filtered_chunk(self, ind):
        """Grabs a single chunk"""
        start0 = ind * self._chunk_size
        end0 = (ind + 1) * self._chunk_size
        return self.filter_chunk(start_frame=start0, end_frame=end0)

    def _iterate_chunk(self, filtered_chunk, pos, end0, start0, ich):
        filtered_chunk0 = self._get_filtered_chunk(ich)
        filtered_chunk[:, pos : pos + end0 - start0] = filtered_chunk0[:, start0:end0]
        pos += end0 - start0

    def get_traces(self, start_frame=None, end_frame=None):
        """Grab filtered data iterating over chunks"""
        if start_frame is None:
            start_frame = 0
        if end_frame is None:
            end_frame = self.n_frames

        if self._chunk_size is not None:
            if end_frame - start_frame > self._chunk_size:
                ich1 = int(start_frame / self._chunk_size)
                ich2 = int((end_frame - 1) / self._chunk_size)

                # write output to temp memmap for parallelization
                with tempfile.NamedTemporaryFile(prefix=self._tmp_folder) as fp:
                    # create temp file
                    filtered_chunk = np.memmap(
                        fp,
                        dtype=self._dtype,
                        shape=(self.n_channels, int(end_frame - start_frame)),
                        mode="w+",
                    )
                    pos_list = []
                    start_list = []
                    end_list = []
                    pos = 0
                    for ich in range(ich1, ich2 + 1):
                        if ich == ich1:
                            start0 = start_frame - ich * self._chunk_size
                        else:
                            start0 = 0
                        if ich == ich2:
                            end0 = end_frame - ich * self._chunk_size
                        else:
                            end0 = self._chunk_size
                        pos_list.append(pos)
                        start_list.append(start0)
                        end_list.append(end0)
                        pos += end0 - start0
                    if len(start_list) > 1:
                        warn(
                            f"Computing across {len(start_list)} chunks. Increasing chunk_size may speed up computation."
                        )
                    Parallel(n_jobs=self.n_jobs)(
                        delayed(self._iterate_chunk)(
                            filtered_chunk, pos, end0, start0, ich
                        )
                        for pos, start0, end0, ich in zip(
                            tqdm(pos_list, disable=not (self.use_tqdm)),
                            start_list,
                            end_list,
                            range(ich1, ich2 + 1),
                        )
                    )
                    if self.flat:
                        return filtered_chunk.astype(self._dtype).flatten()
                    else:
                        return filtered_chunk.astype(self._dtype)

        filtered_chunk = self.filter_chunk(start_frame=0, end_frame=end_frame)
        if self.flat:
            return filtered_chunk.astype(self._dtype).flatten()
        else:
            return filtered_chunk.astype(self._dtype)
