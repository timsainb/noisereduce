from noisereduce.spectralgate.stationary import SpectralGateStationary
from noisereduce.spectralgate.nonstationary import SpectralGateNonStationary

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
if TORCH_AVAILABLE:
    from noisereduce.spectralgate.streamed_torch_gate import StreamedTorchGate


def reduce_noise(
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
        use_torch=False,
        device="cuda",
):
    """
    Reduce noise via spectral gating.

    Parameters
    ----------
    y : np.ndarray [shape=(# frames,) or (# channels, # frames)], real-valued
        input signal
    sr : int
        sample rate of input signal / noise signal
    y_noise : np.ndarray [shape=(# frames,) or (# channels, # frames)], real-valued
        noise signal to compute statistics over (only for stationary noise reduction).
    stationary : bool, optional
        Whether to perform stationary, or non-stationary noise reduction, by default False
    prop_decrease : float, optional
        The proportion to reduce the noise by (1.0 = 100%), by default 1.0
    time_constant_s : float, optional
        The time constant, in seconds, to compute the noise floor in the non-stationary
        algorithm, by default 2.0
    freq_mask_smooth_hz : int, optional
        The frequency range to smooth the mask over in Hz, by default 500
    time_mask_smooth_ms : int, optional
        The time range to smooth the mask over in milliseconds, by default 50
    thresh_n_mult_nonstationary : int, optional
        Only used in nonstationary noise reduction., by default 1
    sigmoid_slope_nonstationary : int, optional
        Only used in nonstationary noise reduction., by default 10
    n_std_thresh_stationary : int, optional
        Number of standard deviations above mean to place the threshold between
        signal and noise., by default 1.5
    tmp_folder : [type], optional
        Temp folder to write waveform to during parallel processing. Defaults to
        default temp folder for python., by default None
    chunk_size : int, optional
        Size of signal chunks to reduce noise over. Larger sizes
        will take more space in memory, smaller sizes can take longer to compute.
        , by default 60000
        padding : int, optional
        How much to pad each chunk of signal by. Larger pads are
        needed for larger time constants., by default 30000
    n_fft : int, optional
        length of the windowed signal after padding with zeros.
        The number of rows in the STFT matrix ``D`` is ``(1 + n_fft/2)``.
        The default value, ``n_fft=2048`` samples, corresponds to a physical
        duration of 93 milliseconds at a sample rate of 22050 Hz, i.e. the
        default sample rate in librosa. This value is well adapted for music
        signals. However, in speech processing, the recommended value is 512,
        corresponding to 23 milliseconds at a sample rate of 22050 Hz.
        In any case, we recommend setting ``n_fft`` to a power of two for
        optimizing the speed of the fast Fourier transform (FFT) algorithm., by default 1024
    win_length : [type], optional
        Each frame of audio is windowed by ``window`` of length ``win_length``
        and then padded with zeros to match ``n_fft``.
        Smaller values improve the temporal resolution of the STFT (i.e. the
        ability to discriminate impulses that are closely spaced in time)
        at the expense of frequency resolution (i.e. the ability to discriminate
        pure tones that are closely spaced in frequency). This effect is known
        as the time-frequency localization trade-off and needs to be adjusted
        according to the properties of the input signal ``y``.
        If unspecified, defaults to ``win_length = n_fft``., by default None
    hop_length : [type], optional
        number of audio samples between adjacent STFT columns.
        Smaller values increase the number of columns in ``D`` without
        affecting the frequency resolution of the STFT.
        If unspecified, defaults to ``win_length // 4`` (see below)., by default None
    use_tqdm : bool, optional
        Whether to show tqdm progress bar, by default False
    n_jobs : int, optional
        Number of parallel jobs to run. Set at -1 to use all CPU cores, by default 1
    use_torch: bool, optional
        Whether to use the torch version of spectral gating, by default False
    device: str, optional
        A device to run the torch spectral gating on, by default "cuda"
    """

    if use_torch:
        if not TORCH_AVAILABLE:
            raise ImportError(
                "Torch is not installed. Please install torch to use torch version of spectral gating."
            )
        if n_jobs != 1:
            raise ValueError(
                "n_jobs must be 1 when using torch version of spectral gating."
            )

    # if using pytorch,
    if use_torch:
        sg = StreamedTorchGate(
            y=y,
            sr=sr,
            stationary=stationary,
            y_noise=y_noise,
            prop_decrease=prop_decrease,
            time_constant_s=time_constant_s,
            freq_mask_smooth_hz=freq_mask_smooth_hz,
            time_mask_smooth_ms=time_mask_smooth_ms,
            thresh_n_mult_nonstationary=thresh_n_mult_nonstationary,
            sigmoid_slope_nonstationary=sigmoid_slope_nonstationary,
            tmp_folder=tmp_folder,
            chunk_size=chunk_size,
            padding=padding,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            clip_noise_stationary=clip_noise_stationary,
            use_tqdm=use_tqdm,
            n_jobs=n_jobs,
            device=device,
        )
    else:
        if stationary:
            sg = SpectralGateStationary(
                y=y,
                sr=sr,
                y_noise=y_noise,
                prop_decrease=prop_decrease,
                n_std_thresh_stationary=n_std_thresh_stationary,
                chunk_size=chunk_size,
                clip_noise_stationary=clip_noise_stationary,
                padding=padding,
                n_fft=n_fft,
                win_length=win_length,
                hop_length=hop_length,
                time_constant_s=time_constant_s,
                freq_mask_smooth_hz=freq_mask_smooth_hz,
                time_mask_smooth_ms=time_mask_smooth_ms,
                tmp_folder=tmp_folder,
                use_tqdm=use_tqdm,
                n_jobs=n_jobs,
            )

        else:
            sg = SpectralGateNonStationary(
                y=y,
                sr=sr,
                chunk_size=chunk_size,
                padding=padding,
                prop_decrease=prop_decrease,
                n_fft=n_fft,
                win_length=win_length,
                hop_length=hop_length,
                time_constant_s=time_constant_s,
                freq_mask_smooth_hz=freq_mask_smooth_hz,
                time_mask_smooth_ms=time_mask_smooth_ms,
                thresh_n_mult_nonstationary=thresh_n_mult_nonstationary,
                sigmoid_slope_nonstationary=sigmoid_slope_nonstationary,
                tmp_folder=tmp_folder,
                use_tqdm=use_tqdm,
                n_jobs=n_jobs,
            )
    return sg.get_traces()
