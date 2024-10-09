[![Build Status](https://travis-ci.com/timsainb/noisereduce.svg?branch=master)](https://travis-ci.com/timsainb/noisereduce)
[![Coverage Status](https://coveralls.io/repos/github/timsainb/noisereduce/badge.svg?branch=master)](https://coveralls.io/github/timsainb/noisereduce?branch=master)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/timsainb/noisereduce/master?filepath=notebooks%2F1.0-test-noise-reduction.ipynb)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/timsainb/noisereduce/blob/master/notebooks/1.0-test-noise-reduction.ipynb)
[![PyPI version](https://badge.fury.io/py/noisereduce.svg)](https://badge.fury.io/py/noisereduce)

<div style="text-align:center">
<p align="center">
  <img src="assets/noisereduce.png", width="100%">
</p>
</div>

# Noise reduction in python using spectral gating
Noisereduce is a noise reduction algorithm in python that reduces noise in time-domain signals like speech, bioacoustics, and physiological signals. It relies on a method called "spectral gating" which is a form of [Noise Gate](https://en.wikipedia.org/wiki/Noise_gate). It works by computing a spectrogram of a signal (and optionally a noise signal) and estimating a noise threshold (or gate) for each frequency band of that signal/noise. That threshold is used to compute a mask, which gates noise below the frequency-varying threshold. 

### How noisereduce works
The basic intuition is that statistics are calculated on  each frequency channel to determine a noise gate. Then the gate is applied to the signal.

The algorithm takes two inputs: 
    1. A *noise* clip containing prototypical noise of clip (optional)
    2. A *signal* clip containing the signal and the noise intended to be removed

### Steps of the Noise Reduction algorithm
1. A spectrogram is calculated over the noise audio clip
2. Statistics are calculated over spectrogram of the noise (in frequency)
3. A threshold is calculated based upon the statistics of the noise (and the desired sensitivity of the algorithm) 
4. A spectrogram is calculated over the signal
5. A mask is determined by comparing the signal spectrogram to the threshold
6. The mask is smoothed with a filter over frequency and time
7. The mask is applied to the spectrogram of the signal, and is inverted
*If the noise signal is not provided, the algorithm will treat the signal as the noise clip, which tends to work pretty well*

# Installation
`pip install noisereduce`

# Usage
See example notebook: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/timsainb/noisereduce/blob/master/notebooks/1.0-test-noise-reduction.ipynb)
Parallel computing example: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/timsainb/noisereduce/blob/master/notebooks/2.0-test-noisereduce-pytorch.ipynb)

## reduce_noise

### Simplest usage
```
from scipy.io import wavfile
import noisereduce as nr
# load data
rate, data = wavfile.read("mywav.wav")
# perform noise reduction
reduced_noise = nr.reduce_noise(y=data, sr=rate)
wavfile.write("mywav_reduced_noise.wav", rate, reduced_noise)
```

### Most important arguments for `reduce_noise`

Choosing parameters that fit your data type is crucial to getting reasonable noise reduction. 

```
  y : np.ndarray [shape=(# frames,) or (# channels, # frames)], real-valued
      input signal
  sr : int
      sample rate of input signal / noise signal
  y_noise : np.ndarray [shape=(# frames,) or (# channels, # frames)], real-valued
      noise signal to compute statistics over (only for non-stationary noise reduction).
  stationary : bool, optional
      Whether to perform stationary (True), or non-stationary (False) noise reduction, by default True
  n_std_thresh : int, optional
      Number of standard deviations above mean to place the threshold between
      signal and noise, by default 1.5
  noise_window_size_nonstationary_ms: float, optional
      The window size (in milliseconds) to compute the noise floor over in the non-stationary algorithm, by default None
  prop_decrease : float, optional
      The proportion to reduce the noise by (1.0 = 100%), by default 1.0
  freq_mask_smooth_hz : int, optional
      The frequency range to smooth the mask over in Hz, by default 500
  time_mask_smooth_ms : int, optional
      The time range to smooth the mask over in milliseconds, by default 50
  use_torch: bool, optional
      Whether to use the torch version of spectral gating, by default False
```

There are a few other important arguments related to computed spectrograms (e.g. `n_fft`). A good way to select spectrogram parameters is to visualize your signal with a spectrogram and choose settings that allow you visually to discriminate between signal and noise. 

## Stationary and nonstationary noise reduction

noisereduce comprises two algorithms:
1. **Stationary Noise Reduction**: Keeps the estimated noise threshold at the same level across the whole signal
2. **Non-stationary Noise Reduction**: Continuously updates the estimated noise threshold over time

### Non-stationary Noise Reduction
- The non-stationary noise reduction algorithm is an extension of the stationary noise reduction algorithm, but allowing the noise gate to change over time. 
- When you know the timescale that your signal occurs on (e.g. a bird call can be a few hundred milliseconds), you can set your noise threshold based on the assumption that events occurring on longer timescales are noise. 
- The nonstationary version of noisereduce computes the noise threshold over a moving window. 
- I discuss stationary and non-stationary noise reduction in [this paper](https://www.frontiersin.org/articles/10.3389/fnbeh.2021.811737/full). 

<div style="text-align:center">
<p align="center">
  <img src="assets/stationary-vs-nonstationary.jpg", width="100%">
</p>
</div>

*Figure caption: Stationary and non-stationary spectral gating noise reduction. (A) An overview of each algorithm. Stationary noise reduction typically takes in an explicit noise signal to calculate statistics and performs noise reduction over the entire signal uniformly. Non-stationary noise reduction dynamically estimates and reduces noise concurrently. (B) Stationary and non-stationary spectral gating noise reduction using the noisereduce Python package (Sainburg, 2019) applied to a Common chiffchaff (Phylloscopus collybita) song (Stowell et al., 2019) with an airplane noise in the background. The bottom frame depicts the difference between the two algorithms.*


## Using noisereduce with torch (faster)
See example notebook: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/timsainb/noisereduce/blob/master/notebooks/3.0-torchgate-as-nn-module.ipynb)

### Simplest usage
The simplest way to use torch is to add the use_torch flag to `reduce_noise`
```
reduced_noise = nr.reduce_noise(y=data, sr=rate, use_torch=True)
```

Alternatively, you can use torchgate directly:

```
import torch
from noisereduce.torchgate import TorchGate as TG
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Create TorchGating instance
tg = TG(sr=8000, nonstationary=True).to(device)

# Apply Spectral Gate to noisy speech signal
noisy_speech = torch.randn(3, 32000, device=device)
enhanced_speech = tg(noisy_speech)
```


# Changelog 

### Version 4 Updates:
- Reformulated the stationary and non-stationary versions of noisereduce to rely on a more similar noise thresholding

### Version 3 Updates:
- Includes a PyTorch-based implementation of Spectral Gating, an algorithm for denoising audio signals. 
- You can now create a noisereduce nn.Module object which allows it to be used either as a standalone module or as part of a larger neural network architecture.
- The run time of the algorithm decreases substantially.

### Version 2 Updates:
- Added two forms of spectral gating noise reduction: stationary noise reduction, and non-stationary noise reduction. 
- Added multiprocessing so you can perform noise reduction on bigger data.
- The new version breaks the API of the old version. 
- The previous version is still available at `from noisereduce.noisereducev1 import reduce_noise`
- You can now create a noisereduce object which allows you to reduce noise on subsets of longer recordings

## Citation
If you use this code in your research, please cite it:
```

@article{sainburg2020finding,
  title={Finding, visualizing, and quantifying latent structure across diverse animal vocal repertoires},
  author={Sainburg, Tim and Thielk, Marvin and Gentner, Timothy Q},
  journal={PLoS computational biology},
  volume={16},
  number={10},
  pages={e1008228},
  year={2020},
  publisher={Public Library of Science}
}

@software{tim_sainburg_2019_3243139,
  author       = {Tim Sainburg},
  title        = {timsainb/noisereduce: v1.0},
  month        = jun,
  year         = 2019,
  publisher    = {Zenodo},
  version      = {db94fe2},
  doi          = {10.5281/zenodo.3243139},
  url          = {https://doi.org/10.5281/zenodo.3243139}
}
```

# Related work and inspiration
- This algorithm is inspired by the Spectral Gate [outlined by Audacity](https://wiki.audacityteam.org/wiki/How_Audacity_Noise_Reduction_Works) used in the *noise reduction effect* ([Link to C++ code](https://github.com/audacity/audacity/blob/master/src/effects/NoiseReduction.cpp)). These algorithms are related but are not the same. 
- [pyroomacoustics](https://github.com/LCAV/pyroomacoustics) has both noise reduction algorithms, and can be used to simulate audio
- [scipy.signal](https://docs.scipy.org/doc/scipy/reference/signal.html) has some noise reduction functions (e.g. Wiener and Savitzky-Golay filter) 
- [pywt](https://pywavelets.readthedocs.io/en/latest/) can be used to denoise signals with wavelet decomposition


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>


