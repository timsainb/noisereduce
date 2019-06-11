[![Build Status](https://travis-ci.com/timsainb/noisereduce.svg?branch=master)](https://travis-ci.com/timsainb/noisereduce)
[![Coverage Status](https://coveralls.io/repos/github/timsainb/noisereduce/badge.svg?branch=master)](https://coveralls.io/github/timsainb/noisereduce?branch=master)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/timsainb/noisereduce/master?filepath=notebooks%2F1.0-test-noise-reduction.ipynb)
[![PyPI version](https://badge.fury.io/py/noisereduce.svg)](https://badge.fury.io/py/noisereduce)
[![DOI](https://zenodo.org/badge/174219582.svg)](https://zenodo.org/badge/latestdoi/174219582)


# Noise reduction in python using spectral gating
- This algorithm is based (but not completely reproducing) on the one [outlined by Audacity](https://wiki.audacityteam.org/wiki/How_Audacity_Noise_Reduction_Works) for the **noise reduction effect** ([Link to C++ code](https://github.com/audacity/audacity/blob/master/src/effects/NoiseReduction.cpp))
- The algorithm requires two inputs: 
    1. A *noise* audio clip comtaining prototypical noise of the audio clip
    2. A *signal* audio clip containing the signal and the noise intended to be removed

## Steps of algorithm
1. An FFT is calculated over the noise audio clip
2. Statistics are calculated over FFT of the the noise (in frequency)
3. A threshold is calculated based upon the statistics of the noise (and the desired sensitivity of the algorithm) 
4. An FFT is calculated over the signal
5. A mask is determined by comparing the signal FFT to the threshold
6. The mask is smoothed with a filter over frequency and time
7. The mask is appled to the FFT of the signal, and is inverted

## Installation
`pip install noisereduce`

*noisereduce optionally uses Tensorflow as a backend to speed up FFT and gaussian convolution. It is not listed in the requirements.txt so because (1) it is optional and (2) tensorflow-gpu and tensorflow (cpu) are both compatible with this package. The package requires Tensorflow 2+ for all tensorflow operations.* 

## Usage
(see notebooks)

```
import noisereduce as nr
# load data
rate, data = wavfile.read("mywav.wav")
# select section of data that is noise
noisy_part = data[10000:15000]
# perform noise reduction
reduced_noise = nr.reduce_noise(audio_clip=data, noise_clip=noisy_part, verbose=True)
```

### Arguments to `noise_reduce`
```
n_grad_freq (int): how many frequency channels to smooth over with the mask.
n_grad_time (int): how many time channels to smooth over with the mask.
n_fft (int): number audio of frames between STFT columns.
win_length (int): Each frame of audio is windowed by `window()`. The window will be of length `win_length` and then padded with zeros to match `n_fft`..
hop_length (int):number audio of frames between STFT columns.
n_std_thresh (int): how many standard deviations louder than the mean dB of the noise (at each frequency level) to be considered signal
prop_decrease (float): To what extent should you decrease noise (1 = all, 0 = none)
pad_clipping (bool): Pad the signals with zeros to ensure that the reconstructed data is equal length to the data
        use_tensorflow (bool): Use tensorflow as a backend for convolution and fft to speed up computation
verbose (bool): Whether to plot the steps of the algorithm
```
<div style="text-align:center">
<p align="center">
  <img src="assets/noisereduce.png", width="100%">
</p>
</div>


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
