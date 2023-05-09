from scipy.io import wavfile
import noisereduce as nr
from noisereduce.generate_noise import band_limited_noise


def test_reduce_generated_noise_stationary_with_noise_clip():
    # load data
    wav_loc = "assets/fish.wav"
    rate, data = wavfile.read(wav_loc)

    # add noise
    noise_len = 2  # seconds
    noise = band_limited_noise(
        min_freq=2000, max_freq=12000, samples=len(data), samplerate=rate) * 10
    noise_clip = noise[: rate * noise_len]
    audio_clip_band_limited = data + noise
    return nr.reduce_noise(
        y=audio_clip_band_limited, sr=rate, y_noise=noise_clip, stationary=True
    )


def test_reduce_generated_noise_stationary_without_noise_clip():
    # load data
    wav_loc = "assets/fish.wav"
    rate, data = wavfile.read(wav_loc)

    # add noise
    noise = band_limited_noise(
        min_freq=2000, max_freq=12000, samples=len(data), samplerate=rate) * 10
    audio_clip_band_limited = data + noise
    return nr.reduce_noise(
        y=audio_clip_band_limited, sr=rate, stationary=True
    )


def test_reduce_generated_noise_nonstationary():
    # load data
    wav_loc = "assets/fish.wav"
    rate, data = wavfile.read(wav_loc)

    # add noise
    noise = band_limited_noise(
        min_freq=2000, max_freq=12000, samples=len(data), samplerate=rate) * 10
    audio_clip_band_limited = data + noise
    return nr.reduce_noise(
        y=audio_clip_band_limited, sr=rate, stationary=False
    )


def test_reduce_generated_noise_batches():
    # load data
    wav_loc = "assets/fish.wav"
    rate, data = wavfile.read(wav_loc)

    # add noise
    noise = band_limited_noise(
        min_freq=2000, max_freq=12000, samples=len(data), samplerate=rate) * 10
    audio_clip_band_limited = data + noise
    return nr.reduce_noise(
        y=audio_clip_band_limited, sr=rate, stationary=False, chunk_size=30000
    )


def test_reduce_torch_cpu_stationary():
    # load data
    wav_loc = "assets/fish.wav"
    rate, data = wavfile.read(wav_loc)

    # add noise
    noise = band_limited_noise(
        min_freq=2000, max_freq=12000, samples=len(data), samplerate=rate) * 10
    audio_clip_band_limited = data + noise
    return nr.reduce_noise(
        y=audio_clip_band_limited, sr=rate, stationary=True, chunk_size=30000, use_torch=True, device='cpu'
    )


# def test_reduce_torch_cpu_stationary_cuda():
#     # load data
#     wav_loc = "assets/fish.wav"
#     rate, data = wavfile.read(wav_loc)
#
#     # add noise
#     noise = band_limited_noise(
#         min_freq=2000, max_freq=12000, samples=len(data), samplerate=rate) * 10
#     audio_clip_band_limited = data + noise
#     return nr.reduce_noise(
#         y=audio_clip_band_limited, sr=rate, stationary=True, chunk_size=30000, use_torch=True, device='cuda'
#     )


def test_reduce_torch_cpu_non_stationary():
    # load data
    wav_loc = "assets/fish.wav"
    rate, data = wavfile.read(wav_loc)

    # add noise
    noise = band_limited_noise(
        min_freq=2000, max_freq=12000, samples=len(data), samplerate=rate) * 10
    audio_clip_band_limited = data + noise
    return nr.reduce_noise(
        y=audio_clip_band_limited, sr=rate, stationary=False, chunk_size=30000, use_torch=True, device='cpu'
    )


# def test_reduce_torch_cpu_non_stationary_cuda():
#     # load data
#     wav_loc = "assets/fish.wav"
#     rate, data = wavfile.read(wav_loc)
#
#     # add noise
#     noise = band_limited_noise(
#         min_freq=2000, max_freq=12000, samples=len(data), samplerate=rate) * 10
#     audio_clip_band_limited = data + noise
#     return nr.reduce_noise(
#         y=audio_clip_band_limited, sr=rate, stationary=False, use_torch=True, device='cuda'
#     )