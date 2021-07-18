from scipy.io import wavfile
import noisereduce as nr
from noisereduce.generate_noise import band_limited_noise
from noisereduce.utils import int16_to_float32, float32_to_int16


def test_reduce_generated_noise_stationary_with_noise_clip():
    # load data
    wav_loc = "assets/fish.wav"
    rate, data = wavfile.read(wav_loc)

    # add noise
    noise_len = 2  # seconds
    noise = (
        band_limited_noise(
            min_freq=2000, max_freq=12000, samples=len(data), samplerate=rate
        )
        * 10
    )
    noise_clip = noise[: rate * noise_len]
    audio_clip_band_limited = data + noise
    return nr.reduce_noise(
        y=audio_clip_band_limited, sr=rate, y_noise = noise_clip, stationary=True
    )


def test_reduce_generated_noise_stationary_without_noise_clip():
    # load data
    wav_loc = "assets/fish.wav"
    rate, data = wavfile.read(wav_loc)

    # add noise
    noise_len = 2  # seconds
    noise = (
        band_limited_noise(
            min_freq=2000, max_freq=12000, samples=len(data), samplerate=rate
        )
        * 10
    )
    audio_clip_band_limited = data + noise
    return nr.reduce_noise(
        y=audio_clip_band_limited, sr=rate, stationary=True
    )

def test_reduce_generated_noise_nonstationary():
    # load data
    wav_loc = "assets/fish.wav"
    rate, data = wavfile.read(wav_loc)

    # add noise
    noise_len = 2  # seconds
    noise = (
        band_limited_noise(
            min_freq=2000, max_freq=12000, samples=len(data), samplerate=rate
        )
        * 10
    )
    noise_clip = noise[: rate * noise_len]
    audio_clip_band_limited = data + noise
    return nr.reduce_noise(
        y=audio_clip_band_limited, sr=rate, stationary=False
    )

def test_reduce_generated_noise_batches():
    # load data
    wav_loc = "assets/fish.wav"
    rate, data = wavfile.read(wav_loc)

    # add noise
    noise_len = 2  # seconds
    noise = (
        band_limited_noise(
            min_freq=2000, max_freq=12000, samples=len(data), samplerate=rate
        )
        * 10
    )
    noise_clip = noise[: rate * noise_len]
    audio_clip_band_limited = data + noise
    return nr.reduce_noise(
        y=audio_clip_band_limited, sr=rate, stationary=False, chunk_size=30000
    )



### Tests for V1
import noisereduce.noisereducev1 as nrv1

def test_reduce_generated_noise():
    # load data
    wav_loc = "assets/fish.wav"
    rate, data = wavfile.read(wav_loc)
    data = int16_to_float32(data)
    # add noise
    noise_len = 2  # seconds
    noise = (
        band_limited_noise(
            min_freq=2000, max_freq=12000, samples=len(data), samplerate=rate
        )
        * 10
    )
    noise_clip = noise[: rate * noise_len]
    audio_clip_band_limited = data + noise
    return nrv1.reduce_noise(
        audio_clip=audio_clip_band_limited, noise_clip=noise_clip, verbose=True
    )


def test_reduce_cafe_noise():
    # load data
    wav_loc = "assets/fish.wav"
    rate, data = wavfile.read(wav_loc)
    data = int16_to_float32(data)
    noise_loc = "assets/cafe_short.wav"
    noise_rate, noise_data = wavfile.read(noise_loc)
    noise_data = int16_to_float32(noise_data)
    # add noise
    snr = 2  # signal to noise ratio
    noise_clip = noise_data / snr
    audio_clip_cafe = data + noise_clip

    # reduce noise
    reduced_noise = nrv1.reduce_noise(
        audio_clip=audio_clip_cafe, noise_clip=noise_clip, verbose=True
    )
    return float32_to_int16(reduced_noise)


def test_reduce_cafe_noise_tf():
    # load data
    wav_loc = "assets/fish.wav"
    rate, data = wavfile.read(wav_loc)
    data = int16_to_float32(data)
    noise_loc = "assets/cafe_short.wav"
    noise_rate, noise_data = wavfile.read(noise_loc)
    noise_data = int16_to_float32(noise_data)
    # add noise
    snr = 2  # signal to noise ratio
    noise_clip = noise_data / snr
    audio_clip_cafe = data + noise_clip

    # reduce noise
    reduced_noise = nrv1.reduce_noise(
        audio_clip=audio_clip_cafe,
        noise_clip=noise_clip,
        use_tensorflow=True,
        verbose=True,
    )
    return float32_to_int16(reduced_noise)