from scipy.io import wavfile
import noisereduce as nr
from noisereduce.generate_noise import band_limited_noise

def test_reduce_generated_noise():
	# load data
	wav_loc = "assets/fish.wav"
	rate, data= wavfile.read(wav_loc)
	data = data/32768.
	# add noise
	noise_len = 2 # seconds
	noise = band_limited_noise(min_freq=2000, max_freq = 12000, samples=len(data), samplerate=rate)*10
	noise_clip = noise[:rate*noise_len]
	audio_clip_band_limited = data+noise
	return nr.reduce_noise(audio_clip=audio_clip_band_limited, noise_clip=noise_clip, verbose=True)

def test_reduce_cafe_noise():
	# load data
	wav_loc = "assets/fish.wav"
	rate, data= wavfile.read(wav_loc)
	data = data/32768.
	noise_loc = "../assets/cafe_short.wav"
	noise_rate, noise_data= wavfile.read(noise_loc)
	noise_data = noise_data/32768
	# add noise
	snr = 2 # signal to noise ratio
	noise_clip = noise_data/snr
	audio_clip_cafe = data + noise_clip
	
	# reduce noise
	return nr.reduce_noise(audio_clip=audio_clip_cafe, noise_clip=noise_clip, verbose=True)