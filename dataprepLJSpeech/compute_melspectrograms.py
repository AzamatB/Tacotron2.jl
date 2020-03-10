# computes mel spectrogram via librosa with the following audio parameters
filter_length  = 1024
hop_length     = 256
win_length     = 1024
n_mel_channels = 80
mel_fmin       = 0.0
mel_fmax       = 8000.0

import numpy as np
import librosa
import os
from os.path import join

datadir = '/Users/aza/Projects/TTS/data/LJSpeech-1.1'
wavsdir = join(datadir, 'wavs')
melspectrogramsdir = join(datadir, 'melspectrograms')

for (root, _, files) in os.walk(wavsdir):
   for file in files:
      wavfile = join(root, file)
      npyfile = join(melspectrogramsdir, file[:-4])
      audio_normalized, sampling_rate = librosa.core.load(wavfile)
      melspectrogram = librosa.feature.melspectrogram(y=audio_normalized, sr=sampling_rate, n_fft=filter_length, hop_length=hop_length, win_length=win_length, n_mels=n_mel_channels, fmin=mel_fmin, fmax=mel_fmax)
      np.save(npyfile, melspectrogram)
