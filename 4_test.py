from tensorflow.keras.models import load_model
from sklearn.externals import joblib
from hparams import *
from processing.proc_audio import from_spectro_to_waveform
import matplotlib.pyplot as plt
import pandas as pd
import librosa
import librosa.filters
import numpy as np
import tensorflow as tf
from scipy import signal
from scipy.io import wavfile


def save_wav(wav, path, sr):
	wav *= 32767 / max(0.01, np.max(np.abs(wav)))
	#proposed by @dsmiller
	wavfile.write(path, sr, wav.astype(np.int16))


# load testing data
decoder_input_testing = joblib.load('out/decoder_input_testing.pkl')
mel_spectro_testing = joblib.load('out/mel_spectro_testing.pkl')
spectro_testing = joblib.load('out/spectro_testing.pkl')
text_input_testing = joblib.load('out/label_testing.pkl')

# load model
saved_model = load_model('results/model.h5')

predictions = saved_model.predict([text_input_testing, decoder_input_testing])

mel_pred = predictions[0]  # predicted mel spectrogram
mag_pred = predictions[1]  # predicted mag spectrogram


item_index = 0  # pick any index

predicted_spectro_item = mag_pred[item_index]
predicted_audio_item = from_spectro_to_waveform(predicted_spectro_item, N_FFT,
                                                HOP_LENGTH, WIN_LENGTH,
                                                N_ITER, WINDOW_TYPE,
                                                MAX_DB, REF_DB, PREEMPHASIS)

import librosa.display
plt.figure(figsize=(14, 5))
save_wav(predicted_audio_item,'./results/temp.wav',sr=SAMPLING_RATE)
librosa.display.waveplot(predicted_audio_item, sr=SAMPLING_RATE)
plt.show()
