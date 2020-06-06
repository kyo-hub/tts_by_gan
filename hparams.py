# Audio/Spectral analysis
N_FFT = 1024
PREEMPHASIS = 0.97
SAMPLING_RATE = 20000
FRAME_LENGTH = 0.05  
FRAME_SHIFT = 0.0125 
HOP_LENGTH = int(SAMPLING_RATE * FRAME_SHIFT)
WIN_LENGTH = int(SAMPLING_RATE * FRAME_LENGTH)
N_MEL = 80
REF_DB = 20
MAX_DB = 100
r = 5
MAX_MEL_TIME_LENGTH = 200  # mel spectrogram
MAX_MAG_TIME_LENGTH = 850  # spectrogram
WINDOW_TYPE='hann'
N_ITER = 50

# Text
NB_CHARS_MAX = 200  # 글자 수 제한

# Deep Learning Model
K1 = 16  # encoder CBHG 의 conv크기
K2 = 8  # post processing CBHG에서의 conv크
BATCH_SIZE = 4
NB_EPOCHS = 5
EMBEDDING_SIZE = 256

# Other
TRAIN_SET_RATIO = 0.9
