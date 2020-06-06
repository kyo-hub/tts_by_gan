import pandas as pd
import numpy as np
from sklearn.externals import joblib
from tqdm import tqdm
from processing.proc_audio import get_padded_spectros
from hparams import *
import tensorflow as tf
import argparse, os

text_name = 'transcript.v.1.3.txt'

def preprocess_in(args):
  in_dir = os.path.join(args.base_dir, args.input)
  out_dir = os.path.join(args.base_dir, args.output)
  os.makedirs(out_dir, exist_ok=True)
  dot_wav_filepaths, index = readfile_path(in_dir, out_dir)
  decoder_input,melspectro_data,spectro_data = calculate_wav_to_number(dot_wav_filepaths)
  write_metadata(decoder_input,melspectro_data,spectro_data,index)

def readfile_path(in_dir, out_dir):
  print('Get file paths')
  index = 1
  dot_wav_filepaths = []
  with open(os.path.join(in_dir, text_name), encoding='utf-8') as f:
    for line in f:
      parts = line.strip().split('|')
      wav_path = os.path.join(in_dir, parts[0])
      dot_wav_filepaths.append(wav_path)
      index += 1
  return dot_wav_filepaths, index

def calculate_wav_to_number(dot_wav_filepaths):
  mel_spectro_data = []
  spectro_data = []
  decoder_input = []
  print('Processing the audio samples (computation of spectrograms)...')
  for filepath in tqdm(dot_wav_filepaths):
      fname, mel_spectro, spectro = get_padded_spectros(filepath, r,
                                                        PREEMPHASIS, N_FFT,
                                                        HOP_LENGTH, WIN_LENGTH,
                                                        SAMPLING_RATE,
                                                        N_MEL, REF_DB,
                                                        MAX_DB)

      decod_inp_tensor = tf.concat((tf.zeros_like(mel_spectro[:1, :]),
                                    mel_spectro[:-1, :]), 0)
      decod_inp = decod_inp_tensor[:, -N_MEL:]
      print(decod_inp.shape)

      # Padding of the temporal dimension
      dim0_mel_spectro = mel_spectro.shape[0]
      dim1_mel_spectro = mel_spectro.shape[1]
      padded_mel_spectro = np.zeros((MAX_MEL_TIME_LENGTH, dim1_mel_spectro))
      padded_mel_spectro[:dim0_mel_spectro, :dim1_mel_spectro] = mel_spectro

      dim0_decod_inp = decod_inp.shape[0]
      dim1_decod_inp = decod_inp.shape[1]
      print(dim0_decod_inp)
      print(dim1_decod_inp)
      padded_decod_input = np.zeros((MAX_MEL_TIME_LENGTH, dim1_decod_inp))
      padded_decod_input[:dim0_decod_inp, :] = decod_inp

      dim0_spectro = spectro.shape[0]
      dim1_spectro = spectro.shape[1]
      padded_spectro = np.zeros((MAX_MAG_TIME_LENGTH, dim1_spectro))
      padded_spectro[:dim0_spectro, :dim1_spectro] = spectro

      mel_spectro_data.append(padded_mel_spectro)
      spectro_data.append(padded_spectro)
      decoder_input.append(padded_decod_input)
  return decoder_input,mel_spectro_data,spectro_data



def split_data(decoder_input, mel_spectro_data, spectro_data,index):
  print(index)
  print('Convert into np.array')
  
  decoder_input_array = np.array(decoder_input)
  mel_spectro_data_array = np.array(mel_spectro_data)
  spectro_data_array = np.array(spectro_data)
  print(decoder_input_array.shape)

  print('Split into training and testing data')
  len_train = int(TRAIN_SET_RATIO * (index-1))

  decoder_input_array_training = decoder_input_array[:len_train]
  decoder_input_array_testing = decoder_input_array[len_train:]

  mel_spectro_data_array_training = mel_spectro_data_array[:len_train]
  mel_spectro_data_array_testing = mel_spectro_data_array[len_train:]

  spectro_data_array_training = spectro_data_array[:len_train]
  spectro_data_array_testing = spectro_data_array[len_train:]
  
  return decoder_input_array_training,decoder_input_array_testing, mel_spectro_data_array_training, mel_spectro_data_array_testing, spectro_data_array_training, spectro_data_array_testing


def write_metadata(decoder_input,melspectro_data,spectro_data,index):
  decoder_input_array_training,decoder_input_array_testing, mel_spectro_data_array_training, mel_spectro_data_array_testing, spectro_data_array_training, spectro_data_array_testing = split_data(decoder_input,melspectro_data,spectro_data,index)
  print(decoder_input_array_training.shape)
  print('Save data as pkl')
  joblib.dump(decoder_input_array_training,'./out/decoder_input_training.pkl')
  joblib.dump(mel_spectro_data_array_training,'./out/mel_spectro_training.pkl')
  joblib.dump(spectro_data_array_training,'./out/spectro_training.pkl')

  joblib.dump(decoder_input_array_testing,'./out/decoder_input_testing.pkl')
  joblib.dump(mel_spectro_data_array_testing,'./out/mel_spectro_testing.pkl')
  joblib.dump(spectro_data_array_testing,'./out/spectro_testing.pkl')
            
def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--base_dir', default='./')
  parser.add_argument('--input', default='in')
  parser.add_argument('--output', default='out')
  args = parser.parse_args()
  preprocess_in(args)


if __name__ == "__main__":
  main()
