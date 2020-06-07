import pandas as pd
from sklearn.externals import joblib
from processing.proc_text import text_to_sequence
from hparams import *
import argparse, os, re
import tensorflow as tf

text_name = 'transcript.v.1.3.txt'
filters = "([.,!?])"

def preprocess_in(args):
  in_dir = os.path.join(args.base_dir, args.input)
  out_dir = os.path.join(args.base_dir, args.output)
  os.makedirs(out_dir, exist_ok=True)
  label,index,length_for_embeding, len_seq_list = mk_label(in_dir, out_dir)
  label_training, label_testing, test_list= split_label(label,index,len_seq_list)
  save_label(label_training, label_testing, length_for_embeding, test_list)
  
def mk_label(in_dir, out_dir):
  index = 1
  label=[]
  diction=set()
  len_seq_list=[]
  with open(os.path.join(in_dir, text_name), encoding='utf-8') as f:
    for line in f:
      parts = line.strip().split('|')
      text = parts[3]
      text = re.sub(re.compile(filters), '', text)
      #텍스트 전처리. 특수기호 제거
      sequence, len_seq = text_to_sequence(text)
      label.append(sequence)
      index += 1
      set_label = set(sequence)
      diction.update(set_label)
      len_seq_list.append(len_seq)
  length_for_embeding = max(diction)
  print(label)
  label = tf.constant(label)
  print(label.shape)
  return label,index,length_for_embeding,len_seq_list

def split_label(label,index, len_seq_list):
    len_train = int(TRAIN_SET_RATIO * (index-1))
    label_training = label[:len_train]
    label_testing = label[len_train:]
    test_list = len_seq_list[len_train:]
    return label_training, label_testing, test_list

def save_label(label_training, label_testing, length_for_embeding, test_list):
    joblib.dump(label_training, 'out/label_training.pkl')
    joblib.dump(label_testing, 'out/label_testing.pkl')
    joblib.dump(length_for_embeding, 'out/length_for_embeding.pkl')
    joblib.dump(test_list,'out/test_list.pkl')

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--base_dir', default='./')
  parser.add_argument('--output', default='out')
  parser.add_argument('--input',default='in')
  args = parser.parse_args()
  preprocess_in(args)


if __name__ == "__main__":
  main()
