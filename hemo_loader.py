import json
import numpy as np
import os
import pandas as pd
import pickle
import random
import torch
import torch.nn.functional as F
from hemo_record import NUM_STEPS, RHC_TYPE, ACC_CHANNELS, ECG_CHANNELS, IGNORE
from torch.utils.data import Dataset, DataLoader


class HemoDataset(Dataset):

  def __init__(self, segments, add_noise, global_stats):
    self.segments = segments
    self.global_stats = global_stats
    self.add_noise = add_noise

  def add_amplitude_noise(self, signal, noise_level=0.05):
    noise = torch.randn_like(signal) * noise_level
    return signal + noise

  def apply_channel_dropout(self, signal, dropout_prob=0.2):
    if torch.rand(1).item() < dropout_prob:
      channels_to_zero = torch.rand(signal.size(0)) < dropout_prob
      signal[channels_to_zero] = 0.0
    return signal

  def get_noisy_val(self, val, min_, max_):
    percent_change = random.randint(min_, max_) / 100
    operation = random.choice([1, -1]) * percent_change
    noisy_val = (1 + operation) * val
    return noisy_val

  def pad(self, signal):
    if signal.shape[-1] < NUM_STEPS:
      padding = NUM_STEPS - signal.shape[-1]
      signal = F.pad(signal, (0, padding))
    elif signal.shape[-1] > NUM_STEPS:
      signal = signal[:, :, :NUM_STEPS]
    return signal

  def invert(self, signal):
    return torch.tensor(signal.T, dtype=torch.float32)

  def __len__(self):
    return len(self.segments)

  def __getitem__(self, index):
    segment = self.segments[index]
    
    shift = random.randint(-100, +100)

    acc = self.pad(self.invert(segment['acc']))
    ecg = self.pad(self.invert(segment['ecg']))
  
    bmi = segment['weight'] / ((segment['height'] / 100) ** 2)
    bmi = torch.tensor(bmi, dtype=torch.float32)
    
    rhc = segment[RHC_TYPE]
    rhc = torch.tensor(np.array([rhc]), dtype=torch.float32)

    if self.add_noise:

      ecg = torch.roll(ecg, shifts=shift, dims=-1)
      ecg = self.add_amplitude_noise(ecg)

      acc = torch.roll(acc, shifts=shift, dims=-1)
      acc = self.add_amplitude_noise(acc)
      acc = self.apply_channel_dropout(acc, dropout_prob=0.3)
      
      bmi = self.get_noisy_val(bmi, 1, 4)

      rhc = self.get_noisy_val(rhc, 1, 2)

    return acc, ecg, bmi, rhc, segment['record_name']


def get_loader(segment_path, global_stats, add_noise, batch_size):
  with open(segment_path, 'rb') as f:
    segments = pickle.load(f)
  dataset = HemoDataset(segments, add_noise, global_stats)
  loader = DataLoader(dataset, batch_size=batch_size, shuffle=add_noise)
  return loader
