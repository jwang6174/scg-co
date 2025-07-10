import json
import math
import numpy as np
import os
import pandas as pd
import random
import statistics
import sys
from pathutil import DATABASE_PATH
from scipy.signal import butter, filtfilt
from wave_record import *

# Define number of time steps.
NUM_STEPS = 30 * SAMPLE_RATE

# Define RHC parameter to predict.
RHC_TYPE = 'Avg. COmL/min'

# Define ACC channels to use.
ACC_CHANNELS = [
  'patch_ACC_lat',
  'patch_ACC_dv',
  'patch_ACC_hf'
]

# Define ECG channels to use.
ECG_CHANNELS = [
  'patch_ECG'
]

# Segment info to ignore when normalizing.
IGNORE = set(['start', 'end', 'record_name', 'data_type'])

# Define records to use and index.
TEST_RECORD_NAME = None
VALID_RECORD_NAME = None
START_RECORD_INDEX = 1


def bandpass_filter(signal, lowcut=1, highcut=40, fs=500, order=4): 
  nyq = 0.5 * fs
  low = lowcut / nyq
  high = highcut / nyq
  b, a = butter(order, [low, high], btype='band')
  return filtfilt(b, a, signal)


def bandpass_all_channels(signal_np, lowcut, highcut, fs): 
  signal_np = signal_np.T
  filtered = np.vstack([
    bandpass_filter(signal_np[i], lowcut, highcut, fs, 4)
    for i in range(signal_np.shape[0])
  ])
  return filtered


def get_start_and_end_times(record_name): 
  path = os.path.join(DATABASE_PATH, 'processed_data', f'{record_name}.json')
  with open(path, 'r') as f:
    data = json.load(f)
    start = datetime.strptime(data['MacStTime'], '%d-%b-%Y %H:%M:%S')
    end = datetime.strptime(data['MacEndTime'], '%d-%b-%Y %H:%M:%S')
    return start, end


def get_logged_challenge_time(record_name): 
  filename = 'PAM_PCWP_timestamp_in_TBME.json'
  filepath = os.path.join(os.path.join(DATABASE_PATH, 'meta_information', filename))
  with open(filepath, 'r') as f:
    data = json.load(f)
    if record_name in data:
      time_str = data[record_name]['PA_VI']['datetime'].strip()
      time_obj = datetime.strptime(time_str, '%H:%M:%S')
      return time_obj
    else:
      raise IndexError(f'Record {record_name} not in {filename}')


def get_last_chamber_time(record_name): 
  path = os.path.join(DATABASE_PATH, 'processed_data', f'{record_name}.json')
  with open(path, 'r') as f:
    data = json.load(f)
    last_time = None
    for chamber, time_str in data['ChamEvents'].items():
      time_obj = datetime.strptime(time_str.strip(), '%d-%b-%Y %I:%M:%S %p')
      if last_time is None or time_obj > last_time:
        last_time = time_obj
    return last_time


def get_sample_index(start_time, sample_time): 
  start_time = start_time.replace(year=2000, month=1, day=1)
  sample_time = sample_time.replace(year=2000, month=1, day=1)
  diff = (sample_time - start_time).total_seconds()
  sample_index = int(diff * SAMPLE_RATE)
  return sample_index


def get_bin_info(segments): 
  vals = [s[RHC_TYPE] for s in segments]
  if RHC_TYPE == 'Avg. COmL/min':
    bin_width = 500
    min_val = int(np.floor(np.min(vals) / 1000) * 1000)
    max_val = int(np.ceil(np.max(vals) / 1000) * 1000)
  elif RHC_TYPE == 'RAM':
    bin_width = 2
    min_val = int(np.floor(np.min(vals) / 10) * 10)
    max_val = int(np.ceil(np.max(vals) / 10) * 10)
  elif RHC_TYPE == 'SVmL/beat':
    bin_width = 5
    min_val = int(np.floor(np.min(vals) / 10) * 10)
    max_val = int(np.ceil(np.max(vals) / 10) * 10)
  else:
    raise ValueError('Undefined RHC type for bin creation')

  num_bins = int(np.ceil((max_val - min_val) / bin_width))
  
  return bin_width, num_bins, min_val, max_val


def balance_segments(segments): 
  # Get bin info.
  bin_width, num_bins, min_val, max_val = get_bin_info(segments)

  # Initialize bins.
  bins = {}
  for i in range(int(np.floor(min_val)), int(np.ceil(max_val)), bin_width):
    bins[i] = []

  # Add segments to bins.
  for segment in segments:
    for bin_start in bins.keys():
      if bin_start <= segment[RHC_TYPE] < bin_start + bin_width:
        bins[bin_start].append(segment)

  # Determine most frequent bin.
  most_freq_bin = None
  most_freq_cnt = 0
  for bin_start, bin_segments in bins.items():
    bin_cnt = len(bin_segments)
    if bin_cnt > most_freq_cnt:
      most_freq_bin = bin_start
      most_freq_cnt = bin_cnt

  # Balance segments.
  for bin_start, bin_segments in bins.items():
    orig_segments = list(bin_segments)
    if len(orig_segments) > 0:
      mod = most_freq_cnt / len(bin_segments)
      while mod >= 2:
        segments.extend(orig_segments)
        mod -= 1
      if mod > 1:
        prob = mod - 1
        sampled = random.sample(orig_segments, int(prob * len(orig_segments)))
        segments.extend(sampled)


def get_global_stats(train_segments, valid_segments, test_segments): 
  sums = {}
  cnts = {}
  maxs = {}
  mins = {}
  avgs = {}
  stds = {}
  sdiff = {}
  stats = {}

  segment_bunches = [train_segments, valid_segments, test_segments]

  for bunch in segment_bunches:
    for segment in bunch:
      for k, v in segment.items():
        if k not in IGNORE:
          sums[k] = sums.get(k, 0) + np.sum(v)
          cnts[k] = cnts.get(k, 0) + v.size
          maxs[k] = np.max([maxs[k], np.max(v)]) if k in maxs else np.max(v)
          mins[k] = np.min([mins[k], np.min(v)]) if k in mins else np.min(v)

  for k in sums:
    avgs[k] = sums[k] / cnts[k]
  
  for bunch in segment_bunches:
    for segment in bunch:
      for k, v in segment.items():
        if k not in IGNORE:
          sdiff[k] = sdiff.get(k, 0) + ((v.flatten() - avgs[k]) ** 2).sum()

  for k in sums:
    stds[k] = np.sqrt(sdiff[k] / (cnts[k] - 1))

  for k in sums:
    stats[f'{k}_min'] = mins[k]
    stats[f'{k}_min'] = mins[k]
    stats[f'{k}_max'] = maxs[k]
    stats[f'{k}_std'] = stds[k]
    stats[f'{k}_avg'] = avgs[k]

  return stats


def dump_segments(path, segments): 
  with open(path, 'wb') as f:
    pickle.dump(segments, f)


def add_baseline_segments(dataset_segments, record_name, rhc_df):
  # Subset RHC dataframe for the given record.
  rhc_subdf = rhc_df[rhc_df['Study ID'].str.strip() == record_name.replace('-', '.')]

  # Get record start and end times.
  start_time, end_time = get_start_and_end_times(record_name)

  # Get static patient attributes.
  metapath = os.path.join(DATABASE_PATH, 'processed_data', f'{record_name}.json')
  with open(metapath, 'r') as f:
    data = json.load(f)
    height = data['height']
    weight = data['weight']
    is_challenge = bool(data['IsChallenge'])

  # Load WFDB record.
  record = wfdb.rdrecord(os.path.join(DATABASE_PATH, 'processed_data', record_name))

  # Get baseline RHC val.
  base_df = rhc_subdf[rhc_subdf['RHC Phase'] == 'Baseline']
  if base_df.shape[0] > 1:
    raise ValueError(f'Multiple baselines found for record {record_name}')
  base_rhc_val = base_df.iloc[0][RHC_TYPE]

  # Get ACC signals.
  acc_indexes = [record.sig_name.index(x) for x in ACC_CHANNELS]
  acc = record.p_signal[:, acc_indexes]
  acc = bandpass_all_channels(acc, 1, 40, SAMPLE_RATE)
  acc = acc.T

  # Get ECG signals.
  ecg_indexes = [record.sig_name.index(x) for x in ECG_CHANNELS]
  ecg = record.p_signal[:, ecg_indexes]
  ecg = bandpass_all_channels(ecg, 0.5, 40, SAMPLE_RATE)
  ecg = ecg.T

  # Get last chamber time if needed.
  if not is_challenge:
    record_end = len(acc)
  else:
    record_end = get_sample_index(start_time, get_last_chamber_time(record_name))

  # Check if both baseline and challenge RHC values are float. If baseline
  # value is not valid, then skip over entire record.
  try:
    base_rhc_val = float(base_rhc_val)
  except:
    print(f'Warning: {record_name} has invalid baseline {RHC_TYPE}')
    return

  # Iterate over baseline and challenges in record, if challenge exists.
  segment_start = 0
  while segment_start < record_end:
    segment_end = int(segment_start + NUM_STEPS)
    if segment_end < record_end:
      if segment_end < record_end:
        acc_segment = acc[segment_start:segment_end,]
        ecg_segment = ecg[segment_start:segment_end,]
        segment = {
          'acc': acc_segment,
          'ecg': ecg_segment,
          'height': np.array([height]),
          'weight': np.array([weight]),
          'bsa': np.array([math.sqrt(height * weight)/3600]),
          'record_name': record_name,
          'start': segment_start,
          'end': segment_end,
          RHC_TYPE: np.array([base_rhc_val])
        }
        dataset_segments.append(segment)
        segment_start += NUM_STEPS // 10
    else:
      return


def get_dataset_baseline_segments(record_names, rhc_df): 
  dataset_segments = []
  for i, record_name in enumerate(record_names):
    add_baseline_segments(dataset_segments, record_name, rhc_df)
  balance_segments(dataset_segments)
  return dataset_segments


def get_record_names_unique(record_names):
  patient_codes = {}
  for record_name in record_names:
    parts = record_name.split('-')
    code = parts[0]
    patient_codes.setdefault(code, []).append(record_name)
  
  unique = []
  for sublist in patient_codes.values():
    if len(sublist) == 1:
      unique.append(sublist[0])

  return unique


def get_average_recording_duration(record_names):
  durations = []
  for record_name in record_names:
    record = wfdb.rdrecord(os.path.join(DATABASE_PATH, 'processed_data', record_name))
    acc_indexes = [record.sig_name.index(x) for x in ACC_CHANNELS]
    acc = record.p_signal[:, acc_indexes]
    duration = len(acc) / SAMPLE_RATE
    durations.append(duration)
  avg = statistics.mean(durations)
  std = statistics.stdev(durations)
  return avg, std


def save_hemo_dataset(dataset_name):
  os.makedirs(os.path.join('datasets', dataset_name), exist_ok=True)

  rhc_path = os.path.join(DATABASE_PATH, 'meta_information', 'RHC_values.csv')
  rhc_df = pd.read_csv(rhc_path)

  all_record_names = get_record_names(DATABASE_PATH)
  unq_record_names = get_record_names_unique(all_record_names)

  print('Unique:', len(unq_record_names))

  duravg, durstd = get_average_recording_duration(all_record_names)
  print(f'Duration: {duravg}, {durstd}')

  random.shuffle(unq_record_names)

  test_valid_pairs = []
  if TEST_RECORD_NAME is None and VALID_RECORD_NAME is None:
    print('WITHOUT prespecified valid record')
    for i in range(0, len(unq_record_names), 2):
      record_name = unq_record_names[i]
      next_idx = i + 1
      if next_idx == len(unq_record_names):
        next_idx = 0
      next_name = unq_record_names[next_idx]
      test_valid_pairs.append((record_name, next_name))
  
  elif TEST_RECORD_NAME is None and VALID_RECORD_NAME is not None:
    print('WITH prespecified valid record')
    for i in range(len(unq_record_names)):
      record_name = unq_record_names[i]
      if record_name != VALID_RECORD_NAME:
        test_valid_pairs.append((record_name, VALID_RECORD_NAME))
  
  elif TEST_RECORD_NAME is not None and VALID_RECORD_NAME is not None:
    print('WITH prespecified test/valid pair')
    test_valid_pairs.append((TEST_RECORD_NAME, VALID_RECORD_NAME))

  print(f'Start record index: {START_RECORD_INDEX}')
  print('Test/valid pairs:', len(test_valid_pairs))

  for i, (test_record_name, valid_record_name) in enumerate(test_valid_pairs, start=START_RECORD_INDEX):
    print(i, test_record_name, '|', valid_record_name)

    train_record_names = list(set(all_record_names) - set([test_record_name, valid_record_name]))
    random.shuffle(train_record_names)

    train_path = os.path.join('datasets', dataset_name, f'train_segments_{i}.pkl')
    valid_path = os.path.join('datasets', dataset_name, f'valid_segments_{i}.pkl')
    test_path = os.path.join('datasets', dataset_name, f'test_segments_{i}.pkl')

    stats_path = os.path.join('datasets', dataset_name, f'global_stats_{i}.json')

    if os.path.exists(train_path): os.remove(train_path)
    if os.path.exists(valid_path): os.remove(valid_path)
    if os.path.exists(test_path): os.remove(test_path)

    if os.path.exists(stats_path): os.remove(stats_path)

    valid_segments = get_dataset_baseline_segments([valid_record_name], rhc_df)
    dump_segments(valid_path, valid_segments)

    test_segments = get_dataset_baseline_segments([test_record_name], rhc_df)
    dump_segments(test_path, test_segments)

    train_segments = get_dataset_baseline_segments(train_record_names, rhc_df)
    dump_segments(train_path, train_segments)

    global_stats = get_global_stats(train_segments, valid_segments, test_segments)
    with open(stats_path, 'w') as f:
      json_str = json.dumps(global_stats, indent=4)
      f.write(json_str)

    log_path = os.path.join('datasets', dataset_name, f'segment_info_{i}.json')
    with open(log_path, 'w') as f:
      json_str = json.dumps({
        'train_records': train_record_names,
        'test_records': test_record_name,
        'valid_records': valid_record_name,
        'train_segments': len(train_segments),
        'valid_segments': len(valid_segments),
        'test_segments': len(test_segments),
      }, indent=4)
      f.write(json_str)


if __name__ == '__main__':
  dataset_name = sys.argv[1]
  save_hemo_dataset(dataset_name)
