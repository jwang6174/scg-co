import json
import numpy as np
import os
import pickle
import random
import sys
import wfdb
from datetime import datetime
from pathlib import Path
from pathutil import DATABASE_PATH
from sklearn.model_selection import train_test_split
from noiseutil import has_noise

SAMPLE_RATE = 500


def get_record_names(db_path): 
  names = set()
  for filename in os.listdir(os.path.join(db_path, 'processed_data')):
    if filename.endswith('.dat') or filename.endswith('.hea'):
      names.add(Path(filename).stem)
  names = sorted(list(names))
  return names


def get_sample_index(start_time, sample_time, sample_rate): 
  diff = (sample_time - start_time).total_seconds()
  sample_index = int(diff * sample_rate)
  return sample_index


def get_PA_challenge_interval(record_name, macStTime, db_path, sample_rate): 
  filepath = os.path.join(db_path, 'meta_information', 'PAM_PCWP_timestamp_in_TBME.json')
  with open(filepath, 'r') as f:
    data = json.load(f)
    if record_name in data:
      try:
        start_str = data[record_name]['PA_VI']['datetime'].strip()
        start_obj = datetime.strptime(start_str, '%H:%M:%S')
        start_idx = get_sample_index(macStTime, start_obj, sample_rate)
        end_str = data[record_name]['PCW_VI']['datetime'].strip()
        end_obj = datetime.strptime(end_str, '%H:%M:%S')
        end_idx = get_sample_index(macStTime, end_obj, sample_rate)
        return start_idx, end_idx
      except:
        return None


def get_chamber_intervals(record_name, chamber, db_path, sample_rate): 
  intervals = []
  with open(os.path.join(db_path, 'processed_data', f'{record_name}.json'), 'r') as f:
    
    data = json.load(f)
    macStTime = datetime.strptime(data['MacStTime'].split()[1], '%H:%M:%S')
    macEndTime = datetime.strptime(data['MacEndTime'].split()[1], '%H:%M:%S')
    chamEvents = data['ChamEvents_in_s']
    if isinstance(chamEvents, dict):
      chamEvents['END'] = (macEndTime - macStTime).total_seconds()
      chamEvents = sorted(chamEvents.items(), key=lambda x: x[1])
      for i, event in enumerate(chamEvents[:-1]):
        if event[0].split('_')[0] == chamber:
          intervals.append((int(event[1] * sample_rate), 
                            int(chamEvents[i+1][1] * sample_rate)))
    
  if chamber == 'PA':
    PA_challenge_interval = get_PA_challenge_interval(record_name, macStTime, 
                                                      db_path, sample_rate)
    if PA_challenge_interval is not None:
      intervals.append(PA_challenge_interval)

  return intervals


def get_channels(record_name, channel_names, start, stop, db_path):    
  record = wfdb.rdrecord(os.path.join(db_path, 'processed_data', record_name))                              
  indexes = [record.sig_name.index(name) for name in channel_names]                                
  channels = record.p_signal[start:stop, indexes]
  return channels


def get_patient_code(record_name): 
  code = record_name.split('-')[0]
  return code


def get_record_names_with_challenge(db_path): 
  with_challenge = []
  for name in get_record_names(db_path):
    with open(os.path.join(db_path, 'processed_data', f'{name}.json'), 'r') as f:
      data = json.load(f)
      if data['IsChallenge'] == 1:
        with_challenge.append(name)
  return with_challenge


def get_record_names_without_challenge(db_path): 
  all_names = get_record_names(db_path)
  with_challenge = get_record_names_with_challenge(db_path)
  no_challenge = sorted(list(set(all_names) - set(with_challenge)))
  return no_challenge


def get_record_names_of_multiple_caths(db_path): 
  multi = []
  counts = {}
  record_names = get_record_names(db_path)
  
  for name in record_names:
    code = get_patient_code(name)
    counts[code] = counts.get(code, 0) + 1

  for name in record_names:
    code = get_patient_code(name)
    if counts[code] > 1:
      multi.append(name)

  return multi


def get_record_names_of_single_caths(db_path):
  single = []
  all_names = get_record_names(db_path)
  multi_names = get_record_names_of_multiple_caths(db_path)
  single = sorted(list(set(all_names) - set(multi_names)))
  return single


def get_groups(list_to_split, n): 
  groups = []
  random.shuffle(list_to_split)
  for i in range(0, len(list_to_split), n):
    group = list_to_split[i:i+n]
    if len(group) == n:
      groups.append(group)
  return groups


def get_global_stats(segments, signal_names): 
  joined = {}
  for segment in segments:
    for signal_name in signal_names:
      signal = segment.get(signal_name)
      if signal is not None:
        joined[signal_name] = signal
      else:
        joined[signal_name] = np.hstack((joined[signal_name], signal))

  stats = {}
  for signal_name, joined_segment in joined.items():
    stats[f'{signal_name}_avg'] = np.mean(joined_segment)
    stats[f'{signal_name}_std'] = np.std(joined_segment)
    stats[f'{signal_name}_max'] = np.max(joined_segment)
    stats[f'{signal_name}_min'] = np.min(joined_segment)

  return stats


def add_local_stats(segments, signal_names): 
  for segment in segments:
    for name in signal_names:
      signal = segment[name]
      min = np.min(signal)
      max = np.max(signal)
      segment[f'{name}_min'] = min
      segment[f'{name}_max'] = max


def get_test_record_names(db_path): 
  set1 = set(get_record_names_without_challenge(db_path))
  set2 = set(get_record_names_of_single_caths(db_path))
  record_names = list(set1.intersection(set2))
  return record_names


def get_train_record_names(test_record_names, db_path): 
  record_names = list(set(get_record_names(db_path)) - set(test_record_names))
  return record_names


def get_record_segments(record_name, acc_channels, chamber, segment_size, 
                        flat_amp_threshold, flat_min_duration, straight_threshold, 
                        min_RHC, max_RHC, db_path, sample_rate): 
  record_segments = []
  segment_size *= sample_rate
  for chamber_start, chamber_end in get_chamber_intervals(record_name, chamber, 
                                                          db_path, sample_rate):
    acc = get_channels(record_name, acc_channels, chamber_start, chamber_end, db_path)
    rhc = get_channels(record_name, ['RHC_pressure'], chamber_start, chamber_end, db_path)
    num_segments = int(acc.shape[0] // segment_size)
    for i in range(num_segments):
      start_idx = int(i * segment_size)
      end_idx = int(start_idx + segment_size)
      acc_seg = acc[start_idx:end_idx]
      rhc_seg = rhc[start_idx:end_idx]
      if not has_noise(rhc_seg[:, 0], flat_amp_threshold, flat_min_duration, 
                       straight_threshold, min_RHC, max_RHC, sample_rate):
        segment = {
          'acc': acc_seg,
          'rhc': rhc_seg,
          'record_name': record_name,
          'start_idx': start_idx,
          'end_idx': end_idx,
        }
        record_segments.append(segment)
  return record_segments


def get_dataset_segments(record_names, acc_channels, chamber, segment_size, 
                         flat_amp_threshold, flat_min_duration, straight_threshold, 
                         min_RHC, max_RHC, db_path, sample_rate): 
  dataset_segments = []
  for record_name in record_names:
    record_segments = get_record_segments(record_name, acc_channels, chamber, 
                                          segment_size, flat_amp_threshold, 
                                          flat_min_duration, straight_threshold, 
                                          min_RHC, max_RHC, db_path, sample_rate)
    dataset_segments.extend(record_segments)
  return dataset_segments


def save_wave_dataset(dataset_name, acc_channels, chamber, segment_size,
                      num_tests, num_folds, flat_amp_threshold, flat_min_duration, 
                      straight_threshold, min_RHC, max_RHC, db_path, sample_rate): 
  print(f'Run wave_record.py for {dataset_name}')

  # Define signal names.
  signal_names = ['acc', 'rhc']

  # Get names of all records that may be included in a test set.
  all_test_records = get_test_record_names(db_path)

  # Randomly split test records into groups of a given length.
  all_groups = get_groups(all_test_records, num_tests)

  # Use first couple of batches for test set.
  test_record_groups = all_groups[:num_folds]

  # Create different hold out sets.
  for i, test_records in enumerate(test_record_groups):
    print(f'Create fold {i+1}/{num_folds}')

    # Identify training records by subtracting test records from all records.
    train_records = get_train_record_names(test_records, db_path)

    # Take a certain number of train records for the valid set.
    valid_records = train_records[:num_tests]
    train_records = train_records[num_tests:]

    # Get train segments.
    train_segments = get_dataset_segments(train_records, acc_channels, chamber, 
                                          segment_size, flat_amp_threshold, 
                                          flat_min_duration, straight_threshold, 
                                          min_RHC, max_RHC, db_path, sample_rate)

    # Get valid segments.
    valid_segments = get_dataset_segments(valid_records, acc_channels, chamber, 
                                          segment_size, flat_amp_threshold, 
                                          flat_min_duration, straight_threshold, 
                                          min_RHC, max_RHC, db_path, sample_rate)

    # Get test segments.
    test_segments = get_dataset_segments(test_records, acc_channels, chamber, 
                                         segment_size, flat_amp_threshold, 
                                         flat_min_duration, straight_threshold, 
                                         min_RHC, max_RHC, db_path, sample_rate)

    # Calculate global stats for each signal across all segments and save 
    # to file. To be used for feature normalization during training.
    all_segments = train_segments + valid_segments + test_segments
    global_stats = get_global_stats(all_segments, signal_names)
    global_stats_path = os.path.join('datasets', dataset_name, f'global_stats_{i+1}.json')
    with open(global_stats_path, 'w') as f:
      json.dump(global_stats, f, indent=2)

    # Add local stats to all segments.
    add_local_stats(all_segments, signal_names)
  
    # Save train segments.
    train_path = os.path.join('datasets', dataset_name, f'train_segments_{i+1}.pkl')
    with open(train_path, 'wb') as f:
      pickle.dump(train_segments, f)

    # Save valid segments.
    valid_path = os.path.join('datasets', dataset_name, f'valid_segments_{i+1}.pkl')
    with open(valid_path, 'wb') as f:
      pickle.dump(valid_segments, f)

    # Save test segments.
    test_path = os.path.join('datasets', dataset_name, f'test_segments_{i+1}.pkl')
    with open(test_path, 'wb') as f:
      pickle.dump(test_segments, f)

    # Save segment info.
    log_path = os.path.join('datasets', dataset_name, f'segment_info_{i+1}.json')
    with open(log_path, 'w') as f:
      json_str = json.dumps({
        'train_records': train_records,
        'test_records': test_records,
        'valid_records': valid_records,
        'train_segments': len(train_segments),
        'valid_segments': len(valid_segments),
        'test_segments': len(test_segments),
      }, indent=4)
      f.write(json_str)


def run(dataset_name): 
  dataset_params_path = os.path.join('datasets', dataset_name, 'params.json')
  with open(dataset_params_path, 'r') as f:
    params = json.load(f)
    save_wave_dataset(
      dataset_name,
      params['acc_channels'],
      params['chamber'],
      params['segment_size'],
      params['num_tests'],
      params['num_folds'],
      params['flat_amp_threshold'],
      params['flat_min_duration'],
      params['straight_threshold'],
      params['min_RHC'],
      params['max_RHC'],
      DATABASE_PATH,
      SAMPLE_RATE
    )


if __name__ == '__main__':
  dataset_name = sys.argv[1]
  run(dataset_name)
