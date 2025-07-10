import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


def get_flat_lines(waveform, amp_threshold, min_duration, sample_rate): 
  min_samples = int(min_duration * sample_rate)
  waveform_series = pd.Series(waveform)

  rolling_diff = waveform_series.rolling(window=min_samples).max() - \
                 waveform_series.rolling(window=min_samples).min()

  flat_indices = rolling_diff[rolling_diff < amp_threshold].index
  flat_segments = []
  start = None

  for i in range(len(flat_indices) - 1):
    if start is None:
      start = flat_indices[i]
    if flat_indices[i + 1] != flat_indices[i] + 1:
      flat_segments.append((start, flat_indices[i]))
      start = None
    if start is not None:
      flat_segments.append((start, flat_indices[-1]))

  return flat_segments


def is_straight_line(waveform, threshold): 
  x = np.arange(len(waveform))
  y = np.array(waveform)
  model = LinearRegression().fit(x.reshape(-1, 1), y)
  r_squared = model.score(x.reshape(-1, 1), y)
  is_straight = r_squared > threshold
  return is_straight


def in_rhc_range(waveform, min_RHC, max_RHC): 
  for val in waveform:
    if val < min_RHC or val > max_RHC:
      return False
  return True


def has_noise(waveform, flat_amp_threshold, flat_min_duration, 
              straight_threshold, min_RHC, max_RHC, sample_rate):
  has_noise = (
    len(get_flat_lines(waveform, flat_amp_threshold, 
                       flat_min_duration, sample_rate)) > 0 or
    is_straight_line(waveform, straight_threshold) or
    not in_rhc_range(waveform, min_RHC, max_RHC)
  )
  return has_noise
