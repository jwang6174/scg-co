import json
import numpy as np
import os
import pandas as pd
import sys
import torch
import torch.nn as nn
from hemo_loader import get_loader
from hemo_record import RHC_TYPE
from hemo_train import CardiovascularPredictor
from pathutil import DATABASE_PATH
from scipy.stats import t, sem

def get_mean_and_ci(vals):
  mean = np.mean(vals)
  h = sem(vals) * t.ppf((1 + 0.95) / 2.0, df=len(vals) - 1)
  ci_lower = mean - h
  ci_upper = mean + h
  return mean, ci_lower, ci_upper


def add_info(rows):
  for row in rows:
    path = os.path.join(DATABASE_PATH, 'processed_data', f"{row['name']}.json")
    with open(path, 'r') as f:
      data = json.load(f)
      row.update(data)


def test(model_name, dataset_name, model_num, data_type):
  model_path = os.path.join('models', 
                           f'{model_name}', 
                           f'model_for_{data_type}.chk')
                          
  checkpoint = torch.load(model_path, weights_only=False)
  model = CardiovascularPredictor()
  model.load_state_dict(checkpoint['model_state_dict'])
  model.eval()

  stats_path = os.path.join('datasets',
                            dataset_name,
                            f'global_stats_{model_num}.json')

  with open(stats_path, 'r') as f:
    global_stats = json.load(f)
  
  loader_path = os.path.join('datasets', 
                              dataset_name, 
                              f'{data_type}_segments_{model_num}.pkl')

  loader = get_loader(loader_path, global_stats, False, 1)
  
  pred = []
  real = None
  name = None
  for i, (acc, ecg, bmi, label, record_name) in enumerate(loader, start=1):

    real_y = label.squeeze(-1).detach().numpy()[0][0]
    pred_y = model(acc, ecg, bmi).detach().numpy()[0][0]
    name = record_name[0]
    real = real_y
    pred.append(pred_y)

  mean, lower_ci, upper_ci = get_mean_and_ci(pred)
  return name, real, mean, lower_ci, upper_ci



if __name__ == '__main__':
  # Example
  # model_names = [
    # 'hemo_model_CO_TRM258_TRM278',
    # 'hemo_model_CO_TRM276_TRM272',
  # ]
  # dataset_names = [
    # 'hemo_data_CO_TRM258_TRM278',
    # 'hemo_data_CO_TRM276_TRM272',
  # ]
  # model_nums = [
    # 10,
    # 11,
  # ]

  rows = []
  for model_name, dataset_name, model_num in zip(model_names, dataset_names, model_nums):
    for data_type in ['test', 'valid']:
      try:
        name, real, mean, lower_ci, upper_ci = test(model_name, dataset_name, model_num, data_type)
        row = {
          'name': name,
          'real': real,
          'pred': mean,
          'lower_ci': lower_ci,
          'upper_ci': upper_ci,
        }
        rows.append(row)
      except Exception as e:
        print(e)
        continue

  add_info(rows)
  df = pd.DataFrame(rows)
  df.to_csv('scg-rhc-ml-results.csv', index=False)
