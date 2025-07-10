import glob
import os
import subprocess
import sys

def get_segment_info_files(dataset):
  path = os.path.join('datasets', dataset, 'segment_info_*.json')
  return [os.path.basename(filepath) for filepath in glob.glob(path)]


def get_model_nums(dataset):
  info_files = get_segment_info_files(dataset)
  model_nums = [int(s.strip('.json').split('_')[2]) for s in info_files]
  return model_nums


if __name__ == '__main__':
  
  # For CI < 2.2:
  # 
  # model_names = [
  #   'hemo_model_CO_TRM276_TRM272',
  #   'hemo_model_CO_TRM273_TRM152',
  #   'hemo_model_CO_TRM254_TRM231',
  #   'hemo_model_CO_TRM223_TRM222',
  #   'hemo_model_CO_TRM186_TRM164',
  #   'hemo_model_CO_TRM198_TRM152',
  #   'hemo_model_CO_TRM184_TRM254',
  #   'hemo_model_CO_TRM270_TRM272',
  #   'hemo_model_CO_TRM274_TRM179',
  #   'hemo_model_CO_TRM163_TRM175',
  #   'hemo_model_CO_TRM203_TRM255',
  #   'hemo_model_CO_TRM243_TRM235',
  #   'hemo_model_CO_TRM248_TRM251',
  #   'hemo_model_CO_TRM215_TRM179',
  #   'hemo_model_CO_TRM268_TRM272',
  #   'hemo_model_CO_TRM187_TRM274',
  #   'hemo_model_CO_TRM128_TRM248',
  #   'hemo_model_CO_TRM178_TRM196',
  #   'hemo_model_CO_TRM181_TRM273',
  # ]
  # dataset_names = [
  #   'hemo_data_CO_TRM276_TRM272',
  #   'hemo_data_CO_TRM273_TRM152',
  #   'hemo_data_CO_TRM254_TRM231',
  #   'hemo_data_CO_TRM223_TRM222',
  #   'hemo_data_CO_TRM186_TRM164',
  #   'hemo_data_CO_TRM198_TRM152',
  #   'hemo_data_CO_TRM184_TRM254',
  #   'hemo_data_CO_TRM270_TRM272',
  #   'hemo_data_CO_TRM274_TRM179',
  #   'hemo_data_CO_TRM163_TRM175',
  #   'hemo_data_CO_TRM203_TRM255',
  #   'hemo_data_CO_TRM243_TRM235',
  #   'hemo_data_CO_TRM248_TRM251',
  #   'hemo_data_CO_TRM215_TRM179',
  #   'hemo_data_CO_TRM268_TRM272',
  #   'hemo_data_CO_TRM187_TRM274',
  #   'hemo_data_CO_TRM128_TRM248',
  #   'hemo_data_CO_TRM178_TRM196',
  #   'hemo_data_CO_TRM181_TRM273',
  # ]
  # model_nums = [
  #   11,
  #   12,
  #   13,
  #   14,
  #   15,
  #   16,
  #   17,
  #   18,
  #   19,
  #   20,
  #   21,
  #   22,
  #   23,
  #   24,
  #   25,
  #   26,
  #   36,
  #   37,
  #   38,
  # ]

  # For CI >= 2.2 and CO < 6:
  # 
  # model_names = [
  #   'hemo_model_CO_TRM148_TRM153',
  #   'hemo_model_CO_TRM160_TRM281',
  #   'hemo_model_CO_TRM197_TRM199',
  #   'hemo_model_CO_TRM208_TRM245',
  #   'hemo_model_CO_TRM219_TRM244',
  #   'hemo_model_CO_TRM252_TRM265',
  #   'hemo_model_CO_TRM253_TRM256',
  #   'hemo_model_CO_TRM260_TRM262',
  #   'hemo_model_CO_TRM275_TRM279',
  #   'hemo_model_CO_TRM174_TRM176',
  #   'hemo_model_CO_TRM233_TRM265',
  # ]
  # dataset_names = [
  #   'hemo_data_CO_TRM148_TRM153',
  #   'hemo_data_CO_TRM160_TRM281',
  #   'hemo_data_CO_TRM197_TRM199',
  #   'hemo_data_CO_TRM208_TRM245',
  #   'hemo_data_CO_TRM219_TRM244',
  #   'hemo_data_CO_TRM252_TRM265',
  #   'hemo_data_CO_TRM253_TRM256',
  #   'hemo_data_CO_TRM260_TRM262',
  #   'hemo_data_CO_TRM275_TRM279',
  #   'hemo_data_CO_TRM174_TRM176',
  #   'hemo_data_CO_TRM233_TRM265',
  # ]
  # model_nums = [
  #   27,
  #   28,
  #   29,
  #   30,
  #   31,
  #   32,
  #   33,
  #   34,
  #   35,
  #   39,
  #   40,
  # ]

  # For CO >= 6:
  # 
  # model_names = [
  #   'hemo_model_CO_TRM107_TRM220',
  #   'hemo_model_CO_TRM234_TRM220',
  #   'hemo_model_CO_TRM127_TRM201',
  #   'hemo_model_CO_TRM172_TRM267',
  #   'hemo_model_CO_TRM282_TRM277',
  #   'hemo_model_CO_TRM155_TRM267',
  # ]
  # dataset_names = [
  #   'hemo_data_CO_TRM107_TRM220',
  #   'hemo_data_CO_TRM234_TRM220',
  #   'hemo_data_CO_TRM127_TRM201',
  #   'hemo_data_CO_TRM172_TRM267',
  #   'hemo_data_CO_TRM282_TRM277',
  #   'hemo_data_CO_TRM155_TRM267',
  # ]
  # model_nums = [
  #   41,
  #   42,
  #   43,
  #   44,
  #   45,
  #   46,
  # ]
   
  for model_name, dataset_name, model_num in zip(model_names, dataset_names, model_nums):
    nohup_name = f'train_log_{model_num}.out'
    print(model_name, dataset_name, model_num)
    with open(nohup_name, 'w') as f:
      subprocess.run(['python', '-u', 'hemo_train.py', model_name, dataset_name, str(model_num)], stdout=f, stderr=subprocess.STDOUT)

