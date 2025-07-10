import os
import shutil

DATABASE_PATH = os.path.join('/', 'home', 'jesse', 'scg-rhc-db')


def clear_dir(paths):
  for path in paths:
    if os.path.exists(path):
      shutil.rmtree(path)
      os.makedirs(path)
      print(f'Cleared {path}')
