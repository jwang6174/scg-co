import json
import numpy as np
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from hemo_loader import get_loader
from hemo_record import RHC_TYPE
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

BATCH_SIZE = 128
NUM_EPOCHS = 20


class Projection(nn.Module):
  
  def __init__(self, embed_dim, num_heads=1):
    super().__init__()
    self.attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
    self.norm = nn.LayerNorm(embed_dim)

  def forward(self, x):
    x = x.permute(0, 2, 1)
    attn_out, _ = self.attn(x, x, x)
    x = self.norm(x + attn_out)
    return x.permute(0, 2, 1)


class FeatureCNN(nn.Module):

  def __init__(self, in_channels):
    super(FeatureCNN, self).__init__()
    self.conv = nn.Sequential(
      nn.Conv1d(in_channels, 16, kernel_size=5, dilation=1, padding=1),
      nn.ReLU(),
      nn.Conv1d(16, 32, kernel_size=5, dilation=4, padding=4),
      nn.ReLU(),
      nn.Conv1d(32, 64, kernel_size=5, dilation=16, padding=16),
      nn.ReLU(),
      nn.Conv1d(64, 128, kernel_size=5, dilation=32, padding=32),
      nn.AdaptiveAvgPool1d(1),
    )
    self.proj = Projection(embed_dim=128, num_heads=1)

  def forward(self, x):
    x = self.conv(x)
    x = self.proj(x)
    return x


class CardiovascularPredictor(nn.Module):

  def __init__(self):
    super(CardiovascularPredictor, self).__init__()
    self.acc_net = FeatureCNN(in_channels=3)
    self.ecg_net = FeatureCNN(in_channels=1)
    self.bmi_mlp = nn.Sequential(
      nn.Linear(1, 8),
      nn.ReLU(),
      nn.Linear(8, 8),
    )
    self.mlp = nn.Sequential(
      nn.Linear(128 + 128 + 8, 128),
      nn.LayerNorm(128),
      nn.ReLU(),
      nn.Dropout(0.3),
      nn.Linear(128, 1)
    )

  def forward(self, acc, ecg, bmi):
    f_acc = self.acc_net(acc).squeeze(-1)
    f_ecg = self.ecg_net(ecg).squeeze(-1)
    f_bmi = self.bmi_mlp(bmi)
    return self.mlp(torch.cat([f_acc, f_ecg, f_bmi], dim=1))


def get_rmse(pred_y, real_y):
  return torch.sqrt(torch.mean((pred_y - real_y) ** 2)).item()


def train(model_name, data_name, data_fold):
  model_dir_path = os.path.join('models', model_name)
  os.makedirs(model_dir_path, exist_ok=True)

  model = CardiovascularPredictor()
  criterion = nn.MSELoss()
  optim = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  model = model.to(device)
  criterion = criterion.to(device)

  model_last_path = os.path.join(model_dir_path, 'model_last.chk')

  if os.path.exists(model_last_path):
    checkpoint = torch.load(model_last_path, weights_only=False)
    epoch = checkpoint['epoch'] + 1
    last_epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['model_state_dict'])
    optim.load_state_dict(checkpoint['optim_state_dict'])
    print(f'Resumed from epoch {epoch}')
  else:
    epoch = 1
    last_epoch = -1

  scheduler = CosineAnnealingWarmRestarts(optim, T_0=10, T_mult=2, last_epoch=last_epoch)

  filepath = os.path.join('datasets', data_name, f'global_stats_{data_fold}.json')
  with open(filepath, 'r') as f:
    global_segment_stats = json.load(f)

  filepath = os.path.join('datasets', data_name, f'valid_segments_{data_fold}.pkl')
  valid_loader = get_loader(filepath, global_segment_stats, False, 1)

  filepath = os.path.join('datasets', data_name, f'test_segments_{data_fold}.pkl')
  if os.path.exists(filepath):
    test_loader = get_loader(filepath, global_segment_stats, False, 1)
  else:
    print('Test filepath does not exist')
    test_loader = []

  filepath = os.path.join('datasets', data_name, f'train_segments_{data_fold}.pkl')
  train_loader = get_loader(filepath, global_segment_stats, True, BATCH_SIZE)

  while epoch <= NUM_EPOCHS:

    model.train()
    train_loss = 0
    for i, item in enumerate(train_loader, start=1):
      
      optim.zero_grad()
      
      acc, ecg, bmi, label, _ = item
      acc = acc.to(device)
      ecg = ecg.to(device)
      bmi = bmi.to(device)
      real_y = label.squeeze(-1).to(device)
      pred_y = model(acc, ecg, bmi)
      loss = criterion(pred_y, real_y)
      loss.backward()
      optim.step()
      train_loss += loss.item()

      if i % 50 == 0:
        print(f'Epoch = {epoch}, '
              f'Batch = {i}/{len(train_loader)}, '
              f'Epoch Loss = {train_loss/i:.5f} ')

    train_loss /= len(train_loader)
    scheduler.step()

    model.eval()
    valid_loss = 0
    test_loss = 0
    with torch.no_grad():
      
      for item in valid_loader:
        acc, ecg, bmi, label, _ = item
        acc = acc.to(device)
        ecg = ecg.to(device)
        bmi = bmi.to(device)
        real_y = label.squeeze(-1).to(device)
        pred_y = model(acc, ecg, bmi)
        valid_loss += get_rmse(pred_y, real_y)
      
      for item in test_loader:
        acc, ecg, bmi, label, _ = item
        acc = acc.to(device)
        ecg = ecg.to(device)
        bmi = bmi.to(device)
        real_y = label.squeeze(-1).to(device)
        pred_y = model(acc, ecg, bmi)
        test_loss += get_rmse(pred_y, real_y)

    valid_loss /= len(valid_loader)
    if len(test_loader) > 0:
      test_loss /= len(test_loader)

    print(f'RHC Type = {RHC_TYPE}')
    print(f'Model = {model_name}')
    print(f'Train Loss = {train_loss}')
    print(f'Valid Loss = {valid_loss}')
    print(f'Test Loss = {test_loss}')
    current_lr = optim.param_groups[0]['lr']
    print(f'LR = {current_lr}')

    checkpoint = {
      'epoch': epoch,
      'model_state_dict': model.state_dict(),
      'optim_state_dict': optim.state_dict(),
      'train_loss': train_loss,
      'valid_loss': valid_loss,
      'test_loss': test_loss
    }
    torch.save(checkpoint, model_last_path)
    torch.save(checkpoint, os.path.join(model_dir_path, f'model_{epoch}.chk'))

    epoch += 1


if __name__ == '__main__':
  model_name = sys.argv[1]
  data_name = sys.argv[2]
  data_fold = sys.argv[3]
  train(model_name, data_name, data_fold)

