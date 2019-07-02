import os
import torch
import torch.nn as nn
import torch.optim as optim

# Torch-specific
use_gpu = torch.cuda.is_available()
device = torch.device('cuda' if use_gpu else 'cpu')
num_workers = 0
print_freq = 20
load_model_path = 'checkpoints/model.pth'

# Hyper-parameters
batch_size = 1
n_epochs = 2
lr = 0.01

# Architecture-specific
from models import UNet
model_name = 'UNet'
n_classes = 3
model = UNet(n_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

# Data-specific
project_root = os.getcwd()
dataset_root = os.path.join(project_root, 'datasets', 'kidney')
results_dir = os.path.join(project_root, 'results')

from data import kidney
train_loader = kidney.train_loader
val_loader = kidney.val_loader
test_loader = kidney.val_loader
