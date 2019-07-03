import os
import torch
import torch.nn.functional as F
import torch.optim as optim

# Torch-specific
use_gpu = torch.cuda.is_available()
device = torch.device('cuda' if use_gpu else 'cpu')
num_workers = 4
print_freq = 20

# Hyper-parameters
batch_size = 1
n_epochs = 2
lr = 0.01

# Architecture-specific
from models import UNet
model_name = 'UNet'
n_classes = 2
model = UNet(n_classes)
criterion = F.cross_entropy
optimizer = optim.Adam(model.parameters())

# Data-specific
# TODO: Make project_root platform agnostic
project_root = "E:\\project_russel\\unet"
dataset_root = os.path.join(project_root, 'datasets', 'kidney')
model_path = os.path.join(project_root, 'checkpoints', 'model.pth')

from data import kidney
train_loader = kidney.train_loader
val_loader = kidney.val_loader
test_loader = kidney.val_loader
