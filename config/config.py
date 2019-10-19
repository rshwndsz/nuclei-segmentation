import os
import torch

# Torch-specific
use_gpu = torch.cuda.is_available()
device = torch.device('cuda' if use_gpu else 'cpu')
num_workers = 4

# Train/val-specific
print_freq = 20
val_freq = 1
resume_from_epoch = 0
min_val_loss = 1000

# Hyper-parameters
batch_size = 4
n_epochs = 20
lr = 1e-3

# Data-specific
# `os.getcwd()` doesn't work in notebooks
# project_root = '/home/shyam/myProjects/unet/'
project_root = '/Users/Russel/myProjects/unet/'
dataset_root = os.path.join(project_root, 'datasets', 'kidney')
model_path = os.path.join(project_root, 'checkpoints', 'model.pth')
model_final_path = os.path.join(project_root, 'checkpoints', 'model_final.pth')
results_dir = os.path.join(project_root, 'results')
