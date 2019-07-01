# Config file
from torch import nn, cuda, optim
from .models import UNet

# Hyper-parameters
batch_size = 128
max_epoch = 2
lr = 0.01
lr_decay = 0.95
weight_decay = 1e-4

# Architecture
model_name = 'SampleNet'
model = UNet()  # name of the model
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),
                       lr=lr,
                       weight_decay=weight_decay)

# Dataset
train_data_root = 'dataset/train'
test_data_root = 'dataset/test'
val_data_root = 'dataset/kidney/val'
load_model_path = 'checkpoints/model.pth'

# torch specific parameters
use_gpu = cuda.is_available()
num_workers = 41
print_freq = 20     # print info every N batch
