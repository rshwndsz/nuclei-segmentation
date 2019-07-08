import torch.nn.functional as F
import torch.optim as optim
from . import config as cfg
from models import UNet


model_name = 'UNet'
n_classes = 2
model = UNet(n_classes)
model = model.to(cfg.device)
criterion = F.cross_entropy
optimizer = optim.Adam(model.parameters())
