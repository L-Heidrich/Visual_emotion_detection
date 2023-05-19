import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torchvision
import torch.optim as optim
import torch.nn.functional as F
from data import get_data
from torchvision.models import resnet50, ResNet50_Weights

model = torchvision.models.resnet50(weights=ResNet50_Weights)
model.fc = nn.Linear(in_features=2048, out_features=8, bias=True)
