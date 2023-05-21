import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torchvision
import torch.optim as optim
from data import get_data
from torchvision.models import ResNet50_Weights
from main import train

model = torchvision.models.resnet50(weights=ResNet50_Weights)
model.fc = nn.Linear(in_features=2048, out_features=8, bias=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
train_ds, test_ds = get_data()

train_loader = DataLoader(train_ds, batch_size=32)
test_loader = DataLoader(train_ds, batch_size=32)

model = model.to(device)
epochs = 30

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(),
                      lr=0.03)

accs, losses = train(model, epochs, train_loader, test_loader, criterion, optimizer, device)
torch.save(model.state_dict(), "./models/emotion_detection_model_resnet.pt")
