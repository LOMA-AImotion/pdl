
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from pathlib import Path
import time
import pandas
from PIL import Image
import numpy as np

from model_trainer import ModelTrainer
from datasets import TrafficSignDataset
from models import FullyConnectedNet, SmallConvNet

### Hyperparameters ###
batch_size = 64
num_epochs = 4
lr = 0.0001
######################

data_root_path = Path("TODO")
train_path = data_root_path / "Train"
test_path = data_root_path / "Test"

# for fully connected 
transforms = transforms.Compose([
    transforms.Resize((45, 45)),
    transforms.ToTensor()
])

# for small CNN
# transforms = transforms.Compose([
#     transforms.Resize((44, 44)),
#     transforms.ToTensor()
# ])


train_set = TrafficSignDataset(data_root_path / "Train.csv", data_root_path, transforms=transforms)
train_dataloader = torch.utils.data.DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)

test_set = TrafficSignDataset(data_root_path / "Test.csv", data_root_path, transforms=transforms)
test_dataloader = torch.utils.data.DataLoader(dataset=test_set, batch_size=batch_size, shuffle=True)

model = FullyConnectedNet()
# model = SmallConvNet()

optimizer = optim.Adam(model.parameters(), lr=lr)
trainer = ModelTrainer(model, train_dataloader, loss_criterion=nn.CrossEntropyLoss(), optimizer=optimizer, learning_rate=lr, num_epochs=num_epochs, num_classes=43)
trainer.train(val_dataloader=test_dataloader)
print("")
print("Training finished. Starting evaluation.")
trainer.multiclass_test(test_dataloader)
