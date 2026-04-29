import torch
import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn as nn
from pathlib import Path

from model_trainer import ModelTrainer
from datasets import TrafficSignDataset

### Hyperparameters ###
batch_size = 48
num_epochs = 1
lr = 1e-4
######################

data_root_path = Path("TODO")
train_path = data_root_path / "Train"
test_path = data_root_path / "Test"

# model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
# model = models.resnet152(weights=models.ResNet152_Weights.IMAGENET1K_V2)
# model = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_SWAG_E2E_V1)
# model = models.resnext50_32x4d(weights=models.ResNeXt50_32X4D_Weights.IMAGENET1K_V1)
model = models.resnext101_32x8d(weights=models.ResNeXt101_32X8D_Weights.IMAGENET1K_V1)
print(model)

model.fc = torch.nn.Linear(model.fc.in_features, out_features=43)
# model.heads[-1] = torch.nn.Linear(model.heads[-1].in_features, out_features=43)

# transforms = models.ResNet18_Weights.IMAGENET1K_V1.transforms
# transforms = models.ResNet152_Weights.IMAGENET1K_V2.transforms
# transforms = models.ViT_B_16_Weights.IMAGENET1K_SWAG_E2E_V1.transforms
# transforms = models.ResNeXt50_32X4D_Weights.IMAGENET1K_V1.transforms
transforms = models.ResNeXt101_32X8D_Weights.IMAGENET1K_V1.transforms
transforms = transforms()

train_set = TrafficSignDataset(data_root_path / "Train.csv", data_root_path, transforms=transforms)
train_dataloader = torch.utils.data.DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)

test_set = TrafficSignDataset(data_root_path / "Test.csv", data_root_path, transforms=transforms)
test_dataloader = torch.utils.data.DataLoader(dataset=test_set, batch_size=batch_size, shuffle=True)


trainer = ModelTrainer(model, train_dataloader, loss_criterion=nn.CrossEntropyLoss(), learning_rate=lr, num_epochs=num_epochs, num_classes=43)
trainer.train(val_dataloader=test_dataloader)
print("")
print("Training finished. Starting evaluation.")
trainer.multiclass_test(test_dataloader)
