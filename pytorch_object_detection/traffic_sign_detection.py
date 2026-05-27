from datasets import TrafficSignDetectionDataset
from model_trainer_detection import ModelTrainerDetection
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from pathlib import Path

# load a model pre-trained on COCO
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")

# replace the classifier with a new one, that has
# num_classes which is user-defined
num_classes = 43  # 1 class (person) + background
# get number of input features for the classifier
in_features = model.roi_heads.box_predictor.cls_score.in_features
# replace the pre-trained head with a new one
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

root_dir = Path("/home/loma-razer-1/Documents/datasets/FullIJCNN2013")
gt_path = Path("/home/loma-razer-1/Documents/datasets/FullIJCNN2013/gt.txt")

whole_dataset = TrafficSignDetectionDataset(gt_path, root_dir) 
train_dataset, val_dataset = torch.utils.data.random_split(whole_dataset, [0.8, 0.2], torch.Generator().manual_seed(42))

trainer = ModelTrainerDetection(model, lr=1e-4) 
trainer.train(train_dataset, batch_size=8, num_epochs=20)
preds = trainer.evaluate(val_dataset, batch_size=8)
trainer.visualize_predictions(val_dataset, preds)
