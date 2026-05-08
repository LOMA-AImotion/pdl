import os
import shutil
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms.functional as F
from torchvision.ops import box_iou
from torchmetrics.detection.iou import IntersectionOverUnion
from tqdm import tqdm


class ModelTrainerDetection:
    def __init__(self, model, lr, device='cuda'):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    def train(self, train_dataset, batch_size, num_epochs=10):
        self.model.train()
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=self.collate_fn) 
        for epoch in range(num_epochs):

            epoch_loss = 0
            for images, targets in tqdm(train_loader):
                images = list(image.to(self.device) for image in images)
                targets = [t for t in targets]

                for d in targets:
                    for k, v in d.items():
                        d[k] = v.to(self.device)

                loss_dict = self.model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                
                self.optimizer.zero_grad()
                losses.backward()
                self.optimizer.step()
                
                epoch_loss += losses.item()
            
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss/len(train_loader)}")
    
    def evaluate(self, val_dataset, batch_size):
        self.model.eval()
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=self.collate_fn)
        
        all_predictions = []
        iou = IntersectionOverUnion()
        with torch.no_grad():
            for images, targets in val_loader:
                images = list(image.to(self.device) for image in images)
                targets = [t for t in targets]

                predictions = self.model(images)
                
                cpu_predictions = [{k: v.cpu() for k, v in pred.items()} for pred in predictions]
                all_predictions.extend(cpu_predictions)
                iou.update(cpu_predictions, targets)

        print("**********************")
        print(f"IoU: \n {iou.compute()["iou"].item()}")
        print("**********************")
                
                
        return all_predictions
    

    def visualize_predictions(self, images, predictions, threshold=0.5, output_folder='predictions'):
        # Ensure the output folder exists and is empty
        if os.path.exists(output_folder):
            shutil.rmtree(output_folder)
        os.makedirs(output_folder, exist_ok=True)
        
        for i in range(len(images)):
            image = images[i][0].permute(1, 2, 0).cpu().numpy()
            image = (image * 255).astype(np.uint8)
            
            boxes = predictions[i]['boxes'].cpu().numpy()
            scores = predictions[i]['scores'].cpu().numpy()
            labels = predictions[i]['labels'].cpu().numpy()

            
            fig, ax = plt.subplots(1, 1, figsize=(12, 9))
            ax.imshow(image)
            
            for box, score, label in zip(boxes, scores, labels):
                if score > threshold:
                    x1, y1, x2, y2 = box
                    rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, fill=False, color='red', linewidth=2)
                    ax.add_patch(rect)
                    ax.text(x1, y1, f'Label: {label} | {score:.2f}', fontsize=12, bbox=dict(facecolor='yellow', alpha=0.5))
            
            # Create a unique name for each image using a timestamp
            filename = f"prediction_{i}.png"
            filepath = os.path.join(output_folder, filename)
            
            plt.savefig(filepath)
            plt.close()

    def collate_fn(self, batch):
        return tuple(zip(*batch))
