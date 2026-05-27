import os
import glob
import torch
import pandas
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
from pathlib import Path


def find_files_with_extension(directory, extension):
    """
    Discover all files with a certain file extension in a directory.

    Parameters:
    directory (str): The directory in which to search for files.
    extension (str): The file extension to search for (e.g., '.txt').

    Returns:
    list: A list of tuples, each containing the filename and its absolute path.
    """
    # Ensure the extension starts with a dot
    if not extension.startswith('.'):
        extension = '.' + extension
    
    # Use glob to find all files with the given extension
    search_pattern = os.path.join(directory, f'*{extension}')
    files = glob.glob(search_pattern)
    
    # Create a list of tuples containing the filename and absolute path
    file_info = [(os.path.basename(file), os.path.abspath(file)) for file in files]
    
    return file_info

class TrafficSignDetectionDataset(torch.utils.data.Dataset):
    
    def __init__(self, groundtruth_path, root_dir, transforms=transforms.Compose([transforms.ToTensor()])):
        self.gt = []

        with open(groundtruth_path) as f:
            for line in f:
                parts = line.strip().split(';')
                filename = parts[0]
                bbox = list(map(int, parts[1:5]))
                label = int(parts[5])
                self.gt.append((filename, bbox, label))

        self.unique_filepaths = find_files_with_extension(root_dir,  ".ppm")
        self.targets_per_file = {}

        self.transforms = transforms

        for filename, bb, class_id in self.gt:
                
            if filename not in self.targets_per_file:
                self.targets_per_file[filename] = []
                
            self.targets_per_file[filename].append({"bbox": bb, "label": class_id})
            
        pass

    def __len__(self):
        return len(self.unique_filepaths)
    
    def __getitem__(self, idx):
        filename, abs_path = self.unique_filepaths[idx]
        image = Image.open(abs_path).convert('RGB')
        
        image_tensor = self.transforms(image) 

        if filename not in self.targets_per_file:
            targets = {
                            'boxes': torch.empty((0, 4), dtype=torch.float32),
                            'labels': torch.empty((0), dtype=torch.int64)
                        }
        else:
            target_for_file = self.targets_per_file[filename]

            boxes = [t['bbox'] for t in target_for_file]
            labels = [t['label'] for t in target_for_file]
            boxes = torch.tensor(boxes, dtype=torch.float32) 
            labels = torch.tensor(labels, dtype=torch.int64)

            targets = {
                'boxes': boxes,
                'labels': labels
            }

        return image_tensor, targets
            

if __name__ == "__main__": 
    root_dir = Path("/home/loma-razer-1/Documents/datasets/FullIJCNN2013/")
    gt_path = Path("/home/loma-razer-1/Documents/datasets/FullIJCNN2013/gt.txt")
    detection_dataset = TrafficSignDetectionDataset(gt_path, root_dir)

    print("Dataset length:", len(detection_dataset))

    image, target = detection_dataset[1]

    print("Image shape:", image.shape)
    print("Boxes:", target['boxes'])
    print("Labels:", target['labels'])

