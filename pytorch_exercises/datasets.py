import torch
import pandas
from PIL import Image
import torchvision.transforms as transforms
import numpy as np

class TrafficSignDataset(torch.utils.data.Dataset):

    def __init__(self, csv_path, root_dir, transforms=transforms.Compose([transforms.ToTensor()])):
        self.data_frame = pandas.read_csv(csv_path)
        self.root_dir = root_dir
        self.transforms = transforms

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()

        path = self.data_frame['Path'][index]
        img_path = self.root_dir / path
        label = torch.Tensor([self.data_frame['ClassId'][index]]).long()

        image = Image.open(img_path)

        if self.transforms:
            image = self.transforms(image)
        
        sample = (image, label)
        return sample
