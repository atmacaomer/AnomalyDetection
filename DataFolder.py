import os
from torchvision.datasets.folder import default_loader
import torch


class CustomTrainDataFolder:
    def __init__(self, root, transform=None):
        self.samples = [os.path.join(root, f) for f in os.listdir(root)]
        self.label = 0
        self.transform = transform
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img = default_loader(self.samples[idx])
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(self.label)
    
class CustomTestDataFolder:
    def __init__(self, root, transform=None, normal_dir:str="NormalVideos"):
        i = 0
        self.samples = []
        self.normal_dir = normal_dir
        self.transform = transform
        for sub_folder in os.listdir(root):
            sub_folder_path = os.path.join(root, sub_folder)
            label = 0 if sub_folder == self.normal_dir else 1
            for file in os.listdir(sub_folder_path):
                file_path = os.path.join(sub_folder_path,file)
                self.samples.append((file_path,label))
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self,idx):
        path, label = self.samples[idx]
        img = default_loader(path)
        if self.transform:
            img = self.transform(img)
        return img, label        

    
