import os
import random
from random import shuffle
import numpy as np
import pandas as pd
import torch
from torch.utils import data
from torchvision import transforms as T
from torchvision.transforms import functional as F
from PIL import Image
import PIL

class ImageFolder(data.Dataset):
    def __init__(self, root, labels_csv_file=None, image_size=224,mode='train',augmentation_prob=0.4,apply_transform=True):
        """Initializes image paths and preprocessing module."""
        self.root = root
        self.img_dir = root#list(map(lambda x: os.path.join(root, x), os.listdir(root)))
        self.images = os.listdir(self.img_dir)
        self.image_size = image_size
        self.mode = mode
        self.RotationDegree = [0,90,180,270]
        self.augmentation_prob = augmentation_prob
        self.apply_transform = apply_transform
        if labels_csv_file:
            self.labels_df = pd.read_csv(labels_csv_file)
        else:
            raise ValueError('The label file is not provided!')
        print("image count in {} path :{}".format(self.mode,len(self.images)))

    def __getitem__(self, index):
        """Reads an image from a file and preprocesses it and returns."""
        image_path = os.path.join(self.img_dir,self.images[index])

        image = Image.open(image_path)

        Transform = []
        Transform.append(T.Resize(size=(self.image_size, self.image_size)))
        if self.apply_transform:        
            p_transform = random.random()

            if p_transform <= self.augmentation_prob:
                RotationDegreeSelection = random.randint(0,3)
                RotationDegree = self.RotationDegree[RotationDegreeSelection]
                Transform.append(T.RandomRotation(RotationDegree,resample=Image.BILINEAR))
                            
                RotationRange = random.randint(-10,10)
                Transform.append(T.RandomRotation((RotationRange,RotationRange),resample=Image.BILINEAR))
                Transform = T.Compose(Transform)
                
                image = Transform(image)

                if random.random() < 0.5:
                    image = F.hflip(image)

                if random.random() < 0.5:
                    image = F.vflip(image)
                    
                Transform = T.ColorJitter(brightness=0.2,contrast=0.2,hue=0.02)                
                image = Transform(image)
                Transform = []

        Transform.append(T.ToTensor())
        Transform = T.Compose(Transform)
        image = Transform(image)

        Norm_ = T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # normalize to [-1, 1] range
        image = Norm_(image)
        
        sample = {
            "image": image,
        }
        sample["label"] = self.labels_df.loc[index, "label"]
        sample["id"] = self.labels_df.loc[index, "id"]
        
        return sample

    def __len__(self):
        """Returns the total number of image files."""
        return len(self.images)

def get_loader(image_path, labels_csv_file, image_size, batch_size, num_workers=8, mode='train',augmentation_prob=0.4, apply_transform=True):
    """Builds and returns Dataloader."""

    dataset = ImageFolder(root = image_path, labels_csv_file=labels_csv_file, image_size =image_size, \
        mode=mode,augmentation_prob=augmentation_prob, apply_transform=apply_transform)
    data_loader = data.DataLoader(dataset=dataset,batch_size=batch_size,shuffle=True,num_workers=num_workers)
    return data_loader
