import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from glob import glob
from PIL import Image
"""
How to prepare datas
1. move datas to ./data
2. unzip image.zip
"""
class ArtDataset(Dataset):
    def __init__(self, split):
        with open('data/{}.csv'.format(split)) as f:
            df = pd.read_csv(f)
        self.ids = df['id'].tolist()
        if split == 'test':
            self.categories = [0] * len(self.ids)
        else:
            self.categories = df['category'].tolist()
        assert(len(self.ids) == len(self.categories)), "Data error"
        
        if split == 'train':
            self.transforms=transforms.Compose([
                        transforms.RandomResizedCrop(224),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize((0.485, 0.456, 0.406),
                                             (0.229, 0.224, 0.225))])
        else:
            self.transforms=transforms.Compose([
                        transforms.Resize(256),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        transforms.Normalize((0.485, 0.456, 0.406),
                                             (0.229, 0.224, 0.225))])
    def __len__(self):
        return len(self.ids)
    
    def __getitem__(self, idx):
        im_id = self.ids[idx]
        label = int(self.categories[idx])
        
        # Open image
        #im_paths = glob('data/total_set/{}*'.format(im_id))
        #if len(im_paths) == 0:
        #    print(im_id, "Not exists")
        #img = Image.open(im_paths[0])
        img = Image.open('data/total_set/{}'.format(im_id))
        img = img.convert("RGB")
        return self.transforms(img), label
    
def get_dataloader(split, batch_size, shuffle, num_workders):
    dataset = ArtDataset(split)
    dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workders
    )
    return dataloader