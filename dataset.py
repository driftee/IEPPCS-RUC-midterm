import torch
from torchvision import transforms, datasets
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import sys
import cfg

DATA_PATH = cfg.DATASET_PATH
DATA_AUG = True
INPUT_SIZE = cfg.INPUT_SIZE

NORM_MEAN =  [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]


from PIL import Image
import random

class RandomRotate(object):
    def __init__(self, degree, p=0.5):
        self.degree = degree
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            rotate_degree = random.uniform(-1*self.degree, self.degree)
            img = img.rotate(rotate_degree, Image.BILINEAR)
        return img


def get_transform(input_size = INPUT_SIZE, imgset = 'train'):
    if 'train' == imgset:
        return transforms.Compose([
            transforms.RandomVerticalFlip(),
            transforms.RandomHorizontalFlip(),
            RandomRotate(15, 0.3),
            transforms.ToTensor(),
            transforms.Normalize(mean = NORM_MEAN, std = NORM_STD)
        ])
    if 'test' == imgset:
        return transforms.Compose([ 
            transforms.ToTensor(),
            transforms.Normalize(mean = NORM_MEAN, std = NORM_STD)
        ])
    
    raise NotImplementedError('imgset {} not supported; supported imgset: train, test'.format(imgset))



class MyDataset(Dataset):
    def __init__(self, imgset, class_dict = None):
        self.img_dir = os.path.join(DATA_PATH, imgset)
        print("Loading {} data from {}...".format(imgset, self.img_dir))
        tmp_list= os.listdir(self.img_dir)
        self.class_dict = {}
        self.img_paths = []
        self.labels = []

        if class_dict != None:
            self.class_dict = class_dict
            tmp_list = [i for i in tmp_list if i in class_dict.values()]

        for k, class_ in enumerate(tmp_list):
            self.class_dict[k] = class_
            child_path = os.path.join(self.img_dir, class_)
            imgs = os.listdir(child_path)
            print("Class: {}, {}".format(class_, len(imgs)))
            for i in imgs:
                self.img_paths.append(os.path.join(child_path, i))
                self.labels.append(k)
            
        print("Class dict:", end = ' ')
        print(self.class_dict)
        self.img_aug = DATA_AUG
        self.transform = get_transform(input_size = INPUT_SIZE, imgset = imgset)
        self.input_size = INPUT_SIZE

    def __getitem__(self, index):
        img_path, label = self.img_paths[index], self.labels[index]
        img = Image.open(img_path).convert('RGB')
        if self.img_aug:
            img = self.transform(img)
        else:
            img = np.array(img)
            img = torch.from_numpy(img)

        return img, torch.from_numpy(np.array(int(label)))

    def __len__(self):
        return len(self.img_paths)
    
    def get_class_name(self, label):
        return self.class_dict[label]
    
    def get_dict(self):
        return self.class_dict



if __name__ =="__main__":
    print("Validating train data...")
    try:
        train_datasets = MyDataset('train')
        train_dataloader = torch.utils.data.DataLoader(train_datasets, batch_size=1, shuffle=True, num_workers=2)
        print("Train data validated.")
    except:
        print("Train data not validated.")

    print("Validating test data...")
    try:
        train_datasets = MyDataset('test')
        train_dataloader = torch.utils.data.DataLoader(train_datasets, batch_size=1, shuffle=True, num_workers=2)
        print("Test data validated.")
    except:
        print("Test data not validated.")