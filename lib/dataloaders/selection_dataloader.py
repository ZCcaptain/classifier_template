import os
import json
import torch
from torch.utils.data import Dataset
from torchvision import transforms, utils
from PIL import Image
import numpy as np
import cv2


# normalize = transforms.Normalize(
#     mean=[0.485, 0.456, 0.406],
#     std=[0.229, 0.224, 0.225]
# )
preprocess = transforms.Compose([
    #transforms.Scale(256),
    #transforms.CenterCrop(224),
    transforms.ToTensor(),
    # normalize
])

def default_loader(path):
    img = cv2.imread(path)
    image = Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))  
    image = image.resize((224,224))
    image = np.array(image)
    img_tensor = preprocess(image)
    return img_tensor


class Selection_Dataset(Dataset):
    def __init__(self, hyper, dataset, loader=default_loader):
        self.hyper = hyper
        self.data_root = hyper.data_root
        self.pic_list = []
        self.label_list = []
        self.loader = loader

        for line in open(os.path.join(self.data_root, dataset), 'r'):
            line = line.strip("\n")
            instance = json.loads(line)

            self.pic_list.append(instance['filename'])
            self.label_list.append(instance['label'])

    def __getitem__(self, index):
        filename = self.pic_list[index]
        label = self.label_list[index]
        img = self.loader(filename)

        return img, label

    def __len__(self):
        return len(self.pic_list)