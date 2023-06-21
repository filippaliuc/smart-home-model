import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import torch
import torchvision
import torch.nn as nn # All neural network modules, nn.Linear, nn.Conv2d, BatchNorm, Loss functions
import torchvision.datasets as datasets # Has standard datasets we can import in a nice way
import torchvision.transforms as transforms # Transformations we can perform on our dataset
import torch.nn.functional as F # All functions that don't have any parameters
from torch.utils.data import DataLoader, Dataset # Gives easier dataset managment and creates mini batches
from torchvision.datasets import ImageFolder
import torch.optim as optim # For all Optimization algorithms, SGD, Adam, etc.
from PIL import Image
import sys 
from firebase import storage, database

from tqdm import tqdm
from torchvision import models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # use gpu or cpu
model = torch.load("model.pth")

model.to(device)

# Check the test set
dataset = ImageFolder("cat-and-dog/test_set/test_set/", 
                     transform=transforms.Compose([
                         transforms.Resize((224, 224)), 
                         transforms.ToTensor(), 
                         transforms.Normalize([0.5]*3, [0.5]*3)
                     ]))
# print(dataset)
dataloader = DataLoader(dataset, batch_size=1, shuffle = False)

def RandomImagePrediction(filepath):
    img_array = Image.open(filepath).convert("RGB")
    data_transforms=transforms.Compose([
        transforms.Resize((224, 224)), 
        transforms.ToTensor(), 
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])
    img = data_transforms(img_array).unsqueeze(dim=0) # Returns a new tensor with a dimension of size one inserted at the specified position.
    load = DataLoader(img)
    
    for x in load:
        x=x.to(device)
        pred = model(x)
        _, preds = torch.max(pred, 1)
        if preds[0] == 1: return (f"Dog")
        else: return (f"Cat")
    
while True:
    try:
        if __name__ == "__main__":

            image_path = 'pet_image.jpeg'
            storage.child('model/pet_image.jpeg').download(image_path)
            compute = database.child("predictie").child("compute").get()

            if compute.val():
                response = RandomImagePrediction("pet_image.jpeg") # dog image
                print(response)

                database.child("predictie").child("tip").set(response)

                os.remove(image_path)
            else:
                response = ""
                database.child("predictie").child("tip").set(response)

    except KeyboardInterrupt:
        break