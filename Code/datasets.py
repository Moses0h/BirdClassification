from __future__ import print_function
import torch
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
from torchvision import transforms
import copy


# Dataset class to preprocess your data and labels
# You can do all types of transformation on the images in this class


class bird_dataset(Dataset):
    # You can read the train_list.txt and test_list.txt files here.
    def __init__(self,root,file_path):
        self.images = []
        self.labels = []
        self.root = root
        for data in open(root + file_path, 'r'):
            path, label = data.split()
            image = Image.open(root + "images/" + path, 'r')
            #print(image.size, image.format, image.mode, path)
            if image.mode != "RGB":
                image = image.convert("RGB")
            self.images.append(image)
            self.labels.append(label)
        #self.images = torch.stack(self.images)
        #self.labels = torch.FloatTensor(self.labels)
        pass

    def __len__(self):
        return len(self.images)
        raise ("Not implemented")

    # Reshape image to (224,224).
    # Try normalizing with imagenet mean = [0.485, 0.456, 0.406] and std = [0.229, 0.224, 0.225] or
    # any standard normalization
    # You can other image transformation techniques too
    def __getitem__(self, item):
        #print("data", list(item.getdata()))
        imagenet_mean = [0.485, 0.456, 0.406]
        imagenet_std = [0.229, 0.224, 0.225]
        
        image = self.images[item]
        label = self.labels[item]
        label = torch.from_numpy(copy.deepcopy(np.asarray(label, dtype='long')))

        image = image.resize((224, 224))
        image = np.asarray(image, dtype='float64').transpose(-1, 0, 1) # Change from (w x h x c) to (c x w x h)
        image1 = torch.from_numpy(copy.deepcopy(np.asarray(image))) # Not normalized + warning, no warning with dtype='float64'
        image1 = torch.div(image1, 225)

        image1 = transforms.Normalize(imagenet_mean, imagenet_std)(image1)

        # print("image1", image1)
        # print(image1.shape)
        # image2 = transforms.ToTensor()(np.array(image)) #Normalized if not have dtype='float64' earlier?
        # print("image2", image2)
        # print(image2.shape) 
        #return image2
        return image1, label
        raise ("Not implemented")