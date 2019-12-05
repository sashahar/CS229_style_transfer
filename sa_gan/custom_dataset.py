from torch.utils.data.dataset import Dataset
from torchvision import transforms
import pandas as pd
import numpy as np
from PIL import Image
import torch
import os

#Custom Pytorch dataloader used to keep track of gender, race, age and image name for each image tensor
#Use the tutorials and examples at the following Github link for guidance:
#https://github.com/utkuozbulak/pytorch-custom-dataset-examples

LABEL_NAMES = ["HAP", "SAD", "SUR", "ANG", "DIS", "FEA"]

class JaffeDataset(Dataset):
    def __init__(self, data_dir, labels_path, transf = None):

        if(transf == None):
            self.trans = transforms.Compose([transforms.Resize((64,64)), transforms.ToTensor()])
        else:
            self.trans = transf

        self.raw_data = pd.read_csv(os.path.join(os.getcwd(), data_dir, labels_path))
        #print(self.raw_data)
        self.img_names = np.asarray(self.raw_data['filename'])
        self.num_samples = len(self.raw_data.index)
        self.data_dir = data_dir


    def __getitem__(self, index):
        labels = np.asarray(self.raw_data[LABEL_NAMES].iloc[index])
        y_layer = torch.Tensor(np.ones((1, 64, 64)) * np.argmax(labels))
        img_name = self.img_names[index]
        # full_img = "_".join([str(age), str(gender), str(race), str(img_name)])

        abs_path = os.path.join(os.getcwd(), os.path.join(self.data_dir, img_name))

        img = Image.open(abs_path)
        img = self.trans(img)
        #transformed_img = torch.cat((img,y_layer), dim = 0)

        return img, y_layer #transformed_img #(labels, transformed_img)

    def __len__(self):
        return self.num_samples
