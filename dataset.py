#!pip install rasterio
import os
import torch
import numpy as np
import pandas as pd
import random
import rasterio as rio
from scipy.interpolate import interp1d
import datetime
import torch.utils.data 
from torch.utils.data import Dataset, DataLoader, random_split
from itertools import islice
import torchvision.transforms as transforms


# helper functions for class dataset
def load_img(image_stack, nbands= 0):
  sits_ = rio.open(image_stack).read()
  nImages = sits_.shape[0]
  nbands = nbands
  new_shape = (nImages // nbands, nbands, sits_ .shape[1], sits_ .shape[2])
  sits_ = sits_.reshape(new_shape)
  sits_ = torch.from_numpy(sits_.astype('float16'))
  return sits_ 
    
def date_list(path):
  with open(path, 'r') as f:
      dates = f.readlines()
  dates = [d.strip() for d in dates]
  dates = [datetime.datetime.strptime(d, "%Y%m%d").strftime('%Y-%m-%d')  for d in dates]
  return dates


  class SITSDataset(Dataset):
    def __init__(
        self, folder, norm = True,transform = None,year = ['2018', '2019']):
        super(SITSDataset, self).__init__()

        self.folder, self.nbands, self.transform, self.norm = folder, 10, transform, norm

        self.year18, self.year19  = year[0], year[1]

        self.dataframe18 = pd.read_csv(os.path.join(self.folder, f"df_patches_{self.year18}.csv" ))
        self.dataframe19 = pd.read_csv(os.path.join(self.folder, f"df_patches_{self.year19}.csv" ))

        self.sits18, self.sits19 = self.dataframe18["Patch_path"], self.dataframe19["Patch_path"]

        self.num_images =  self.sits18 + self.sits19

        self.date_path18 =  os.path.join(self.folder, f"{self.year18}_dates.txt")
        self.date_path19 =  os.path.join(self.folder, f"{self.year19}_dates.txt")

        #self.dates18 = date_list(self.date_path18)
        #self.dates19 = date_list(self.date_path19)

    def __len__(self):
        return len(self.num_images)

    def __getitem__(self, index):
        sits_18 = self.sits18[index]
        sits_19 = self.sits19[index]

        # date_18 = self.dates18[index]
        # date_19 = self.dates19[index]

        sits_18 = load_img(sits_18, self.nbands)
        sits_19 = load_img(sits_19, self.nbands)

        # apply normalisations
        if self.norm == True:
            mean_18 = np.loadtxt(os.path.join(self.folder, f"{self.year18}_mean.txt"))
            mean_19 = np.loadtxt(os.path.join(self.folder, f"{self.year19}_mean.txt"))

            std_18 = np.loadtxt(os.path.join(self.folder, f"{self.year18}_std.txt"))
            std_19 = np.loadtxt(os.path.join(self.folder, f"{self.year19}_std.txt"))

            self.norm_18 = transforms.Compose([transforms.Normalize(mean_18, std_18)])
            self.norm_19 = transforms.Compose([transforms.Normalize(mean_19, std_19)])

            sits_18, sits_19 = self.norm_18(sits_18) , self.norm_19(sits_19)

        # Apply transforms
        if self.transform is not None:
            #sits_ = self.transform(sits_18, sits_19)
            sits_combined = torch.cat([sits_18, sits_19])
            print(sits_combined.shape)
            sits_combined = self.transform(sits_combined)

        return sits_combined 