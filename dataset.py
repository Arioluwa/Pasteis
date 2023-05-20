#!pip install rasterio
import os
import torch
import numpy as np
import pandas as pd
import rasterio as rio
import datetime
import torch.utils.data as tdata
import torchvision.transforms as transforms


class SITSDataset(tdata.Dataset):
  def __init__(
      self, folder, norm = True,transform = None,year = ['2018', '2019']):
    super(SITSDataset, self).__init__()
    self.folder, self.nbands, self.transform, self.norm = folder, 10, transform, norm
    self.year18, self.year19  = year[0], year[1]
    self.dataframe18 = pd.read_csv(os.path.join(self.folder, f"df_patches_{self.year18}.csv" ))
    self.dataframe19 = pd.read_csv(os.path.join(self.folder, f"df_patches_{self.year19}.csv" ))
    self.sits18, self.sits19 = self.dataframe18["Patch_path"], self.dataframe19["Patch_path"]
    self.num_images =  self.sits18 + self.sits19
    self.dates18 = self.date_list(os.path.join(self.folder, f"{self.year18}_dates.txt"))
    self.dates19 = self.date_list(os.path.join(self.folder, f"{self.year19}_dates.txt"))
  
  def __len__(self):
    return len(self.num_images)

  def date_list(self, path):
    with open(path, 'r') as f:
        dates = f.readlines()
    dates = [d.strip() for d in dates]
    dates = [datetime.datetime.strptime(d, "%Y%m%d").strftime('%Y-%m-%d')  for d in dates]
    return dates

  def load_img(self, image_stack, nbands= 0):
    sits_ = rio.open(image_stack).read()
    nImages = sits_.shape[0]
    nbands = nbands
    new_shape = (nImages // nbands, nbands, sits_ .shape[1], sits_ .shape[2])
    sits_ = sits_.reshape(new_shape)
    sits_ = torch.from_numpy(sits_.astype('float16'))
    return sits_ 

  def __getitem__(self, index):
    sits_18 = self.sits18[index]
    sits_19 = self.sits19[index]

    sits_18 = self.load_img(sits_18, self.nbands)
    sits_19 = self.load_img(sits_19, self.nbands)
    
    # apply normalisations
    if self.norm == True:
      mean_18 = np.loadtxt(os.path.join(self.folder, f"{self.year18}_mean.txt"))
      mean_19 = np.loadtxt(os.path.join(self.folder, f"{self.year19}_mean.txt"))

      std_18 = np.loadtxt(os.path.join(self.folder, f"{self.year18}_std.txt"))
      std_19 = np.loadtxt(os.path.join(self.folder, f"{self.year19}_std.txt"))

      self.norm_18 = transforms.Compose([transforms.Normalize(mean_18, std_18)])
      self.norm_19 = transforms.Compose([transforms.Normalize(mean_19, std_19)])

      sits_18, sits_19 = self.norm_18(sits_18) , self.norm_19(sits_19)
    
    if np.random.rand() < 0.5:
      dates = (self.dates18, self.dates19)
      sample = (sits_18, sits_19, dates)
    else:
      dates = (self.dates19, self.dates18)
      sample = (sits_19, sits_18, dates)      

    #  Apply transforms
    if self.transform is not None:
      sits_ = self.transform(sample)
      
    return sits_ 

    