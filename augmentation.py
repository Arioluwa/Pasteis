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


class Jittering(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img, sigma = 0.03):
      img = img.numpy()
      jitter = img + np.random.normal(loc=0., scale=sigma, size=img.shape)
      return jitter


class Scaling(object):
  def __init__(self, p):
    self.p = p

  def __call__(self, img, sigma = 0.1):
    output_size = (img.shape[1],img.shape[2], img.shape[3]) 
    factor = np.random.normal(loc=1., scale=sigma, size= output_size) 
    return np.multiply(img, factor) 


class WindowSlice(object):

  def __init__(self, p):
    self.p = p

  def __call__(self,img, magnitude= 0.1 ):
    ts = img.copy()

    if not magnitude or magnitude <= 0 or magnitude >= 1:
        return ts
    seq_len = ts.shape[0]
    win_len = int(round(seq_len * (1 - magnitude)))
    if win_len == seq_len:
        return ts
    start = np.random.randint(0, seq_len - win_len)
    end = start + win_len

    x_old = np.linspace(0, 1, num=win_len)
    x_new = np.linspace(0, 1, num=seq_len)
    y_old = ts[start:end] 
    f = interp1d(x_old, y_old, axis=0, kind='linear', bounds_error=False, fill_value='extrapolate')
    ts_interpolated = torch.from_numpy(f(x_new)).clone().detach()
    ts[start:end] = ts_interpolated[start:end]

    return ts



### a few helper functions ### 

def date_list(path):
  with open(path, 'r') as f:
      dates = f.readlines()
  dates = [d.strip() for d in dates]
  dates = [datetime.datetime.strptime(d, "%Y%m%d").strftime('%Y-%m-%d')  for d in dates]
  return dates


def create_months_segment(dates):
  segments = []
  curr_month = dates[0].split('-')[1]
  curr_segment = []
  for i in range(len(dates)):
    month = dates[i].split('-')[1]
    if month != curr_month:
      segments.append(curr_segment)
      curr_month = month
      curr_segment = []
      curr_segment.append(i)
  segments.append(curr_segment)


def create_seasonal_segment(dates):
  segments = []
  curr_season = ""
  curr_segment = []
  for i in range(len(dates)):
    month = int(dates[i].split('-')[1])
    if month in [12, 1, 2]:
        season = "winter"
    elif month in [3, 4, 5]:
        season = "spring"
    elif month in [6, 7, 8]:
        season = "summer"
    elif month in [9, 10, 11]:
        season = "autumn"
    if season != curr_season:
        if curr_segment:
            segments.append(curr_segment)
        curr_season = season
        curr_segment = []
    curr_segment.append(i)
  segments.append(curr_segment)


class WindowWarp(object):
  def __init__(self, p, date_dir1, date_dir2):
    self.p = p
    self.path18  = date_dir1
    self.path19  = date_dir2

  def __call__(self, ts,  timeframe = "monthly", scale=[0.5, 2.0]):
    assert isinstance(ts, tuple) and len(ts) == 2, "Input time series must be a tuple of two arrays"
    assert isinstance(timeframe, str) and timeframe in ["monthly", "seasonal"], "Invalid timeframe"
    assert isinstance(scale, list) and all(isinstance(x, float) and 0.0 <= x <= 5.0 for x in scale), "Invalid scaling factors"

    # load dates files for each time series
    dates18 = date_list(self.path18)
    dates19 = date_list(self.path19)

    ts = np.copy(ts) # Create new time series for output
    ts_18 , ts_19 = ts
      
    output_ts_18 = np.zeros_like(ts_18)
    output_ts_19 = np.zeros_like(ts_19)

    scaling_factor = random.choice(scale)

    if timeframe == "monthly":

                          ########## first time series    #############

      # Slice time series into segments based on months
      segs_18 = create_months_segment(dates18) # segmented ts
      seg_18 = random.choice(segs_18) # Choose a random segment and scaling factor

      # index the window segments
      strt_idx_18 = seg_18[0]  
      end_idx_18 = seg_18[-1]
      first_seg_18 = ts_18[:strt_idx_18,...]  # 1st part of the original time series 
      last_segment_18 = ts_18[end_idx_18+1:,...]  # other part of the original time series 
      
      # create the segment window to warp
      window_b4_warp_18 = ts_18[strt_idx_18:end_idx_18 + 1] #.to_numpy()
      window_b4_warp_steps_18 = np.linspace(strt_idx_18, end_idx_18, end_idx_18 - strt_idx_18 + 1)
      
      #
      warped_window_len_18 = int(((end_idx_18 - strt_idx_18) + 1) * scaling_factor)
      new_steps_18 = np.linspace(strt_idx_18, end_idx_18, warped_window_len_18)
     
      interp_18= interp1d(window_b4_warp_steps_18, window_b4_warp_18, axis=0, kind='linear', bounds_error=False, fill_value='extrapolate')
      warped_segment_18 = interp_18(new_steps_18)
     
      # interpolate the warped window and replace to return to the original dimension
      concat_segms_18 = np.concatenate((first_seg_18, warped_segment_18 , last_segment_18 ))
      

      concat_steps_18 = np.linspace(0, concat_segms_18.shape[0],concat_segms_18.shape[0] )
      output_steps_18 = np.linspace(0, ts_18.shape[0], ts_18.shape[0])

      interp__18 = interp1d(concat_steps_18, concat_segms_18, axis=0, kind='linear', bounds_error=False, fill_value='extrapolate')
      output_18 = interp__18(output_steps_18)
      output_ts_18 = output_18
      

                                      ########## second time series    #############

      # Slice time series into segments based on months
      segs_19 = create_months_segment(dates19)
      seg_19 = random.choice(segs_19)

      # index the window segments
      strt_idx_19 = seg_19[0]  
      end_idx_19 = seg_19[-1]
      first_seg_19 = ts_19[:strt_idx_19,...]  # 1st part of the original time series 
      last_segment_19 = ts_19[end_idx_19 + 1:,...]  # other part of the original time series 
      
      # create the segment window to warp
      window_b4_warp_19 = ts_19[strt_idx_19:end_idx_19 + 1] #.to_numpy()
      window_b4_warp_steps_19 = np.linspace(strt_idx_19, end_idx_19, end_idx_19 - strt_idx_19 + 1)
      
      #
      warped_window_len_19 = int(((end_idx_19 - strt_idx_19) + 1) * scaling_factor)
      new_steps_19 = np.linspace(strt_idx_19, end_idx_19, warped_window_len_19)
     
      interp_19 = interp1d(window_b4_warp_steps_19, window_b4_warp_19, axis=0, kind='linear', bounds_error=False, fill_value='extrapolate')
      warped_segment_19 = interp_19(new_steps_19)
     
      # interpolate the warped window and replace to return to the original dimension
      concat_segms_19 = np.concatenate((first_seg_19, warped_segment_19 , last_segment_19 ))
      

      concat_steps_19 = np.linspace(0, concat_segms_19.shape[0],concat_segms_19.shape[0] )
      output_steps_19 = np.linspace(0, ts_19.shape[0], ts_19.shape[0])

      interp__19 = interp1d(concat_steps_19, concat_segms_19, axis=0, kind='linear', bounds_error=False, fill_value='extrapolate')
      output_19 = interp__19(output_steps_19)
      output_ts_19 = output_19

      output = torch.cat([output_ts_18, output_ts_19 ])
      return output


    if timeframe == "seasonal":

                                              ########## first time series    #############
      segs_18 = create_seasonal_segment(dates18)
      seg_18 = random.choice(segs_18) # Choose a random segment and scaling factor

      # index the window segments
      strt_idx_18 = seg_18[0]  
      end_idx_18 = seg_18[-1]
      first_seg_18 = ts_18[:strt_idx_18,...]  # 1st part of the original time series 
      last_segment_18 = ts_18[end_idx_18+1:,...]  # other part of the original time series 
      
      # create the segment window to warp
      window_b4_warp_18 = ts_18[strt_idx_18:end_idx_18 + 1] #.to_numpy()
      window_b4_warp_steps_18 = np.linspace(strt_idx_18, end_idx_18, end_idx_18 - strt_idx_18 + 1)
      
      #
      warped_window_len_18 = int(((end_idx_18 - strt_idx_18) + 1) * scaling_factor)
      new_steps_18 = np.linspace(strt_idx_18, end_idx_18, warped_window_len_18)
     
      interp_18= interp1d(window_b4_warp_steps_18, window_b4_warp_18, axis=0, kind='linear', bounds_error=False, fill_value='extrapolate')
      warped_segment_18 = interp_18(new_steps_18)
     
      # interpolate the warped window and replace to return to the original dimension
      concat_segms_18 = np.concatenate((first_seg_18, warped_segment_18 , last_segment_18 ))
      

      concat_steps_18 = np.linspace(0, concat_segms_18.shape[0],concat_segms_18.shape[0] )
      output_steps_18 = np.linspace(0, ts_18.shape[0], ts_18.shape[0])

      interp__18 = interp1d(concat_steps_18, concat_segms_18, axis=0, kind='linear', bounds_error=False, fill_value='extrapolate')
      output_18 = interp__18(output_steps_18)
      output_ts_18 = output_18



       ########## second time series    #############

      # Slice time series into segments based on months
      segs_19 = create_seasonal_segment(dates19)
      seg_19 = random.choice(segs_19)

      # index the window segments
      strt_idx_19 = seg_19[0]  
      end_idx_19 = seg_19[-1]
      first_seg_19 = ts_19[:strt_idx_19,...]  # 1st part of the original time series 
      last_segment_19 = ts_19[end_idx_19 + 1:,...]  # other part of the original time series 
      
      # create the segment window to warp
      window_b4_warp_19 = ts_19[strt_idx_19:end_idx_19 + 1] #.to_numpy()
      window_b4_warp_steps_19 = np.linspace(strt_idx_19, end_idx_19, end_idx_19 - strt_idx_19 + 1)
      
      #
      warped_window_len_19 = int(((end_idx_19 - strt_idx_19) + 1) * scaling_factor)
      new_steps_19 = np.linspace(strt_idx_19, end_idx_19, warped_window_len_19)
     
      interp_19 = interp1d(window_b4_warp_steps_19, window_b4_warp_19, axis=0, kind='linear', bounds_error=False, fill_value='extrapolate')
      warped_segment_19 = interp_19(new_steps_19)
     
      # interpolate the warped window and replace to return to the original dimension
      concat_segms_19 = np.concatenate((first_seg_19, warped_segment_19 , last_segment_19 ))
      

      concat_steps_19 = np.linspace(0, concat_segms_19.shape[0],concat_segms_19.shape[0] )
      output_steps_19 = np.linspace(0, ts_19.shape[0], ts_19.shape[0])

      interp__19 = interp1d(concat_steps_19, concat_segms_19, axis=0, kind='linear', bounds_error=False, fill_value='extrapolate')
      output_19 = interp__19(output_steps_19)
      output_ts_19 = output_19

      output = torch.cat([output_ts_18, output_ts_19 ])
      return output



class TrainTransform(object):
  def __init__(self):
    self.transform = transforms.Compose([
        Jittering(p= 1.0),
        Scaling(p = 1.0),
        WindowSlice(p=0.8),
        WindowWarp(p=0.8, date_dir1= d_path18, date_dir2= d_path19)
    ])
    self.transform_prime = transforms.Compose([
        Jittering(p= 0.1),
        Scaling(p = 1.0),
        WindowSlice(p=0.2),
        WindowWarp(p=0.2, date_dir1= d_path18, date_dir2= d_path19)
        
    ])

  def __call__(self, sample):
    sits_18, sits_19 = sample[0], sample[1]
    sits_combined = torch.cat([sits_18, sits_19], dim=1)
    x1 = self.transform(sample)
    x2 = self.transform_prime(sample)
    return x1, x2
