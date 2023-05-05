#!pip install rasterio
import os
import torch
import numpy as np
import pandas as pd
import random
import rasterio as rio
import calendar
from scipy.interpolate import interp1d
import datetime
from datetime import timedelta
import torch.utils.data 
from torch.utils.data import Dataset, DataLoader, random_split
from itertools import islice
import torchvision.transforms as transforms


class Jittering(object):
    def __init__(self, p, sigma):
        self.p = p
        self.sigma = sigma

    def __call__(self, sits):
      # sits is a tuple of 2 time series
      jittered = [img + np.random.normal(loc=0., scale=self.sigma, size=img.shape) for img in sits]
      return tuple(jittered)
      


class Scaling(object):
    def __init__(self, p, sigma):
        self.p = p
        self.sigma = sigma

    def __call__(self, sits):
        # sits is a tuple of 2 time series
        scaled_ = tuple(np.multiply(img, np.random.normal(loc=1., scale=self.sigma, size=(img.shape[1],img.shape[2], img.shape[3])))
                        for img in sits)
        return scaled_ 


class WindowSlice(object):
    def __init__(self, p, magnitude):
        self.p = p
        self.magnitude = magnitude

    def __call__(self, sits):
        ts1, ts2 = sits
        # Perform window slicing on both time series in the tuple
        ts1_sliced = self._window_slice(ts1)
        ts2_sliced = self._window_slice(ts2)
        # Return the tuple of sliced time series
        return ts1_sliced, ts2_sliced
        #return tuple(ts1_sliced, ts2_sliced)

    def _window_slice(self, ts):
        # ts is a 1D numpy array representing a time series
        #ts = ts.copy()
        if not self.magnitude or self.magnitude <= 0 or self.magnitude >= 1:
          return ts
        seq_len = ts.shape[0]
        win_len = int(round(seq_len * (1 - self.magnitude)))
        if win_len == seq_len:
            return ts
        start = np.random.randint(0, seq_len - win_len)
        end = start + win_len

        x_old = np.linspace(0, 1, num=win_len)
        x_new = np.linspace(0, 1, num=seq_len)
        y_old = ts[start:end].numpy()
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
    curr_segment = [0]
    for i in range(1, len(dates)):
        month = dates[i].split('-')[1]
        if month != curr_month:
            segments.append(curr_segment)
            curr_month = month
            curr_segment = [i]
        else:
            curr_segment.append(i)
    segments.append(curr_segment)
    return segments

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
  return segments


class WindowWarp(object):
  def __init__(self, p, date_dir1, date_dir2,  timeframe = "", scale=[None, None]):
    self.p = p
    self.path18  = date_dir1
    self.path19  = date_dir2
    self.timeframe = timeframe
    self.scale = scale
    self.scaling_factor = random.choice(self.scale)

  def __call__(self, sits):
    ts1, ts2 = sits
    # Perform window slicing on both time series in the tuple
    ts1_sliced = self.warp(ts1, self.path18 )
    ts2_sliced = self.warp(ts2, self.path19)
    # Return the tuple of sliced time series
    return ts1_sliced, ts2_sliced

  def warp(self, ts, path):
    dates = date_list(path)
    ts = np.copy(ts) 
    output_ts = np.zeros_like(ts)
    if self.timeframe == "monthly":
      segments = create_months_segment(dates) # segmented ts
      
      segment = random.choice(segments)
      #print(segment)

      # index the window segments
      strt_idx = segment[0]  
      end_idx = segment[-1]
      first_seg = ts[:strt_idx,...]  # 1st part of the original time series 
      last_seg = ts[end_idx+1:,...]  # other part of the original time series 
      
      # create the segment window to warp
      window_b4_warp= ts[strt_idx:end_idx + 1] #.to_numpy()
      window_b4_warp_steps = np.linspace(strt_idx, end_idx, end_idx - strt_idx + 1)
      
      #
      warped_window_len = int(((end_idx - strt_idx) + 1) * self.scaling_factor)
      new_steps = np.linspace(strt_idx, end_idx, warped_window_len)
     
      interp= interp1d(window_b4_warp_steps, window_b4_warp, axis=0, kind='linear', bounds_error=False, fill_value='extrapolate')
      warped_segment = interp(new_steps)
     
      # interpolate the warped window and replace to return to the original dimension
      concat_segms = np.concatenate((first_seg, warped_segment , last_seg))
      concat_steps = np.linspace(0, concat_segms.shape[0],concat_segms.shape[0] )
      output_steps = np.linspace(0, ts.shape[0], ts.shape[0])

      interp_ = interp1d(concat_steps, concat_segms, axis=0, kind='linear', bounds_error=False, fill_value='extrapolate')
      output = interp_(output_steps)
      output_ts = output
      return output_ts

    if self.timeframe == "seasonal":
      segments = create_seasonal_segment(dates) 
      
      segment = random.choice(segments)

      # index the window segments
      strt_idx = segment[0]  
      end_idx = segment[-1]
      first_seg = ts[:strt_idx,...]  # 1st part of the original time series 
      last_seg = ts[end_idx+1:,...]  # other part of the original time series 
      
      # create the segment window to warp
      window_b4_warp= ts[strt_idx:end_idx + 1] #.to_numpy()
      window_b4_warp_steps = np.linspace(strt_idx, end_idx, end_idx - strt_idx + 1)
      
      #
      warped_window_len = int(((end_idx - strt_idx) + 1) * self.scaling_factor)
      new_steps = np.linspace(strt_idx, end_idx, warped_window_len)
     
      interp= interp1d(window_b4_warp_steps, window_b4_warp, axis=0, kind='linear', bounds_error=False, fill_value='extrapolate')
      warped_segment = interp(new_steps)
     
      # interpolate the warped window and replace to return to the original dimension
      concat_segms = np.concatenate((first_seg, warped_segment , last_seg))
      concat_steps = np.linspace(0, concat_segms.shape[0],concat_segms.shape[0] )
      output_steps = np.linspace(0, ts.shape[0], ts.shape[0])

      interp_ = interp1d(concat_steps, concat_segms, axis=0, kind='linear', bounds_error=False, fill_value='extrapolate')
      output = interp_(output_steps)
      output_ts = output
      return output_ts

class Resample(object):
  def __init__(self, p, date_dir1, date_dir2):
    self.p = p
    self.path18 = date_dir1
    self.path19 = date_dir2
    self.samedates = self.common_dates(self.path18, self.path19)
    self.dates18 = date_list(self.path18)
    self.dates19 = date_list(self.path19)

  def __call__(self, sits):
    ts1, ts2 = sits
    resampled1, _ = self.resample(ts1, self.path18, self.samedates,  year=2018)
    resampled2 , _ = self.resample(ts1, self.path19, self.samedates, year=2019)

    return resampled1, resampled2

  def date_list(self, path):
    with open(path, 'r') as f:
     dates = f.readlines()
    dates = [d.strip() for d in dates]
    dates = [datetime.datetime.strptime(d, "%Y%m%d").strftime('%Y-%m-%d')  for d in dates]
    return dates

  def common_dates(self, path1, path2):
    dates1 = path1
    dates2 = path2
    with open(path1, 'r') as d1, open(path2, 'r') as d2:
      list1 = [datetime.datetime.strptime(d.strip(), "%Y%m%d").timetuple().tm_yday for d in d1.readlines()]
      list2 = [datetime.datetime.strptime(d.strip(), "%Y%m%d").timetuple().tm_yday for d in d2.readlines()]

    common_dates = list(set(list1) & set(list2))
    common_dates.sort()

    return common_dates

  
  def get_date_from_yday(self, yday, year):
      date = datetime(year, 1, 1) + timedelta(yday - 1)
      return date.strftime("%Y-%m-%d")

  def resample(self, img, date_path, samedates, year=None):
    # get the original dates of the sits
  
    acquisition_dates = self.date_list(date_path)
    # Create list of all dates in 2018
    year_dates = []
    for month in range(1, 13):
        num_days = calendar.monthrange(year, month)[1]
        for day in range(1, num_days+1):
            date_str = f'year-{month:02d}-{day:02d}'
            year_dates.append(date_str)

    # Create new array to hold upsampled data
    upsampled_data = np.zeros((365, 10, 64, 64))
    interpolated_dates = []
    upsampled_dates = []
    #upsample data
    for i, date in enumerate(year_dates):
      if date in acquisition_dates:
        # Copy corresponding image data to new array
        index = acquisition_dates.index(date)
        upsampled_data[i] = img[index]
      else:
        interpolated_dates.append(date)
        # Perform interpolation to generate new image
        f = interp1d(np.arange(img.shape[0]), img, axis=0, kind='linear')
        upsampled_data[i] = f(i % img.shape[0])
    upsampled_dates = acquisition_dates + interpolated_dates
    upsampled_dates.sort()

    no_change_dates = [self.get_date_from_yday(yday, year) for yday in self.same_dates]
    downsampled_data = np.zeros((img.shape))
    downsampled_dates = []
    for i, dt in enumerate(upsampled_dates):
      if dt in no_change_dates:
        idx = no_change_dates.index(dt)
        downsampled_data[idx] = upsampled_data[i]
        downsampled_dates.append(dt)
       
    random_indices = random.sample(range(0, 366), 20)
    for i, index in enumerate(random_indices):
        if upsampled_dates[index] not in no_change_dates:
            downsampled_data[i + len(no_change_dates)] = upsampled_data[index]
        downsampled_dates.append(upsampled_dates[index])
        downsampled_dates.sort()

    return downsampled_data, downsampled_dates


class CutMix(object):
    def __init__(self, p, date_dir1, date_dir2, timeframe="", alpha=0, beta=0):
      self.p = p
      self.path18 = date_dir1
      self.path19 = date_dir2
      self.timeframe = timeframe
      self.alpha = alpha
      self.beta = beta
      self.mask = np.random.beta(self.alpha, self.beta)

      self.dates18 = date_list(self.path18)
      self.dates19 = date_list(self.path19)
      self.segs18 = create_s(self.dates18)
      self.segs19 = create_s(self.dates19)
      self.same_month_segs18, self.same_month_segs19 = compare_segments(self.segs18, self.segs19)

    def __call__(self, sits):
      ts1, ts2 = sits
      # Choose a random segment from the first time series
      idx1 = np.random.randint(len(self.same_month_segs18))
      start1, end1, month1 = self.same_month_segs18[idx1]

      # Choose a random segment from the second time series of the same month
      month2_segments = [(start2, end2) for start2, end2, month2 in self.same_month_segs19 if month2 == month1]
      if len(month2_segments) > 0:
        idx2 = np.random.randint(len(month2_segments))
        start2, end2 = month2_segments[idx2]

        # Ensure that both segments have the same length
        seg_length = end1 - start1 + 1
        if end2 - start2 + 1 >= seg_length:
          # Apply the cutmix augmentation
          lam = np.random.beta(self.alpha, self.beta)
          lam = max(lam, 1 - lam)
          bbx1, bby1, bbx2, bby2 = rand_bbox(ts1.shape, lam)
          ts1[:, :, bbx1:bbx2, bby1:bby2] = ts2[:, :, bbx1:bbx2, bby1:bby2] * self.mask + ts1[:, :, bbx1:bbx2, bby1:bby2] * (1. - self.mask)
          ts2[:, :, bbx1:bbx2, bby1:bby2] = ts2[:, :, bbx1:bbx2, bby1:bby2] * (1. - self.mask) + ts1[:, :, bbx1:bbx2, bby1:bby2] * self.mask

      return ts1, ts2

    def date_list(self, path):
      with open(path, 'r') as f:
        dates = f.readlines()
      dates = [d.strip() for d in dates]
      dates = [datetime.datetime.strptime(d, "%Y%m%d").strftime('%Y-%m-%d')  for d in dates]
      return dates

    def create_s(self, dates):
      segments = []
      curr_month = dates[0].split('-')[1]
      curr_start = 0
      for i in range(1, len(dates)):
        month = dates[i].split('-')[1]
        if month != curr_month:
          curr_end = i - 1
          if curr_end - curr_start >= 2:
            segments.append((curr_start, curr_end, curr_month))
            curr_month = month
            curr_start = i
      curr_end = len(dates) - 1
      if curr_end - curr_start >= 2:
        segments.append((curr_start, curr_end, curr_month))
      return segments

    def compare_segments(self, segs1, segs2):
      same_month_segs1 = []
      same_month_segs2 = []
      for seg in segs1:
        month = seg[2]
        for other_seg in segs2:
          if other_seg[2] == month:
            same_month_segs1.append(seg)
            same_month_segs2.append(other_seg)
            break
      return same_month_segs1, same_month_segs2


    def rand_bbox(self, size, lam):
      W = size[2]
      H = size[3]
      cut_rat = np.sqrt(1. - lam)  # lam = 
      cut_w = int(W * cut_rat)
      cut_h = int(H * cut_rat)

      # uniform
      cx = np.random.randint(W)
      cy = np.random.randint(H)

      bbx1 = np.clip(cx - cut_w // 2, 0, W)
      bby1 = np.clip(cy - cut_h // 2, 0, H)
      bbx2 = np.clip(cx + cut_w // 2, 0, W)
      bby2 = np.clip(cy + cut_h // 2, 0, H)

      return bbx1, bby1, bbx2, bby2

class TrainTransform(object):
  def __init__(self):
    self.transform = transforms.Compose([
        Jittering(p= 1.0, sigma = 0.03),
        Scaling(p = 1.0, sigma = 0.1),
        WindowSlice(p=0.8,  magnitude= 0.1),
        WindowWarp(p=0.5, date_dir1= d_path18, date_dir2= d_path19, timeframe = "monthly", scale=[0.5, 2.0]),
        Resample(p= 0.5, date_dir1 = d_path18, date_dir2 = d_path19),
        CutMix(p=0.5, date_dir1 = d_path18, date_dir2 = d_path19, timeframe="monthly", alpha=1.0, beta=1.0)

    ])
    self.transform_prime = transforms.Compose([
        Jittering(p= 0.1, sigma = 0.03),
        Scaling(p = 1.0, sigma = 1.0),
        WindowSlice(p=0.2,  magnitude= 0.1),
        WindowWarp(p=0.5, date_dir1= d_path18, date_dir2= d_path19, timeframe = "monthly", scale=[0.5, 2.0]),
        Resample(p= 0.5, date_dir1 = d_path18, date_dir2 = d_path19),
        CutMix(p=0.5, date_dir1 = d_path18, date_dir2 = d_path19, timeframe="monthly", alpha=1.0, beta=1.0)
    ])

  def __call__(self,sample):
    
    x = self.transform(sample)
    x_prime = self.transform(sample)

    return x, x_prime

augmentations = TrainTransform()
