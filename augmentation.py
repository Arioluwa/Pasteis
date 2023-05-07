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
      # sits is a tuple of 2 time series and dates
      ts1, ts2, dates = sits
      if np.random.rand() < self.p:
        jittered = ts1 + np.random.normal(loc=0., scale=self.sigma, size=ts1.shape) 
      else:
        jittered = ts1
      return jittered, ts2, dates
      
      


class Scaling(object):
    def __init__(self, p, sigma):
        self.p = p
        self.sigma = sigma

    def __call__(self, sits):
        # sits is a tuple of 2 time series and dates
      ts1, ts2, dates = sits
      if np.random.rand() < self.p:
        scaled = np.multiply(ts1, np.random.normal(loc=1., scale=self.sigma, size=(ts1.shape[1],ts1.shape[2], ts1.shape[3])))
      else:
        scaled = ts1
      return scaled , ts2 , dates


class WindowSlice(object):
    def __init__(self, p, magnitude):
        self.p = p
        self.magnitude = magnitude

    def __call__(self, sits):
      # sits is a tuple of 2 time series and dates
        ts1, ts2, dates = sits
        if np.random.rand() < self.p:
          # Perform window slicing on the first time series in the tuple
          ts1_sliced = self._window_slice(ts1)
        else:
          ts1_sliced = ts1

        return ts1_sliced , ts2 , dates

    def _window_slice(self, ts):
        # ts is a 1D numpy array representing a time series
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


class WindowWarp(object):
  def __init__(self, p, timeframe = None, scale=[None, None]):
    self.p = p
    if timeframe==None:
      if np.random.rand() < 0.5:
        self.timeframe = "monthly"
      else:
        self.timeframe = "seasonal"
    else:
      self.timeframe = timeframe
    self.scale = scale
    self.scaling_factor = random.choice(self.scale)

  def __call__(self, sits):
    ts1, ts2, dates = sits
    dates1, dates2 = dates
    # Perform window slicing on both time series in the tuple
    if np.random.rand() < self.p:
      ts1_sliced = self._warp(ts1, dates1)
    else:
      ts1_sliced = ts1
    # Return the tuple of sliced time series
    return ts1_sliced , ts2 , dates

  def create_months_segment(self, dates):
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



  def create_seasonal_segment(self, dates):
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

  def _warp(self, ts, dates):
    ts_dates = dates
    ts = np.copy(ts) 
    output_ts = np.zeros_like(ts)
    if self.timeframe == "monthly":
      segments = self.create_months_segment(ts_dates) # segmented ts

    if self.timeframe == "seasonal":
      segments = self.create_seasonal_segment(ts_dates)  

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


class Resample(object):
  def __init__(self, p):
    self.p = p
    
  def __call__(self, sits):
    ts1, ts2, dates = sits
    date1, date2 = dates
    self.samedates = self.common_dates(date1, date2)

    if np.random.rand() < self.p:
      resampled1, _ = self._resample(ts1, date1,year=2018)
    else:
      resampled1 = ts1  

    return resampled1, ts2 , dates

  def common_dates(self, datelist1, datelist2):
    
    #with open(datelist1, 'r') as d1, open(datelist1, 'r') as d2:
    list1 = [datetime.datetime.strptime(d, "%Y-%m-%d").timetuple().tm_yday for d in datelist1]
    list2 = [datetime.datetime.strptime(d, "%Y-%m-%d").timetuple().tm_yday for d in datelist2]

    common_dates = list(set(list1) & set(list2))
    common_dates.sort()

    return common_dates

  
  def get_date_from_yday(self, yday, year):
      date = datetime(year, 1, 1) + timedelta(yday - 1)
      return date.strftime("%Y-%m-%d")

  def _resample(self, img, date_list,year=None):
    # get the original dates of the sits
  
    acquisition_dates = date_list
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
    def __init__(self, p, timeframe="", alpha=0, beta=0):
      self.p = p
      self.alpha = alpha
      self.beta = beta
      self.mask = np.random.beta(self.alpha, self.beta)

      if timeframe==None:
        self.timeframe = "monthly"
      else:
        self.timeframe = timeframe
      
    def __call__(self, sits):
      ts1, ts2 , dates = sits
      dates1 , dates2 = dates

      segs18 = self.create_s(dates1)
      segs19 = self.create_s(dates2)
      same_month_segs18, same_month_segs19 = self.compare_segments(segs18, segs19)
      # Choose a random segment from the first time series
      idx1 = np.random.randint(len(same_month_segs18))
      start1, end1, month1 = same_month_segs18[idx1]

      # Choose a random segment from the second time series of the same month
      month2_segments = [(start2, end2) for start2, end2, month2 in same_month_segs19 if month2 == month1]
      if len(month2_segments) > 0:
        idx2 = np.random.randint(len(month2_segments))
        start2, end2 = month2_segments[idx2]

        # Ensure that both segments have the same length
        seg_length = end1 - start1 + 1
        if end2 - start2 + 1 >= seg_length:
          # Apply the cutmix augmentation
          lam = np.random.beta(self.alpha, self.beta)
          lam = max(lam, 1 - lam)
          bbx1, bby1, bbx2, bby2 = self.rand_bbox(ts1.shape, lam)
          ts1[:, :, bbx1:bbx2, bby1:bby2] = ts2[:, :, bbx1:bbx2, bby1:bby2] * self.mask + ts1[:, :, bbx1:bbx2, bby1:bby2] * (1. - self.mask)
          ts2[:, :, bbx1:bbx2, bby1:bby2] = ts2[:, :, bbx1:bbx2, bby1:bby2] * (1. - self.mask) + ts1[:, :, bbx1:bbx2, bby1:bby2] * self.mask

      return ts1, ts2

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
        Jittering(p= 0.9, sigma = 0.03),
        Scaling(p = 0.9, sigma = 0.1),
        WindowSlice(p=0.8,  magnitude= 0.1),
        WindowWarp(p=0.5, timeframe = "monthly", scale=[0.5, 2.0]),
        Resample(p= 0.5),
        CutMix(p=0.5, timeframe="monthly", alpha=1.0, beta=1.0)

    ])
    self.transform_prime = transforms.Compose([
        Jittering(p= 0.9, sigma = 0.03),
        Scaling(p = 0.9, sigma = 0.1),
        WindowSlice(p=0.2,  magnitude= 0.1),
        WindowWarp(p=0.5, timeframe = "monthly", scale=[0.5, 2.0]),
        Resample(p= 0.5),
        CutMix(p=0.5, timeframe="monthly", alpha=1.0, beta=1.0)
    ])

  def __call__(self,sample, dates):
    
    x = self.transform(sample, dates)
    x_prime = self.transform(sample, dates)

    return x, x_prime

augmentations = TrainTransform()
