import torch
import numpy as np
from scipy.interpolate import interp1d
import datetime
import torchvision.transforms as transforms
import random


class Jittering(object):
    def __init__(self, p, sigma):
      self.p = p
      self.sigma = sigma

    def __call__(self, sits):
      # sits is a tuple of 2 time series and dates
      ts1, ts2, dates = sits

      if np.random.rand() < self.p:
        t1_jittered = ts1 + torch.normal(mean=0., std = self.sigma, size=(ts1.shape)) 
      else:
        t1_jittered = ts1
        
      return t1_jittered, ts2, dates
      
      
      
class Scaling(object):
    def __init__(self, p, sigma):
      self.p = p
      self.sigma = sigma

    def __call__(self, sits):
      # sits is a tuple of 2 time series and dates
      ts1, ts2, dates = sits

      if np.random.rand() < self.p:
        factor = torch.normal(mean=1., std = self.sigma, size=(ts1.shape[1], ts1.shape[2], ts1.shape[3]))
        t1_scaled = torch.multiply(ts1, factor)
      else:
        t1_scaled = ts1

      return t1_scaled , ts2 , dates


class Window_Interpolation(object):
    def __init__(self, p, crop_range):
        self.p = p
        self.crop_range = crop_range

    def __call__(self, sits):
      # sits is a tuple of 2 time series and dates
        ts1, ts2, dates = sits
        dates1, _ = dates

        if np.random.rand() < self.p:
          # Perform window slicing on the first time series in the tuple
          ts1_win_interp = self._window_interp(ts1, dates1)
        else:
          ts1_win_interp = ts1

        return ts1_win_interp , ts2 , dates

    def day_of_year(self, date_list):
      date_list = [d.strip() for d in date_list]
      date_list = [datetime.datetime.strptime(d, "%Y-%m-%d").timetuple().tm_yday for d in date_list]
      doy_s = [d for d in date_list]
      return doy_s

    def _window_interp(self, ts, dates):
        # ts is a 1D numpy array representing a time series
        seq_len = ts.shape[0]
        low, high = self.crop_range
        crop_size = np.random.uniform(low, high)
        doy = torch.tensor(self.day_of_year(dates))

        win_len = int(round(seq_len * (1 - crop_size)))
        if win_len == seq_len:
            return ts
        start = int(torch.randint(0, seq_len - win_len, (1,)).item())
        end = start + win_len

        dates_a = torch.cat((doy[:start], doy[end+1:]))
        ts_a = torch.cat((ts[:start],ts[end+1:]))

        f = interp1d(dates_a, ts_a, axis=0, kind='linear', bounds_error=False, fill_value='extrapolate')
        ts_interpolated = torch.from_numpy(f(doy[start:end+1]))
        ts[start:end+1] = ts_interpolated
        
        return ts


class Window_Warping(object):
  def __init__(self, p, timeframe = None, scale=[None, None]):
    self.p = p
    self.scale = scale

    if timeframe==None:
      if np.random.rand() < 0.5:
        self.timeframe = "monthly"
      else:
        self.timeframe = "seasonal"
    else:
      self.timeframe = timeframe
    

  def __call__(self, sits):
    ts1, ts2, dates = sits
    dates1, _ = dates
    # Perform window slicing on both time series in the tuple
    if np.random.rand() < self.p:
      ts1_win_warped = self._warp(ts1, dates1)
    else:
      ts1_win_warped = ts1
    # Return the tuple of sliced time series
    return ts1_win_warped , ts2 , dates

  def day_of_year(self, date_list): 
    date_list = [d.strip() for d in date_list]
    date_list = [datetime.datetime.strptime(d, "%Y-%m-%d").timetuple().tm_yday for d in date_list]
    date = [d for d in date_list]
    return date

  def create_months_segment(self, dates): 
    segments = []
    curr_month = int(dates[0].split('-')[1])
    curr_segment = [datetime.datetime.strptime(dates[0], '%Y-%m-%d').timetuple().tm_yday]
    for i in range(1, len(dates)):
        month = int(dates[i].split('-')[1])
        day_of_year = datetime.datetime.strptime(dates[i], '%Y-%m-%d').timetuple().tm_yday
        if month != curr_month:
            segments.append(curr_segment)
            curr_month = month
            curr_segment = [day_of_year]
        else:
            curr_segment.append(day_of_year)
    segments.append(curr_segment)
    return segments

  def calculate_day_of_year(self, month, day):
    days_in_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    day_of_year = day
    for i in range(month - 1):
        day_of_year += days_in_month[i]
    return day_of_year



  def create_seasonal_segment(self, dates):
    segments = []
    curr_season = ""
    curr_segment = []
    
    for i in range(len(dates)):
        date_parts = dates[i].split('-')
        month = int(date_parts[1])
        day = int(date_parts[2])
        day_of_year = self.calculate_day_of_year(month, day)
        
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
            
        curr_segment.append(day_of_year)
    
    segments.append(curr_segment)
    return segments



  def _warp(self, ts, dates):
    doy = self.day_of_year(dates)
    scaling_factor = random.choice(self.scale)
    
    if self.timeframe == "monthly":
      segments = self.create_months_segment(dates) 
    elif self.timeframe == "seasonal":
      segments = self.create_seasonal_segment(dates) 
    else:
      raise ValueError("Invalid timeframe provided") 

    segment = random.choice(segments)

    # start and end doy of chosen window segment
    strt_doy = segment[0]  
    end_doy = segment[-1]

    # convert the doy identity of the segments to indices of the time steps
    indices=[]
    for i, d_y in enumerate(doy):
      if d_y == strt_doy:
        indices.append(i)
      if d_y == end_doy:
        indices.append(i)

    strt_idx = indices[0]
    end_idx = indices[1]
    
    dates_a = doy[:strt_idx]+ doy[end_idx+1:]
    
    first_seg = ts[:strt_idx]  
    last_seg = ts[end_idx+1:] 

    ts_a = np.concatenate((first_seg, last_seg))

    scaled_window_len = int((doy[end_idx]-doy[strt_idx]) * scaling_factor)
    new_points = np.linspace(doy[strt_idx], doy[end_idx], num=scaled_window_len).astype(int)

    f = interp1d(dates_a, ts_a, axis = 0, kind="linear", bounds_error=False, fill_value='extrapolate') 
    ts_warped = f(new_points)
    

    # second interpolation to return to original shape
    concat_segs = np.concatenate((first_seg, ts_warped, last_seg))
    newer_points = doy[:strt_idx] + list(new_points)+ doy[end_idx+1:] 
    
    f2 = interp1d(newer_points, concat_segs, axis = 0, kind="linear", bounds_error=False, fill_value='extrapolate')
    warped = f2(doy)
    warped = torch.from_numpy(warped)

    return warped



class Resampling(object):
  def __init__(self, p, num_new_dates, constant_ratio = None):
    self.p = p
    self.num_new_dates = num_new_dates
    self.constant_ratio = constant_ratio
    
  def __call__(self, sits):
    ts1, ts2, dates = sits
    date1, date2 = dates
    d_ = None
    if np.random.rand() < self.p:
      resampled1, d_ = self._resample(ts1, date1)
    else:
      resampled1 = ts1 
      d_ = date1

    dates = d_ , date2 
    return resampled1, ts2 , dates
  
  def get_days_of_year(self, year):
    days_of_year = []
    days_in_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]  # Number of days in each month
    day_of_year = 1
    for month, num_days in enumerate(days_in_month, start=1):  # Iterate through each month
        for day in range(1, num_days + 1):  # Iterate through each day of the current month
            days_of_year.append(day_of_year)
            day_of_year += 1
        if month == 12:
            break  # Break the loop after adding the days of December
    return days_of_year


  def day_of_year(self, dates): 
    dates = [d.strip() for d in dates]
    dates = [datetime.datetime.strptime(d, "%Y-%m-%d").timetuple().tm_yday for d in dates]
    doy_s = [d for d in dates]
    return doy_s

  def convert_day_of_year_to_dates(self, day_of_year_list, year):
    dates = []
    for day_of_year in day_of_year_list:
      day_of_year = day_of_year
      date = datetime.datetime(int(year), 1, 1) + datetime.timedelta(int(day_of_year) - 1)
      date_string = date.strftime("%Y-%m-%d")
      dates.append(date_string)
    return dates


  def _resample(self, ts, dates):
    first_date = dates[0]
    year = first_date.split('-')[0]
    acq_doy = self.day_of_year(dates)
    entire_doy = self.get_days_of_year(year)
    num_dates_to_keep_unchanged = int(round(ts.shape[0] * self.constant_ratio))
    
    acquisition_day_1 = acq_doy[0]
    acquisition_day_last = acq_doy[-1]
  
    all_dates_within_bounds = [yday for yday in entire_doy if acquisition_day_1 <= yday <= acquisition_day_last]

    dates_to_choose_from = []
    for d in all_dates_within_bounds:
        if d not in acq_doy:
            dates_to_choose_from.append(d)

    chosen_dates = np.random.choice(dates_to_choose_from, size=self.num_new_dates, replace=False)
    upsampled_dates = list(chosen_dates) + acq_doy
    upsampled_dates.sort()
    upsampled_dates = np.nan_to_num(upsampled_dates)
    
    ## interpolate 
    f = interp1d(acq_doy, ts, axis = 0, kind="linear", bounds_error=False, fill_value= 'extrapolate')
    upsampled_ts = f(upsampled_dates)

    # downsample the new data back to the original shape, keeping some dates from the initial acquisition constant, interpolate other points
    constant_dates = np.random.choice(acq_doy, size = num_dates_to_keep_unchanged, replace = False)
    
    dates_to_choose_from2 = []
    for d in upsampled_dates:
      if d not in constant_dates:
        dates_to_choose_from2.append(d)
    chosen_dates2 = np.random.choice(dates_to_choose_from2, size = len(acq_doy) - num_dates_to_keep_unchanged , replace = False )
    downsampled_dates = list(chosen_dates2)  + list(constant_dates )
    downsampled_dates.sort()

    ## interpolate 
    f2 = interp1d(upsampled_dates, upsampled_ts, axis = 0, kind="linear", bounds_error=False, fill_value= 'extrapolate' )
    downsampled_ts = torch.from_numpy(f2(downsampled_dates))
    downsampled_dates_ = self.convert_day_of_year_to_dates(downsampled_dates, year)

    return downsampled_ts, downsampled_dates_

class Cut_Mixing(object):
    def __init__(self, p,  alpha=0, beta=0, timeframe=""):
      self.p = p
      self.alpha = alpha
      self.beta = beta
      

      if timeframe==None:
        self.timeframe = "monthly"
      else:
        self.timeframe = timeframe
      
    def __call__(self, sits):
      ts1, ts2 , dates = sits
      dates1 , dates2 = dates
      self.mask = torch.distributions.beta.Beta(self.alpha, self.beta).sample()
      segs18 = self.create_s(dates1)
      segs19 = self.create_s(dates2)
      same_month_segs18, same_month_segs19 = self.compare_segments(segs18, segs19)
      
      # Choose a random segment from the first time series
      idx1 = torch.randint(len(same_month_segs18), (1,))
      start1, end1, month1 = same_month_segs18[idx1]       

      # Choose a random segment from the second time series of the same month
      start2, end2, _ = None, None, None
      for i, m_seg in enumerate(same_month_segs19):
        if m_seg[2] == month1:
            start2, end2, _ = same_month_segs19[i]
            break

      # cutmix augmentation
      num_channels = ts1.shape[1]
      for channel in range(num_channels):
        ts1[start1:end1+1, channel, :, :] = ts2[start1:end1+1, channel, :, :] * self.mask + ts1[start1:end1+1, channel,  :, :] * (1. - self.mask)
        ts2[start2:end2+1, channel,  :, :] = ts2[start2:end2+1, channel,  :, :] * (1. - self.mask) + ts1[start2:end2+1, channel,  :, :] * self.mask

      return ts1, ts2 , dates

    def create_s(self, dates): # creates monthly segments and attach each month for identification
      segments = []
      start_index = 0
      end_index = 0
      current_month = dates[0].split('-')[1]
      for i in range(1, len(dates)):
          if dates[i].split('-')[1] != current_month:
              segments.append((start_index, end_index, str(current_month)))
              start_index = i
              current_month = dates[i].split('-')[1]
          end_index = i
      segments.append((start_index, end_index, str(current_month)))
      return segments

    def compare_segments(self,segs1, segs2):
      same_month_segs1 = []
      same_month_segs2 = []
      for seg in segs1:
          month = seg[2]
          if (seg[1]-seg[0] + 1) >= 3:
              for other_seg in segs2:
                  if other_seg[2] == month and (other_seg[1]-other_seg[0] + 1) >= 3:
                      same_month_segs1.append(seg)
                      same_month_segs2.append(other_seg)
                      break          
      return same_month_segs1, same_month_segs2
   

class TrainTransform(object):
  def __init__(self):
    self.transform = transforms.Compose([
        Jittering(p= 0.9, sigma = 0.03),
        Scaling(p = 0.9, sigma = 0.1),
        Window_Interpolation(p=0.5, crop_range=[0.6, 0.9]),
        Cut_Mixing(p=0.5,  alpha=1.0, beta=1.0, timeframe="monthly"),
        Window_Warping(p=0.5,timeframe = None, scale=[0.5, 2.0]),
        Resampling(p= 0.5, num_new_dates=10, constant_ratio = 0.6)
    ])
    self.transform_prime = transforms.Compose([
        Jittering(p= 0.9, sigma = 0.03),
        Scaling(p = 0.9, sigma = 0.1),
        Window_Interpolation(p=0.5, crop_range=[0.6, 0.9]), 
        Cut_Mixing(p=0.5,  alpha=1.0, beta=1.0, timeframe="monthly"),               
        Window_Warping(p=0.5, timeframe = None, scale=[0.5, 2.0]),
        Resampling(p= 0.5, num_new_dates=10, constant_ratio = 0.6),
    ])

  def __call__(self,sample):
    
    x, _, xdates = self.transform(sample)
    x_prime, _, xprime_dates = self.transform(sample)
    dates = (xdates[0], xprime_dates[0])

    return x, x_prime, dates


