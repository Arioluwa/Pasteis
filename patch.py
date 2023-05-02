
import os
import numpy as np
import pands as pd
import rasterio as rio
import rasterio
from rasterio.windows import Window



class SITSPatches:
  def __init__(self, sits_path, output_dir, patch_size=0, year=0):
    # self.__init__()
    self.sits_path = sits_path
    self.out_dir = output_dir
    self.patch_size = patch_size
    self.year = year

    dtypes = np.dtype([("Patch_path", str),("Rows", int),("Columns", int)])
    self.dataframe = pd.DataFrame(np.empty(0, dtype=dtypes))

  def patch_sits(self):
    out_year_dir = os.path.join(self.out_dir, self.year)
    if not os.path.exists(out_year_dir):
        os.makedirs(out_year_dir)
    patches = []
    rows = []
    cols = []

    with rio.open(self.sits_path) as src:
      for x in range(0, src.width, self.patch_size):
          for y in range(0, src.height, self.patch_size):
              window = Window(x, y, self.patch_size, self.patch_size)
              patch = src.read(window=window) #-- ??, out_shape=(src.count, self.patch_size, self.patch_size)
              output_file = f'patch_{x}_{y}.tif'
              patch_path = os.path.join(out_year_dir, output_file)
              patches.append(patch_path)
              rows.append(x)
              cols.append(y)
              self.dataframe=pd.DataFrame({'Patch_path': patches, 'Rows': rows, 'Columns': cols})
              
              with rio.open(patch_path,
                            'w',
                            driver='GTiff', 
                            width=self.patch_size,
                            height=self.patch_size,
                            count=src.count,
                            dtype=patch.dtype,
                            crs=src.crs,
                            transform=src.window_transform(window)
                            ) as dst:
                  dst.write(patch)


    self.dataframe.to_csv(os.path.join(self.out_dir, f'patches_{self.year}.csv'), index=False)

project_dir = "/share/projects/erasmus/pasteis"
for y in ['data/2018/2018_subset.tif', 'data/2019/2019_subset.tif']:
    patches_ = SITSPatches(sits_path=os.path.join(project_dir, y),output_dir = os.path.join(project_dir,"/prepData/patches"), patch_size = 32, year=str(y))
    patches_.patch_sits()

# if __name__ == "__main__" :
#     project_dir = "/share/projects/erasmus/pasteis"
#     for y in ['data/2018/2018_subset.tif', 'data/2019/2019_subset.tif']:
#         patches_ = SITSPatches(sits_path=os.path.join(project_dir, y),output_dir = os.path.join(project_dir,"/prepData/patches"), patch_size = 32, year=str(y))
#         patches_.patch_sits()
