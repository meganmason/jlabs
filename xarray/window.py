import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import glob
import os
import time
import xarray as xr

# import dask.array as da
from dask.diagnostics import ProgressBar
from dask.distributed import Client, LocalCluster

import matplotlib.patheffects as PathEffects

# fname = '~/Documents/projects/thesis/results/output/compiled_SUPERsnow.nc' #ARS
fname = '~/Documents/research/sierra/data/compiled_SUPERsnow.nc' #BSU
#~~~~ ds full
ds = xr.open_dataset(fname,  chunks={'time':1,'x':1000,'y':1000})
ds.close()

#~~~~ ds peak
dpeak = ds.isel(time=[0,7,18,30,42,49]) #0,7,22,28,41,49]
dpeak.close()

ds = dpeak
ds.close()

# legends/labels
d_str = pd.to_datetime(ds['time'].values).strftime("%Y-%m-%d")
print('number of legend labels:', len(d_str))
# print(d_str[0][:4])

plt.figure(figsize=(10,10))
plt.imshow(ds.snow.isel(time=1))

plt.show()

d = ds.isel(x=slice(8000,9000), y=slice(3750,4750))
plt.figure(figsize=(10,10))
plt.imshow(d.snow.isel(time=1))
