# %matplotlib inline

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
import os
import time
import xarray as xr

fname = '~/Documents/projects/thesis/results/output/compiled_SUPERsnow.nc'

ds = xr.open_dataset(fname,  chunks={'time':1,'x':1000,'y':1000})
ds.close()

#Select peak snow dates
dsmall = ds.sel(time='2013')
dsmall.close()

## MM working
ds = dsmall
ds.close()

# Normalized variance for flights closest to peak SWE dates
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 1). mean snow depth for each year (get 6 means for each year, 2013-2018 (compute without zeros ideally))
# means=ds.snow.mean(dim=('x', 'y')) #ds.mean(dim=('x','y')--mean over all layers)
# means

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 2). standardize lidar

def standard(x):
    a = np.where(x>0, x, np.nan)
    return x/np.nanmean(a)

s = xr.apply_ufunc(standard, ds.snow, dask='parallelized', output_dtypes=[np.int16])

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 3). compute normalized variance for peak SWE dates
s_var=np.nanstd(s,axis=0) #np language


#plot
fig = plt.figure(figsize=(15, 10))
plt.imshow(np.where(s_var>0,s_var,np.nan),cmap='jet', vmax=1)
plt.colorbar()
plt.show()
