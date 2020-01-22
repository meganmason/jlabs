import pandas as pd
import numpy as np
import xarray as xr
import glob
import matplotlib.pyplot as plt
import seaborn as sns

path = '/Users/meganmason491/Documents/research/sierra/data/stations/*.csv'
flist = glob.glob(path)

############ stations #########################################################
# #dataframe of files
# files_df = pd.DataFrame()
#
# for f in sorted(flist):
#     site_id = f.split("/")[-1]
#     site_id = site_id[:3]
#     print(site_id)
#
#     kv = {'site-id':site_id, 'filename': f}
#     files_df = files_df.append(kv, ignore_index=True)
#
# files_df.set_index('site-id', inplace=True)
#
# #dataframe of snow depths
#
# for row in files_df.iterrows():
#     print(row.name)
#     # files.df.loc[]
#     # pd.read_csv('')

# dfs = []
#
# for f in sorted(flist):
#     d = pd.read_csv(f, header=1, index_col=0, names=['DEPTH [in]','FLAG'])
#     pd.concat(d)
#
# print(d)


#######  lidar  ###############################################################
fname = '~/Documents/research/sierra/data/compiled_SUPERsnow.nc' #BSU

ds = xr.open_dataset(fname,  chunks={'time':1,'x':1000,'y':1000})
ds.close()
ds['snow'] = ds.snow / 10
ds.attrs['units'] = 'cm'
ds=ds.astype(np.int16, copy=False)

print(ds)

dan = ds.sel(x=slice(301551, 301552), y=slice(4196788, 4196789))
print(dan)
dan.plot()
# tum =
#
# sli =
