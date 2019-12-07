import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
import glob
import os
import time
import xarray as xr
# import earthpy as et
# import earthpy.spatial as es
# import earthpy.plot as ep
# import dask.array as da
from dask.diagnostics import ProgressBar
from dask.distributed import Client, LocalCluster
import seaborn as sns
sns.set_style('white')
sns.set_context("poster") #[poster, paper, talk, notebook]
import warnings; warnings.simplefilter('ignore')
import sys

fs_titles = 24
fs_labels = 24
fs_axes = 20
fs_text = 20
fs_legend = 20

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#load lidar
# fname = '/home/meganmason/Documents/projects/thesis/results/output/compiled_SUPERsnow_20m.nc' #ARS
# fname = '~/Documents/projects/thesis/results/output/compiled_SUPERsnow.nc' #ARS
# fname = '~/Documents/research/sierra/data/20m_analysis/compiled_SUPERsnow_20m.nc' #BSU
fname = '~/Documents/research/sierra/data/compiled_SUPERsnow.nc' #BSU
which_yrs=range(2018,2019)
print(which_yrs)

for which_yr in which_yrs:

    print('~~~~~YEAR~~~~~',which_yr)
    # #~~~~ ds load
    ds = xr.open_dataset(fname,  chunks={'time':1,'x':1000,'y':1000})

    # #~~~~ ds peak (cloest to peak SWE dates)
    # dpeak = ds.isel(time=[0,7,18,30,42,49]) #0,7,22,28,41,49]
    # dpeak.close()

    # ds = dpeak

    #~~~~ ds small
    # which_yr = '2016'
    dsmall = ds.sel(time='{}'.format(which_yr))
    dsmall.close()

    ds = dsmall

    #~~~convert to cm
    ds['snow'] = ds.snow / 10
    ds.attrs['units'] = 'cm'

    ds.close()
    ds

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #load terrain
#     path = '/home/meganmason/Documents/projects/thesis/data/processing_lidar/depths_3m/equal_extent/terrain/*.nc' #ARS
    path = '/Users/meganmason491/Documents/research/sierra/data/terrain/*.nc' #BSU
    fpath = glob.glob(path)
    terrain=xr.open_mfdataset(fpath, concat_dim=None, chunks={'x':1000, 'y':1000}, parallel=True).rename({'Band1':'hillshade'}).drop('transverse_mercator') #combine='nested',

    terrain=np.flip(terrain.hillshade,0)
    terrain=terrain.where(ds.mask==1)
    terrain=terrain.to_dataset()
    terrain.close()

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #compute stats from xarray
    x_gt0 = ds.snow.where(ds.snow>0)
    mu_gt0 = x_gt0.mean(dim=('x', 'y'))
    sig_gt0 = x_gt0.std(dim=('x', 'y'))

    #rescaled
    # rescaled = (x_gt0 / mu_gt0)

    #standardize
    stdize = ((x_gt0 - mu_gt0) / sig_gt0)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #compute standard deviation over time deminsion
    stdize_std_over_time = stdize.std(dim='time')

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #PLOT imshow() standardized snow depths over time
    fig= plt.figure(figsize=(25,15))
    plt.imshow(terrain.hillshade, cmap='gray', alpha=.6)
    h = plt.imshow(stdize_std_over_time, cmap='jet', alpha=.6, vmax=1.0)
#     cbar = plt.colorbar(h)
    cbar.set_label('$\sigma$')
    plt.title('{}'.format(which_yr))
#     plt.title('standard deviation over time for flights nearest peak SWE dates', fontsize=fs_titles)
    plt.axis('off')
    plt.savefig('../figs/stdize_std_{}'.format(which_yr), dpi=300, transparent=True)
    print('first plot complete')
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #consider elevation
    #flatten
    dem_flat = ds.dem.values.flatten()
    dem_flat = np.where(dem_flat>0, dem_flat, np.nan)

    s_flat = stdize_std_over_time.values.flatten()
    s_flat = np.where(s_flat>0, s_flat, np.nan)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #PLOT elevation vs standard deviation
    fig = plt.figure(figsize=(7, 5))
    plt.hexbin(dem_flat[::10], s_flat[::10], mincnt=11, bins=1000, vmax=500, cmap='cividis')
    plt.xlabel('Elevation [m]')
    plt.ylabel('$\sigma$')
    plt.title('{}'.format(which_yr))
    plt.ylim(0,5)
    plt.colorbar(label='frequency')
    plt.savefig('../figs/elev_vs_sigma_{}'.format(which_yr), dpi=300, transparent=True)
    print('second plot complete')
