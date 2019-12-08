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
sns.set_context("talk") #[poster, paper, talk, notebook]
import warnings; warnings.simplefilter('ignore')
import sys

from windrose import WindroseAxes
import matplotlib.cm as cm

fs_titles = 24
fs_labels = 24
fs_axes = 20
fs_text = 20
fs_legend = 20

print('~~~~~RUNNING~~~~~ :: elev_and_aspect_post_flat.py')
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# load flattened numpys
#elevation
dem_flat = np.load('dem_flat.npy')
#aspect
asp_flat = np.load('asp_flat.npy')

print('length of dem array:', len(dem_flat))
print('length of aspect array:', len(dem_flat))

num_flights=[6,11,10,13,9,2] #iterate through this list for titles
# num_flights=[7,10] #iterate through this list for titles
# melt_flights=10
# acum_flights=7
peak_flights=6
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# #load lidar
# # fname = '/home/meganmason/Documents/projects/thesis/results/output/compiled_SUPERsnow_20m.nc' #ARS
# # fname = '~/Documents/projects/thesis/results/output/compiled_SUPERsnow.nc' #ARS
# # fname = '~/Documents/research/sierra/data/20m_analysis/compiled_SUPERsnow_20m.nc' #BSU
fname = '~/Documents/research/sierra/data/compiled_SUPERsnow.nc' #BSU
# #~~peak
# which_yr='2013'
# # #~~annual
which_yrs=range(2013,2019)

for i, which_yr in enumerate(which_yrs):

    print('~~~~~YEAR~~~~~',which_yr)
    # #~~~~ ds load
    # ds = xr.open_dataset(fname,  chunks={'time':1,'x':1000,'y':1000})
    #
    # #~~~~ ds peak (cloest to peak SWE dates)
    # # dpeak = ds.isel(time=[0,7,18,30,42,49])
    # # dpeak.close()
    # #
    # # ds = dpeak
    #
    # #~~~~ ds small
    # dsmall = ds.sel(time='{}'.format(which_yr))
    # dsmall.close()
    #
    # ds = dsmall
    #
    # #~~~convert to cm
    # ds['snow'] = ds.snow / 10
    # ds.attrs['units'] = 'cm'
    #
    # ds.close()
    # ds

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #load terrain
    #     path = '/home/meganmason/Documents/projects/thesis/data/processing_lidar/depths_3m/equal_extent/terrain/*.nc' #ARS
    # path = '/Users/meganmason491/Documents/research/sierra/data/terrain/*.nc' #BSU
    # fpath = glob.glob(path)
    # terrain=xr.open_mfdataset(fpath, concat_dim=None, chunks={'x':1000, 'y':1000}, parallel=True).rename({'Band1':'hillshade'}).drop('transverse_mercator') #combine='nested',
    #
    # terrain=np.flip(terrain.hillshade,0)
    # terrain=terrain.where(ds.mask==1)
    # terrain=terrain.to_dataset()
    # terrain.close()

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # #compute stats from xarray
    # x_gt0 = ds.snow.where(ds.snow>0)
    # mu_gt0 = x_gt0.mean(dim=('x', 'y'))
    # sig_gt0 = x_gt0.std(dim=('x', 'y'))
    #
    # #rescaled
    # # rescaled = (x_gt0 / mu_gt0)
    #
    # #standardize
    # stdize = ((x_gt0 - mu_gt0) / sig_gt0)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #compute standard deviation over time deminsion
    # stdize_std_over_time = stdize.std(dim='time')
    # print('stdize std over time complete')
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #PLOT imshow() standardized snow depths over time
    # fig= plt.figure(figsize=(8,5))
    # plt.imshow(terrain.hillshade, cmap='gray', alpha=.6)
    # h = plt.imshow(stdize_std_over_time, cmap='jet', alpha=.6, vmax=1.0)
    # #     cbar = plt.colorbar(h)
    # # cbar.set_label('$\sigma$')
    # plt.title('{}'.format(which_yr))
    # #     plt.title('standard deviation over time for flights nearest peak SWE dates', fontsize=fs_titles)
    # plt.axis('off')
    # plt.savefig('../figs/stdize_std_{}'.format(which_yr), dpi=300, transparent=True)
    # print('first plot complete')
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #consider elevation
    #flatten
    # dem_flat = ds.dem.values.flatten()
    # dem_flat = np.where(dem_flat>0, dem_flat, np.nan)

    # s_flat = stdize_std_over_time[::100].values.flatten()
    # # s_flat = np.where(s_flat>0, s_flat, np.nan) #that was dumb to code....should delete!
    # print('s is flat')
    # np.save('s_flat_{}'.format(which_yr), s_flat)


    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #PLOT elevation vs standard deviation
    # fig = plt.figure(figsize=(6, 4))
    # plt.hexbin(dem_flat, s_flat, mincnt=10, bins=1000, vmax=500, cmap='cividis')
    # # plt.xlabel('Elevation [m]')
    # plt.ylabel('$\sigma$ of SDV')
    # plt.title('{}'.format(which_yr))
    # plt.ylim(-.5,2.5)
    # # plt.colorbar(label='frequency')
    # plt.savefig('../figs/elev_vs_sigma_{}'.format(which_yr), dpi=300, transparent=True)
    # print('plot complete')

    # load standardized depth values
    s_flat = np.load('s_flat_{}.npy'.format(which_yr))
    s_flat = np.where(s_flat>0, s_flat, np.nan)
    print('starting elevation figure')

    #~~~~PLOT ELEVATION VS STD OF STANDARDIZED DEPTH VALUES~~~~~~~~~~~~~~~
    fig = plt.figure(figsize=(8, 5))
    # plt.hexbin(dem_flat, s_flat, mincnt=10, bins=10000, vmin=100, vmax=1000, cmap='cividis')
    plt.hexbin(dem_flat, s_flat, mincnt=20, bins=150, vmin=0, vmax=100, cmap='cividis')
    plt.axhline(y=1, color='k', linestyle='--')
    #
    # if which_yr == 2018:
    #     plt.xlabel('Elevation [m]')
    #
    # if which_yr == 2013:
    #     plt.colorbar(label='frequency')

    # plt.colorbar(label='frequency')
    # plt.xlabel('Elevation [m]')
    plt.ylabel('$\sigma$ of SDV')
    # plt.title('{} - {} flights'.format(which_yr, num_flights[i]))
    # plt.title('{} - {} flights'.format(which_yr, melt_flights))
    # plt.title('{} - {} flights'.format(which_yr, acum_flights))
    # plt.title('{} - {} flights'.format(which_yr, peak_flights))
    plt.annotate('{} - {} flights'.format(which_yr, num_flights[i]), xy=(1100,1.75), fontsize=fs_labels-4)

    plt.ylim(-.1,2)
    plt.xlim(1000,4000)
    # plt.tight_layout()
    plt.savefig('../figs/elev_vs_sigma_{}'.format(which_yr), dpi=300, transparent=True)
    print('elevation plot complete')

    # # ~~~~change s_flat for aspect PLOT
    # s_flat = np.where(s_flat>=0, s_flat, np.nan)
    # s_flat = np.where(s_flat<=2, s_flat, np.nan)
    #
    # print('starting aspect figure')
    # ax = WindroseAxes.from_ax()
    # ax.bar(asp_flat[~np.isnan(s_flat)], s_flat[~np.isnan(s_flat)], nsector=12, normed=True, bins=np.arange(0.001, 2, .5), cmap=cm.plasma_r)
    #
    # # if which_yr == 2013:
    # #     ax.set_legend(fontsize=28)#prop=FontProperties(size='large'))
    #
    # # ax.set_legend(fontsize=28)
    # ax.set_title('{}'.format(which_yr))
    # plt.savefig('../figs/rose_{}'.format(which_yr), dpi=300, transparent=True)
    # print('aspect plot complete')
