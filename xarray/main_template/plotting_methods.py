import seaborn as sns
import numpy as np
import xarray as xr
from dask.diagnostics import ProgressBar
import matplotlib.pyplot as plt
import pandas as pd

from stats_methods import *


def color_palette(palette_type, flight_count, show_pal=False):
    '''
    This method gives options to the color scheme for line plots

    args:
        palette_type: color scheme [qualitative 'q', sequential 's', diverge 'd']
        flight_count: number of flights to process
        show_pal: see output or not after running option

    '''
    if palette_type is 'q' : #'qualitative' or 'q':
    # qualitative:
        c_pal = sns.color_palette("Dark2", flight_count)
        if show_pal==True:
            c_shw = sns.palplot(c_pal)


    elif palette_type is 's' : #'sequential' or 's':
    # sequential:
        c_pal = sns.color_palette(sns.cubehelix_palette(flight_count, start=.7, rot=-.75,dark=.25, light=.75)) # create color palette (plt.plot(c=c_pal[i]))
        if show_pal==True:
            c_shw = sns.palplot(sns.cubehelix_palette(flight_count, start=.7, rot=-.75,dark=.25, light=.75)) # show colors


    elif palette_type is 'd' : #'diverge' or 'd':
    # diverging:
        c_pal = sns.color_palette("RdBu_r", flight_count)
        if show_pal==True:
            c_shw = sns.palplot(c_pal)

    return c_pal


def make_my_super_plot(ds, title = '', color_scheme = 's', xlim=[]):
    '''
    WRITE STUFF HERE
    '''
    d_str = pd.to_datetime(ds['time'].values).strftime("%m-%d-%Y")
    binx = np.arange(0,2500,1) #(start, stop, step by [cm])
    # binx = np.arange(0,250,0.01)

    f=plt.figure(num=0, figsize=(14,7))
    a=plt.gca()
    c_pal = color_palette(color_scheme, ds.time.size)

    with ProgressBar(): #SAVED FIGS FROM HERE
        for i in range(ds.time.size):
            h, bx = xr.apply_ufunc(histogram, ds.snow[i].values/10, binx, dask='parallelized', output_dtypes=[np.float32])
            step_hist_plt(h, bx, i, a, c_pal[i], d_str[i], alpha=0.2, lw=4, shaded=False)

        a.set_title(title)
        a.set_xlim(*xlim)
        a.set_xlabel('snow depth [cm]')
        a.set_ylabel('frequency')
        a.legend()
        a.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
        plt.tight_layout()
        plt.show()
