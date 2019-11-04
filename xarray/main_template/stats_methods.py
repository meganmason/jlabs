
import numpy as np

def histogram(x, binx):
    '''
    makes histogram for all values >0 in lidar images
    '''


    hist, bin_x = np.histogram(x[x !=0], bins=binx)
    return hist, bin_x[:-1]




def step_hist_plt(hist, bin_x, date_iter, axis, line_color, date_label, alpha=0.4, lw=2, shaded=True):
    '''
    computes step plot for histogram values
    '''

    if shaded is True:
      axis.step(bin_x, hist, lw=lw, c=line_color, label = date_label)
      axis.fill_between(bin_x, hist, step="pre", alpha=alpha)

    if shaded is False:
      axis.step(bin_x, hist, lw=lw, c=line_color, label = date_label)
