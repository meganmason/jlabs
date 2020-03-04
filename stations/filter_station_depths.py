
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

file = '~/Desktop/tuolumne_SD_station.csv'

# read in, assign index
data = pd.read_csv(file, index_col = ['DateTime'])

# just in case
data.sort_index(inplace=True)

filtered_data = data.copy()
stations = list(data.columns)
colors = ['r','b','g']

# simple max filter
max_filter =  440
idx = filtered_data > max_filter
filtered_data[idx] = np.nan

# I used 3, 4, and 5 here.
# 5 'looks' better, but how much you manipulate the data might matter
# depending on the application.
filtered_data = filtered_data.rolling(11).median()

# dangerous!
# filtered_data = filtered_data.interpolate(limit = 20)

plt.close(0)
plt.figure(0)
a = plt.gca()

for i, stn in enumerate(stations):
    # a.plot(data[stn].values,
    #        linewidth = 0.5,
    #        linestyle = ':',
    #        color = colors[i],
    #        label = '{}: raw'.format(stn))
    a.plot(filtered_data[stn].values,
           color = colors[i],
           label = '{}: filt'.format(stn))

a.legend()

plt.show()
