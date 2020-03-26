#!/opt/anaconda/envs/seisenv/bin python
# -*- coding: utf-8 -*-
"""
Spun off of med_spec_loop_v3.py on Jan 23, 2020.
Goal is to identify the occurrence rate of IQs.

Created on Fri Feb 19 12:06:45 2016
Completed _v1 on March 27, 2017

@author: tbartholomaus

Wrapper script that calculates the median spectra from a timeseries of seismic
data, following the methods described in Bartholomaus et al., 2015, in GRL.

This wrapper script handles the data loading and manipulation, and passes a 
coarse_duration length of seismic data to the script get_med_spectra, to
calculate the median spectra of many small samples of seismic data with length 
fine_duration.  get_med_spectra.py returns the median spectra, which is stored
in an array, and finally plotted.

_v2 Nov 29, 2017: Modified to deconvolve the instrument response from the
    waveforms.
_v3 Nov 15, 2018: Modified to use a parameter file for all parameters.
    
Can read input station from commandline using:
    >> nohup python -u ./med_spec_loop_v3.py med_spec.par XF BBWL > BBWL.log &    

"""

#%matplotlib inline

import numpy as np
import matplotlib.pyplot as plt
#import matplotlib.mlab as mlab
from matplotlib import dates as mdates
import pandas as pd

import datetime as dt
import pickle

import glob
import os, time
import sys
import configparser
#import imp
#imp.reload(get_med_spectra)

#from clone_inv import clone_inv
import sys
sys.path.append('/data/stor/basic_data/seismic_data/med_spec')
sys.path.append('/Users/timb/Documents/syncs/OneDrive - University of Idaho/RESEARCHs/med_spec')
from UTCDateTime_funcs import UTCfloor, UTCceil, UTC2dn, UTC2dt64 # custom functions for working with obspy UTCDateTimes

from clone_inv import clone_inv


os.environ['TZ'] = 'UTC' # Set the system time to be UTC
time.tzset()

import seaborn

from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

# %% Read in the output of the big runs

IQ_path = '/Users/timb/Documents/syncs/OneDrive - University of Idaho/RESEARCHs/MoVE_Gulley_Greenland/Seis_analysis/IQ_occurrence/'
IQ_file = 'Events_SELC_200128.pickle'
# IQ_file = 'Events_SELC.pickle'

with open(IQ_path + IQ_file, 'rb') as f:  # Python 3: open(..., 'rb')
    t, t_dt64, pct_event, max_amp, noise_level, env_rank, envelopes, event_thresh, noise_pctile_thresh, station, run_start_time = pickle.load(f, encoding='latin1')
    # t, t_dt64, pct_event, max_amp, noise_level, env_rank, envelopes, station, run_start_time = pickle.load(f, encoding='latin1')


#%% Create the CSV file
temp_dict = {'time':t_dt64, 'pct_event':pct_event, 'max_amp':max_amp, 'noise_level':noise_level}
IQocc = pd.DataFrame(data = temp_dict, 
                     columns=['time', 'pct_event', 'max_amp', 'noise_level']
                     )

now_time_str = dt.datetime.now().strftime("%m/%d/%Y, %H:%M")
meta = ('##  Output from analyses of seismic data recorded at SELC during 2018.\n',
        '##  These outputs are posted every 5 minutes, representing the \n', 
        '##      value calculated over the following 10 min.\n'
        '##  CSV file written on ' + now_time_str + ' UTC. \n',
        '##  '+ '\n',
        '##  Data written from pickle file "' + IQ_file + '"'+ '\n',
        '##  Data use an event_thresh of ' + str(event_thresh)+ ' m/s in order to identify an event.\n', 
        '##  Data use a noise_pctile_thresh of ' + str(noise_pctile_thresh)+ ' to identify \n',
        '##      what percentile amplitudes are considered background noise.\n',
        '##  '+ '\n'
        )

csv_file = 'IQ_occurrence.csv'
with open(csv_file,'w') as fd:
    for i in range(len(meta)):
        fd.writelines(meta[i])

IQocc.to_csv(csv_file, mode='a')
#%% Basic seismic timeseries
fig, ax = plt.subplots(num = 11, clear=True, nrows=3, sharex=True)
ax[0].plot(t_dt64, pct_event)
ax[1].plot(t_dt64, noise_level *1e6)
ax[2].plot(t_dt64, max_amp *1e6)

ax[0].grid('on')
ax[1].grid('on')
ax[2].grid('on')

ax[0].set_ylim(0, .6)
ax[1].set_ylim(0, 2e-1)
ax[2].set_ylim(0, 500)

ax[0].set_ylabel('Portion of time\nwith events')
ax[1].set_ylabel('Noise level (um/s)')
ax[2].set_ylabel('Max amplitude (um/s)')
fig.autofmt_xdate()

#%%
fig, ax = plt.subplots(num = 2, clear=True, nrows=1, sharex=True)
ax.scatter(noise_level *1e6, pct_event, s=4)
seaborn.kdeplot(noise_level *1e6, pct_event)
# ax[1].plot(t_dt64, noise_level *1e6)
ax.set_xlabel('noise level')
ax.set_ylabel('percent of events')
#%% Look at the full envelopes
fig, ax = plt.subplots(num = 5, clear=True)
# mesh = ax.pcolormesh(t_dt64, env_rank*100, np.log10(envelopes),
#                       vmin=-8, vmax=-5)
mesh = ax.pcolormesh(t_dt64[:5:], env_rank[:20:]*100, 
                     np.log10(envelopes[:20:, :5:]),
                      vmin=-8, vmax=-5)
plt.colorbar(mesh)
fig.autofmt_xdate()
ax.set_ylabel('Percentile of seismic amplitudes')

#%% Look at what percentiles are affected by events
pctiles = np.array( [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9] )
levels = np.empty( (len(pctiles), len(t_dt64) ) )
for i in range(len(pctiles)):
    pct_ind = np.where(env_rank > pctiles[i])[0][0]
    # print(env_rank[pct_ind])
    levels[i,:] = envelopes[pct_ind, :]

plt.subplots(num=6, clear=True)
plt.plot(t_dt64, levels.T)
fig.autofmt_xdate()


#%% What are typical shapes of the envelopes?
# times = np.array( (np.datetime64('2018-07-21 19:00'), 
#                    np.datetime64('2018-07-22 09:00'),
#                    np.datetime64('2018-07-24 17:00'),                   
#                    np.datetime64('2018-07-25 17:00'),
#                    np.datetime64('2018-07-25 19:00'),
#                    np.datetime64('2018-07-25 20:00'),
#                    np.datetime64('2018-07-25 21:00'),
#                    np.datetime64('2018-07-26 02:00'),
#                    np.datetime64('2018-07-26 12:00'),
#                    np.datetime64('2018-07-27 02:00'),
#                    ) )
times = np.array( (np.datetime64('2018-08-05 19:00'), 
                   np.datetime64('2018-08-10 09:00'),
                   np.datetime64('2018-08-11 17:00'),                   
                   np.datetime64('2018-08-12 17:00'),
                   np.datetime64('2018-08-12 19:00'),
                   np.datetime64('2018-08-13 20:00'),
                   np.datetime64('2018-08-14 21:00'),
                   np.datetime64('2018-08-14 02:00'),
                   np.datetime64('2018-08-15 12:00'),
                   np.datetime64('2018-08-15 02:00'),
                   ) )

levels = np.empty( (len(times), len(env_rank) ) )

fig, ax = plt.subplots(num=7, clear=True)

for i in range(len(times)):
    time_ind = np.where(t_dt64 > times[i])[0][0]
    # print(env_rank[pct_ind])
    levels[i,:] = envelopes[:, time_ind]
    ax.plot(env_rank*100, levels[i,:].T*1e6, label=str(times[i]))

# plt.subplots(num=7, clear=True)
# plt.plot(env_rank, levels.T*1e6, label=str(times))
# fig.autofmt_xdate()
ax.set_ylim(0, 1)
fig.legend(loc = 'upper left')
