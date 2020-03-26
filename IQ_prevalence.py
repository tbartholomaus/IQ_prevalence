#!/opt/anaconda/envs/seisenv/bin python
# -*- coding: utf-8 -*-
"""
Goal is to identify the occurrence rate of icequakes and other properties of seismic data.
Initially written for the MoVE project.

This script processes the data and produces time series of IQ and other seismic properties.
Another script, "IQCounter_analyze.py" is used to analyze the timeseries data that is
output from this script.

Processing steps include
+ Remove instrument response
+ Highpass filter the data above 2 Hz to get clear of microseisms
+ Hilbert transform the data to find the envelope of amplitudes
+ Sort the seismic amplitudes
+ Quantify the histogram properties of these sorted envelopes:
    + How many of the envelope amplitudes exceed the threshold event_thresh? (a percent)
    + What is the amplitude of the background seismic amplitudes,
          i.e, those near the bottom noise_pctile_thresh of seismic amplitudes (presumably unaffected by events
    + What are the peak seismic amplitudes within a given time window?
+ Save these output for eventual plotting.  This script takes about 12 hrs or so to run.

IQ_prevalence.py spun off of IQCounter_loop_200128.py on Mar 26, 2020, to support use of
    par files and make more generalizable for hosting on github.
IQCounter_loop.py script spun off of med_spec_loop_v3.py on Jan 23, 2020.
@author: tbartholomaus


    
Can read input station from commandline using:
    >> nohup python -u ./IQ_prevalence.py IQ_prev.par XF BBWL > BBWL.log &    

"""

#%matplotlib inline

import numpy as np
import matplotlib.pyplot as plt
#import matplotlib.mlab as mlab
from matplotlib import dates as mdates
import pandas as pd

import obspy
from obspy.core import read
from obspy import UTCDateTime
from obspy.clients.fdsn import Client
#from obspy import signal

import datetime as dt
import pickle

import glob
import os, time
import sys
import configparser

# sys.path.append('/data/stor/basic_data/seismic_data/med_spec')
from UTCDateTime_funcs import UTCfloor, UTCceil, UTC2dn, UTC2dt64 # custom functions for working with obspy UTCDateTimes

# from clone_inv import clone_inv


os.environ['TZ'] = 'UTC' # Set the system time to be UTC
time.tzset()

#%% READ FROM THE PAR FILE
config = configparser.ConfigParser()
config.read(sys.argv[1])

data_source = config['DEFAULT']['data_source']
network = config['DEFAULT']['network']
station = config['DEFAULT']['station']
channel = config['DEFAULT']['channel']


data_dir = config['DEFAULT']['data_dir']
resp_file = config['DEFAULT']['resp_file']
out_dir = config['DEFAULT']['out_dir']


# A set of parameters that define how the script will be run
pp = {'coarse_duration': float(config['DEFAULT']['pp_coarse_duration']), # 3600.0,  # s
      'coarse_overlap' : float(config['DEFAULT']['pp_coarse_overlap']) } # 0.5}   # Ratio of overlap

pre_filt = ( float(config['DEFAULT']['pre_filt0']), 
             float(config['DEFAULT']['pre_filt1']), 
             float(config['DEFAULT']['pre_filt2']), 
             float(config['DEFAULT']['pre_filt3']) )#0.01, .02, 90, 100.)

event_thresh = eval(config['DEFAULT']['event_thresh']) # nm/s or other seismic velocity unit.  Choose the lowest possible amplitude threshold 
                         #      that is definitively an event, not background amplitude.  Amplitudes greater than
                         #      this threshold will be considered an event.
noise_pctile_thresh = float(config['DEFAULT']['noise_pctile_thresh']) # Choose the noise percentile amplitude that is not affected by events.
                         #      This percentile amplitude is considered representative of background noise levels
                         #      and may be considered to result from subglacial hydrology.

t_start = config['DEFAULT']['t_start']
t_end = config['DEFAULT']['t_end']

# ALLOW FOR OPTIONAL ARGV OVERRIDES OF THE PAR FILE
# sys.argv consists [0] is the file path, [1] is the par_file path, 
#   [2] (if it exists) is the network, [3] is the station, [4] is channel
if len(sys.argv) == 3: # i.e., one override of the par file
    network = sys.argv[2]#'7E'
elif len(sys.argv) == 4: # i.e., two overrides of the par file
    network = sys.argv[2]#'7E'
    station = sys.argv[3]#'BBWL'#TWLV'
elif len(sys.argv) == 5: # i.e., two overrides of the par file
    network = sys.argv[2]#'7E'
    station = sys.argv[3]#'BBWL'#TWLV'
    channel = sys.argv[4]



#%%
# # network = 'YF'#'DK'
# # station = 'JIG2'#'ILULI'
# network = 'IU'#'DK'
# station = 'SFJD'#'ILULI'
# # network = 'DK'
# # station = 'ILULI'
# channel = 'HHZ'

# data_source = 'IRIS'

# pp = dict()
# pp['coarse_duration'] = 10*60
# pp['coarse_overlap'] = 0.5
# pre_filt = (0.1, .2, 90, 100.)


filename = 'SELC.XX..HHZ.2018.206'


# st = obspy.read(data_dir + filename)
# st.remove_response(inv)


#%% PRINT OUTPUTS ABOUT THE RUN, FOR THE PURPOSE OF RECORDING IN THE LOG FILE
run_start_time = dt.datetime.now()
print('\n\n' + '===========================================')
print(station + ' run started: ' + '{:%b %d, %Y, %H:%M}'.format(run_start_time))
print('===========================================' + '\n')

print('Run executed from "' + os.getcwd() + '/"')
print('Run arguments consist of: ' + str(sys.argv))


print('\n\n' + '-------------------------------------------')
print('Start display of parameter file: ' + sys.argv[1])
par_time_sec = os.path.getmtime(sys.argv[1])
print('Parameter file last saved: ' + time.ctime( par_time_sec ) )
print('-------------------------------------------')
with open(sys.argv[1], 'r') as fin:
#with open(par_file[0], 'r') as fin:
    print(fin.read())
print('-------------------------------------------')
print('End display of parameter file: ' + sys.argv[1])
print('-------------------------------------------' + '\n\n')

if len(sys.argv) > 2:
    print('-------------------------------------------')
    print('Parameter file overwritten with: ' + str(sys.argv[2:]) )
    print('-------------------------------------------' + '\n\n')



#%% Read in the first set of data prior to beginning the analysis
file_counter = 0#82

print('Loading first data')
if data_source=='local':
    # READ IN FROM LOCAL FILES
#    inv_file = resp_file + 'TAKU_station.xml'
    inv_file = resp_file # For mseed files
    inv = obspy.read_inventory(inv_file)
    # inv = clone_inv(inv, network, station)

    inv = inv.select(channel=channel, station=station) # subset the inventory to just that which is necessary for script

    file_names = glob.glob(data_dir + station + '/*' + channel[-2:] +'*')
    file_names.sort()
    st = read(file_names[file_counter])

    # t_start = UTCDateTime('2018-07-20 00:00')
    # t_end = UTCDateTime('2018-08-01 00:00')
    t_start = inv[0][0].start_date # Start and end the timeseries according to the dates during which the station was running.
    t_end = inv[0][0].end_date
    # t_start = UTCDateTime(config['DEFAULT']['t_start']) # for mseed files       Start and end the timeseries according to the dates during which the station was running.
    # t_end = UTCDateTime(config['DEFAULT']['t_end']) # for mseed files       
    # inv[0][0][0].start_date = t_start
    # inv[0][0][0].end_date = t_end
    
    while(st[0].stats.starttime < inv[0][0][0].start_date): # If the first stream is before the t_start, then skip and go to the next file
        if file_counter == 0:
            print('File(s) found that pre-date t_start from the par file.')
        print(' Skipping file: ' + file_names[file_counter])
        file_counter += 1 
        st = read(file_names[file_counter])
    st.merge(method=0)
    print('\n')

if data_source!='local':
    fdsn_client = Client(data_source)#'ETH')#'IRIS')

    # READ IN FROM FDSN CLIENT
    inv = fdsn_client.get_stations(network=network, station=station, channel=channel, 
                                   location='', level="response")
    # t_start = inv[0][0].start_date
    # t_end = inv[0][0].end_date
    t_start = UTCDateTime('2018-07-20 00:00')
    t_end = UTCDateTime('2018-08-01 00:00')
    
    # Read in and remove instrument response from first day
    st = fdsn_client.get_waveforms(network=network, station=station, location='',
                                   channel=channel, starttime=t_start, endtime=t_start+86400)


# sys.exit()

st_IC = st.copy().remove_response(inventory=inv, output="VEL", pre_filt=pre_filt)
st_IC.filter('highpass', freq=2.0) # Filter the data

#%% Initialize some output variables.

# create an np.array of times at which to evaluate the seismic data.
#    The actual, calculated median powers will be calculated for the coarse_duration
#    beginning at this time.
t = np.arange( t_start, t_end, pp['coarse_duration'] * pp['coarse_overlap'])#, dtype='datetime64[D]')
#t = t[:-2]
t_dt64 = UTC2dt64(t) # Convert a np.array of obspy UTCDateTimes into datenums for the purpose of plotting

Fs_old = 0 # initialize the sampling rate with zero, to ensure proper running in the first iteration
load_trigger = np.max( (3600, pp['coarse_duration']) ) # s  Trigger for loading the next waveform file
    # When the time t is within load_trigger seconds of the end of the present
    # stream st, then load the next block (typically day) of seismic data.

env_rank = np.linspace( 0, 1, num=int(5e3) )  # 0 to 5e3 ranking of different powers.
envelopes = np.full( (len(env_rank), len(t) ), np.nan)

pct_event = np.full( len(t), np.nan ) # initialize the final array of pct_event
max_amp = np.full( len(t), np.nan ) # initialize the final array of max_amp, the array of maximum amplitudes in each snippet
noise_level = np.full( len(t), np.nan ) # initialize the final array of noise levels

#%% Start the big for loop that will go through each time and each miniseed 
#       file and calculate the median powers.  These are each time of the
#       spectrogram, each x-axis.
run_start_time = dt.datetime.now()
print('\n\n' + '===========================================')
print(station + ' run started: ' + '{:%b %d, %Y, %H:%M}'.format(run_start_time))
print('===========================================' + '\n\n')


#%% Start the big for loop that will go through each time and each miniseed 
#       file and calculate the median powers.  These are each time of the
#       spectrogram, each x-axis.

flag_up = False

# for i in np.arange(25000, 30000):#range(len(t)): # Loop over all the t's, however, the for loop will never complete
for i in np.arange(len(t)): # Loop over all the t's, however, the for loop will never complete

        
    tr_trim = st_IC[0].copy() # copy instrument corrected trace
    # the minus small number and False nearest_sample make the tr include the first data point, but stop just shy of the last data point
#    tr_trim.trim(t[i], t[i] + pp['coarse_duration'] - 0.00001, nearest_sample=False)
    tr_trim.trim(t[i], t[i] + pp['coarse_duration'] - tr_trim.stats.delta)


    # If the trimmed trace ends within load_trigger of the end of  
    #   the data stream, then load the next file.
    #   This keeps away tr_trim away from the end of the st_IC, which is tapered.
    while tr_trim.stats.endtime > st_IC[-1].stats.endtime - load_trigger:
        file_counter += 1
        print('--- Try to load in a new stream ---')
        # print ('Time within run: ' + str(t[i]) )
        
#        if t[i]+86400 >= t_end:
#            break # break out of the for loop when there are no more files to load.
        
        print("{:>4.0%}".format(float(i)/len(t)) + ' complete.  Current time in run: ' + t[i].strftime('%d %b %Y, %H:%M'))
        
        try:
            if data_source=='local':
            # Read in local file
                # Read in another day volume as another trace, and merge it 
                #   into the stream "st".  When the st is instrument corrected, t[i]
                #   will have moved on, away from the beginning of the st.
                # print('About to read in new datafile: ' + str(dt.datetime.now() - time_temp) )             
                print('About to read in new datafile: ' + file_names[file_counter] + ' at time '+ str(dt.datetime.now()))#
                st += read(file_names[file_counter])
                # sys.exit() ###########
                st.merge(fill_value='interpolate')#method=0) # Merge the new and old day volumes

                st.trim(starttime=t[i] - load_trigger, endtime=st[0].stats.endtime ) # trim off the part of the merged stream that's already been processed.

            if data_source!='local':
                # Read in from FDSN client
                st = fdsn_client.get_waveforms(network=network, station=station, location='',
                                   channel=channel, starttime=t[i]-load_trigger, endtime=t[i]+86400)
            
            
        except Exception as e:
            print(e)
            break # If there is no data to load, break out of the while loop 
                  #    and go to the next time step.

        

#        print('about to remove response: ' + str(dt.datetime.now() - time_temp) )            
        st_IC = st.copy().remove_response(inventory=inv, output="VEL", pre_filt=pre_filt)
        st_IC.filter('highpass', freq=2.0) # Filter the seismic data above this frequency level.
#        print('response now removed: ' + str(dt.datetime.now() - time_temp) )             

        print(st_IC)
#        IC_counter = 0 # Reset the IC_counter so that new st_IC will be created soon
        tr_trim = st_IC[0].copy() # copy instrument corrected trace
        # the minus small number and False nearest_sample make the tr include the first data point, but stop just shy of the last data point
#        tr_trim.trim(t[i], t[i] + pp['coarse_duration'] - 0.0000001, nearest_sample=False)
        tr_trim.trim(t[i], t[i] + pp['coarse_duration'] - tr_trim.stats.delta)
#        print(tr_trim.stats.npts)

#    print(tr_trim)
#    print(st)
    
    # Skip incomplete data records
    if tr_trim.stats.npts < pp['coarse_duration'] * int(tr_trim.stats.sampling_rate) * 1:
        print('Incomplete coarse_duration at ' + UTCDateTime.strftime(t[i], "%d %b %y %H:%M") + ': Skipping')

#        if flag_up:
#            sys.exit()
            
#        print(tr_trim.stats.npts)    
        continue
    
#    print('Calculating icequake occurrence for ' + UTCDateTime.strftime(t[i], "%d/%m/%y %H:%M"))
   
    # THESE NEXT 15-20 LINES ARE ALL THE ACTION FOR THE ICEQUAKE DETECTIONS.
    tr_env = obspy.signal.filter.envelope(tr_trim.data) # Calculate the Hilbert transformed envelope of the seismic data.         
    pct_event_tmp = np.sum(tr_env > event_thresh) / len(tr_env) # What percent of the time is occupied by seismic signals
                             #      in excess of the event_thresh amplitude?

    noise_level_tmp = np.percentile(tr_env, noise_pctile_thresh)
    # print( pct_event_tmp, noise_level_tmp )

    # Sort the envelope data from lowest amplitudes to highest amplitudes,
    #   and then write the sorted seismic data into an array for the purpose
    #   of later examining what the X percentile seismic amplitude is.
    # np.arange(len(tr_env)) / len(tr_env), np.sort(tr_env*1e6)
    envelopes[:,i] = np.interp(env_rank, 
                               np.arange(len(tr_env))/len(tr_env), 
                               np.sort(tr_env) )
    
    
    # freqs, Pdb, Fs_old = get_med_spectra_v1.med_spec(tr_trim, pp, Fs_old)
    pct_event[i]    = pct_event_tmp # Save the percent events
    max_amp[i]      = np.max(tr_env)
    noise_level[i]  = noise_level_tmp # Save the percent events

# At the end of the big for loop:
print('\n\n' + '===========================================')
print(station + ' run complete: ' + '{:%b %d, %Y, %H:%M}'.format(dt.datetime.now()))
print('Elapsed time: ' + str(dt.datetime.now() - run_start_time))
print('===========================================' + '\n\n')

# %% Save and load the output of the big runs
# Use pandas to save the output:
# IQocc = pd.DataFrame(data = (t, pct_event, noise_level), columns=['t', 'pct_event', 'noise_level', 'max_env', '5pct_env']


yymmdd = '{:%y%m%d}'.format(dt.datetime.now())
# Saving the objects:
with open(out_dir + 'Events_' + station + '_' + yymmdd + '.pickle', 'wb') as f:  # Python 3: open(..., 'wb')
    pickle.dump([t, t_dt64, pct_event, max_amp, noise_level, env_rank, envelopes, event_thresh, noise_pctile_thresh, station, run_start_time], f)

# with open('Events_SELC.pickle', 'rb') as f:  # Python 3: open(..., 'rb')
#     t, t_dt64, pct_event, max_amp, noise_level, env_rank, envelopes, station, run_start_time = pickle.load(f, encoding='latin1')
