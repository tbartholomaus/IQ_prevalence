# IQ_prevalence
Repository contains scripts to examine the prevalence of icequakes or other events within a set of miniseed files

There are two critical scripts here and a par file that includes parameters for running the scripts.

## IQ_prevalence.py
The script `IQ_prevalence.py` analyzes the miniseed files and pickles the analyzed output: a time series of IQ and other seismic properties.  This script should be run from a terminal window (rather than an IDE like Spyder), and requires input from a par file to run.  An example par file, `IQ_prev.par` is included in this repo.

Execute this script from the bash command line using a modification of the following command:
```
nohup python -u /data/stor/basic_data/seismic_data/git_repos/IQ_prevalence/IQ_prevalence.py IQ_prev_MoVE.par XX SELC > SELC_events.log &
```

Processing steps followed by `IQ_prevalence.py` include
+ Remove instrument response
+ Highpass filter the data above 2 Hz to get clear of microseisms
+ Hilbert transform the data to find the envelope of amplitudes
+ Sort the seismic amplitudes
+ Quantify the histogram properties of these sorted envelopes:
    + How many of the envelope amplitudes exceed the threshold event_thresh? (a percent)
    + What is the amplitude of the background seismic amplitudes,
          i.e, those near the bottom noise_pctile_thresh of seismic amplitudes (presumably unaffected by events
    + What are the peak seismic amplitudes within a given time window?
+ Save these output for eventual plotting.

This script takes about 12 hrs or so to run on a summer's worth of data.

## IQCounter_analyze.py
Another script,`IQCounter_analyze.py` is used to analyze the timeseries data that is
output from the `IQ_prevalence.py` script.  This script is expected to be run from an IDE, such as Spyder.

