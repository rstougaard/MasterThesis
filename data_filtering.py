import gt_apps as my_apps
import json
import time
import pyLikelihood
from BinnedAnalysis import *
import matplotlib.pyplot as plt
import numpy as np
import concurrent.futures
import subprocess

# Name of the source in your XML model
source_name = '4FGL J0319.8+4130'  # Example source name in the XML model

# Define the initial and final time
start_time = 239557417  # (	2008Aug05 at 00:00:00.000 UTC )MET start time
end_time = 747645295    # MET end time

# Define the duration of each time bin (e.g., one month in seconds)
time_interval_name = 'month'

if time_interval_name == "month":
    time_interval = 2592000  # 30 days in seconds
elif time_interval_name == 'week':
    time_interval = 604800
elif time_interval_name == "3 days"
    time_interval =  259 200

print("Starting analysis with the following parameters:")
print(f"Source Name: {source_name}")
print(f"Time interval: {time_interval_name}")

time.sleep(5)  # 5 seconds of "dead" time

print("Filtering events...")
my_apps.filter['evclass'] = 128
my_apps.filter['evtype'] = 3
my_apps.filter['ra'] = 49.9507
my_apps.filter['dec'] = 41.5117
my_apps.filter['rad'] = 15
my_apps.filter['emin'] = 100
my_apps.filter['emax'] = 300000
my_apps.filter['zmax'] = 90
my_apps.filter['tmin'] = start_time
my_apps.filter['tmax'] = end_time
my_apps.filter['infile'] = '@./data/NGC1275_events.list'
my_apps.filter['outfile'] = './data/NGC1275_filtered.fits'
my_apps.filter.run()

my_apps.maketime['scfile'] = './data/NGC1275_SC.fits'
my_apps.maketime['filter'] = '(DATA_QUAL>0)&&(LAT_CONFIG==1)'
my_apps.maketime['roicut'] = 'no'
my_apps.maketime['evfile'] = './data/NGC1275_filtered.fits'
my_apps.maketime['outfile'] = './data/NGC1275_filtered_gti.fits'
my_apps.maketime.run()


# Initialize the start time and index
current_time = start_time
index = 0

# Loop over each month until the end time
while current_time < end_time:
    # Calculate the end time for the current month
    next_time = min(current_time + time_interval, end_time)
    
    # Define the filter settings with the current index
    my_apps.filter['tmin'] = current_time
    my_apps.filter['tmax'] = next_time
    my_apps.filter['infile'] = './data/NGC1275_filtered_gti.fits'
    my_apps.filter['outfile'] = f'./data/LC_{time_interval_name}/NGC1275_{time_interval_name}_{index}.fits'
    my_apps.filter.run()
    
    # Update the current time and index
    current_time = next_time
    index += 1