import gt_apps as my_apps
import json
import time
import pyLikelihood
from BinnedAnalysis import *
import matplotlib.pyplot as plt
import numpy as np
import concurrent.futures
import subprocess

#start_time = 239557417  (2008Aug05 at 00:00:00.000 UTC)MET start time

def data_filtering(source_name, start_time, end_time, emin, emax, ra, dec, time_interval_name):
    source_name_cleaned = source_name.replace(" ", "").replace(".", "dot").replace("+", "plus").replace("-", "minus")
    if time_interval_name == "month":
        time_interval = 2592000  # 30 days in seconds
    elif time_interval_name == 'week':
        time_interval = 604800
    elif time_interval_name == "3 days":
        time_interval =  259200

    print("Starting filtering with the following parameters:")
    print(f"Source Name: {source_name}")
    print(f"Time interval: {time_interval_name}")

    time.sleep(5)  # 5 seconds of "dead" time

    eventlist_command0 = [
            f'ls ./data/{source_name_cleaned}/*PH*.fits > ./data/{source_name_cleaned}/events.list'
        ]
    eventlist_command1 = [
            f'cat ./data/{source_name_cleaned}/events.list'
        ]

    # Run the command using subprocess
    subprocess.run(eventlist_command0, shell=True, check=True)
    subprocess.run(eventlist_command1, shell=True, check=True)

    print("Filtering events...")
    my_apps.filter['evclass'] = 128
    my_apps.filter['evtype'] = 3
    my_apps.filter['ra'] = ra
    my_apps.filter['dec'] = dec
    my_apps.filter['rad'] = 15
    my_apps.filter['emin'] = emin
    my_apps.filter['emax'] = emax
    my_apps.filter['zmax'] = 90
    my_apps.filter['tmin'] = start_time
    my_apps.filter['tmax'] = end_time
    my_apps.filter['infile'] = f'@./data/{source_name_cleaned}/events.list'
    my_apps.filter['outfile'] = f'./data/{source_name_cleaned}/filtered.fits'
    my_apps.filter.run()

    my_apps.maketime['scfile'] = f'./data/{source_name_cleaned}/SC.fits'
    my_apps.maketime['filter'] = '(DATA_QUAL>0)&&(LAT_CONFIG==1)'
    my_apps.maketime['roicut'] = 'no'
    my_apps.maketime['evfile'] = f'./data/{source_name_cleaned}/filtered.fits'
    my_apps.maketime['outfile'] = f'./data/{source_name_cleaned}/filtered_gti.fits'
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
        my_apps.filter['infile'] = f'./data/{source_name_cleaned}/filtered_gti.fits'
        my_apps.filter['outfile'] = f'./data/{source_name_cleaned}/LC_{time_interval_name}/{time_interval_name}_{index}.fits'
        my_apps.filter.run()
        
        # Update the current time and index
        current_time = next_time
        index += 1

    return