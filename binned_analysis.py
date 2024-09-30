import subprocess
import time
from multiprocessing import Pool
from tqdm import tqdm
import json
import gt_apps as my_apps
from GtApp import GtApp
expCube2= GtApp('gtexpcube2','Likelihood')
from BinnedAnalysis import *
import os
import argparse
import numpy as np

def check_paths(time_interval_name):
    # List of paths
    paths = [
        f'./data/LC_{time_interval_name}/ltcube/',
        f'./data/LC_{time_interval_name}/ccube/',
        f'./data/LC_{time_interval_name}/expcube/',
        f'./data/LC_{time_interval_name}/expmap/',
        f'./data/LC_{time_interval_name}/models/',
        f'./data/LC_{time_interval_name}/srcmap/',
        f'./data/LC_{time_interval_name}/likeresults/'
    ]

    # Check and create directories
    for path in paths:
        os.makedirs(path, exist_ok=True)  # This will create the directory if it doesn't exist
        print(f"Ensured existence of: {path}")
    pass

# Your existing generate_files, source_maps, and run_binned_likelihood functions here
def generate_files(vars):
    ####### Livetime Cube #######
    i, time_interval_name, short_name = vars
    my_apps.expCube['evfile'] = f'./data/LC_{time_interval_name}/{short_name}_{time_interval_name}_{i}.fits'
    my_apps.expCube['scfile'] = f'./data/{short_name}_SC.fits'
    my_apps.expCube['outfile'] = f'./data/LC_{time_interval_name}/ltcube/{short_name}_ltcube_{i}.fits'
    my_apps.expCube['zmax'] = 90
    my_apps.expCube['dcostheta'] = 0.025
    my_apps.expCube['binsz'] = 1
    my_apps.expCube.run()

    ####### Counts Cube #######
    my_apps.evtbin['evfile'] = f'./data/LC_{time_interval_name}/{short_name}_{time_interval_name}_{i}.fits'
    my_apps.evtbin['outfile'] = f'./data/LC_{time_interval_name}/ccube/{short_name}_ccube_{i}.fits'
    my_apps.evtbin['scfile'] = 'NONE'
    my_apps.evtbin['algorithm'] = 'CCUBE'
    my_apps.evtbin['nxpix'] = 100
    my_apps.evtbin['nypix'] = 100
    my_apps.evtbin['binsz'] = 0.2
    my_apps.evtbin['coordsys'] = 'CEL'
    my_apps.evtbin['xref'] = 49.9507
    my_apps.evtbin['yref'] = 41.5117
    my_apps.evtbin['axisrot'] = 0
    my_apps.evtbin['proj'] = 'AIT'
    my_apps.evtbin['ebinalg'] = 'FILE'
    my_apps.evtbin['ebinfile'] = 'energy_bins_gtbindef.fits'
    my_apps.evtbin.run()

    ####### Exposure Map #######
    expCube2['infile'] = f'./data/LC_{time_interval_name}/ltcube/{short_name}_ltcube_{i}.fits'
    expCube2['cmap'] = 'none'
    expCube2['outfile'] = f'./data/LC_{time_interval_name}/expmap/{short_name}_BinnedExpMap_{i}.fits'
    expCube2['irfs'] = 'P8R3_SOURCE_V3'
    expCube2['evtype'] = '3'
    expCube2['nxpix'] = 1800
    expCube2['nypix'] = 900
    expCube2['binsz'] = 0.2
    expCube2['coordsys'] = 'CEL'
    expCube2['xref'] = 49.9507
    expCube2['yref'] = 41.5117
    expCube2['axisrot'] = 0
    expCube2['proj'] = 'AIT'
    expCube2['ebinalg'] = 'FILE'
    expCube2['ebinfile'] = 'energy_bins_gtbindef.fits'
    expCube2.run()

    ####### Make model #######
    ##### Run make4FGLxml Command #####
    make4FGLxml_command = [f'make4FGLxml ./data/gll_psc_v32.xml --event_file ./data/LC_{time_interval_name}/{short_name}_{time_interval_name}_{i}.fits --free_radius 5.0 --norms_free_only True --sigma_to_free 25 --variable_free True']
    
    # Run the command using subprocess
    subprocess.run(make4FGLxml_command, shell=True, check=True)
    pass

def source_maps(vars):
    i, time_interval_name, short_name = vars
    ####### Source Map #######
    my_apps.srcMaps['expcube'] = f'./data/LC_{time_interval_name}/ltcube/{short_name}_ltcube_{i}.fits'
    my_apps.srcMaps['cmap'] = f'./data/LC_{time_interval_name}/ccube/{short_name}_ccube_{i}.fits'
    my_apps.srcMaps['srcmdl'] = f'./data/LC_{time_interval_name}/models/{short_name}_input_model_{i}.xml'
    my_apps.srcMaps['bexpmap'] = f'./data/LC_{time_interval_name}/expmap/{short_name}_BinnedExpMap_{i}.fits'
    my_apps.srcMaps['outfile'] = f'./data/LC_{time_interval_name}/srcmap/{short_name}_srcmap_{i}.fits'
    my_apps.srcMaps['irfs'] = 'P8R3_SOURCE_V3'
    my_apps.srcMaps['evtype'] = '3'
    my_apps.srcMaps.run()
    pass

def run_binned_likelihood(vars):
    i, source_name, short_name, time_interval_name, minimal_energy, maximal_energy = vars
    ####### Binned Likelihood Analysis #######
    obs = BinnedObs(
        srcMaps=f'./data/LC_{time_interval_name}/srcmap/{short_name}_srcmap_{i}.fits',
        binnedExpMap=f'./data/LC_{time_interval_name}/expmap/{short_name}_BinnedExpMap_{i}.fits',
        expCube=f'./data/LC_{time_interval_name}/ltcube/{short_name}_ltcube_{i}.fits',
        irfs='CALDB'
    )
    
    like = BinnedAnalysis(obs, f'./data/LC_{time_interval_name}/models/{short_name}_input_model_{i}.xml', optimizer='NewMinuit')
    likeobj = pyLikelihood.NewMinuit(like.logLike)
    like.fit(verbosity=0, covar=True, optObject=likeobj)
    
    log_likelihood = like.logLike.value()
    TS = like.Ts(source_name)
    convergence = likeobj.getRetCode()
    
    flux = like.flux(source_name, emin=minimal_energy, emax=maximal_energy)
    flux_error = like.fluxError(source_name, emin=minimal_energy, emax=maximal_energy)
    
    flux_data = {
        f'{time_interval_name}': i,
        'flux': flux,
        'flux_error': flux_error
    }
    
    print(f"Saving flux data: {i}")
    with open(f'./data/LC_{time_interval_name}/likeresults/flux_{time_interval_name}_{i}.json', 'w') as f:
        json.dump(flux_data, f)
    
    return (i, log_likelihood, TS, convergence)
    
def run_analysis(source_name, short_name, num_workers, num_time_intervals,
                time_interval_name, start_month, minimal_energy, maximal_energy):

    # Your existing gtbindef_energy_command and subprocess call here
    gtbindef_energy_command = [
        'gtbindef', 
        'E',
        'energy_bins_gtbindef.txt',
        'energy_bins_gtbindef.fits' ,
        'MeV']

    subprocess.run(gtbindef_energy_command, check=True)

    # Print parameters
    print("Starting analysis with the following parameters:")
    print(f"Source Name: {source_name}")
    print(f"Energy Range: {minimal_energy} - {maximal_energy} MeV")
    print(f"Time interval: {time_interval_name}")
    print(f"Number of Time Intervals: {num_time_intervals}")
    print(f"Start interval of Time Interval: {start_month}")
    print('Number of cores chosen: ', num_workers)

    time.sleep(5)  # 5 seconds of "dead" time

    print("Starting the analysis now...")

    running_args = []
    running_args_likelihood = []
    for i in range(start_month,num_time_intervals):
        running_args.append((i, time_interval_name, short_name))
        running_args_likelihood.append((i, source_name, short_name, time_interval_name, minimal_energy, maximal_energy))

    # Main analysis loop
    with Pool(num_workers) as p:
        list(tqdm(p.map(generate_files, running_args), total=num_time_intervals))
        list(tqdm(p.map(source_maps, running_args), total=num_time_intervals))
        list(tqdm(p.map(run_binned_likelihood, running_args_likelihood), total=num_time_intervals))

    print("Analysis done!")

