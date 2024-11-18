import subprocess
import time
from multiprocessing import Pool
import pyLikelihood
from tqdm import tqdm
import json
import gt_apps as my_apps
from GtApp import GtApp
expCube2= GtApp('gtexpcube2','Likelihood')
from BinnedAnalysis import *
import os
import argparse
import xml.etree.ElementTree as ET
import numpy as np

def check_paths(source_name, time_interval_name):
    source_name_cleaned = source_name.replace(" ", "").replace(".", "dot").replace("+", "plus").replace("-", "minus")
    # List of paths
    paths = [
        f'./data/{source_name_cleaned}/LC_{time_interval_name}/ltcube/',
        f'./data/{source_name_cleaned}/LC_{time_interval_name}/ccube/',
        f'./data/{source_name_cleaned}/LC_{time_interval_name}/expcube/',
        f'./data/{source_name_cleaned}/LC_{time_interval_name}/expmap/',
        f'./data/{source_name_cleaned}/LC_{time_interval_name}/models/',
        f'./data/{source_name_cleaned}/LC_{time_interval_name}/srcmap/',
        f'./data/{source_name_cleaned}/LC_{time_interval_name}/CountsSpectra/',
        f'./data/{source_name_cleaned}/LC_{time_interval_name}/likeresults/',
        f'./data/{source_name_cleaned}/LC_{time_interval_name}/fit_params/'
    ]

    # Check and create directories
    for path in paths:
        os.makedirs(path, exist_ok=True)  # This will create the directory if it doesn't exist
        print(f"Ensured existence of: {path}")
    pass
''''''
# Your existing generate_files, source_maps, and run_binned_likelihood functions here
def generate_files(vars):
    ####### Livetime Cube #######
    i, source_name, time_interval_name, ra, dec, short_name = vars
    source_name_cleaned = source_name.replace(" ", "").replace(".", "dot").replace("+", "plus").replace("-", "minus")
    my_apps.expCube['evfile'] = f'./data/{source_name_cleaned}/LC_{time_interval_name}/{time_interval_name}_{i}.fits'
    my_apps.expCube['scfile'] = f'./data/{source_name_cleaned}/SC.fits'
    my_apps.expCube['outfile'] = f'./data/{source_name_cleaned}/LC_{time_interval_name}/ltcube/ltcube_{i}.fits'
    my_apps.expCube['zmax'] = 90
    my_apps.expCube['dcostheta'] = 0.025
    my_apps.expCube['binsz'] = 1
    my_apps.expCube.run()

    ####### Counts Cube #######
    my_apps.evtbin['evfile'] = f'./data/{source_name_cleaned}/LC_{time_interval_name}/{time_interval_name}_{i}.fits'
    my_apps.evtbin['outfile'] = f'./data/{source_name_cleaned}/LC_{time_interval_name}/ccube/ccube_{i}.fits'
    my_apps.evtbin['scfile'] = 'NONE'
    my_apps.evtbin['algorithm'] = 'CCUBE'
    my_apps.evtbin['nxpix'] = 100
    my_apps.evtbin['nypix'] = 100
    my_apps.evtbin['binsz'] = 0.2
    my_apps.evtbin['coordsys'] = 'CEL'
    my_apps.evtbin['xref'] = ra
    my_apps.evtbin['yref'] = dec
    my_apps.evtbin['axisrot'] = 0
    my_apps.evtbin['proj'] = 'AIT'
    my_apps.evtbin['ebinalg'] = 'FILE'
    my_apps.evtbin['ebinfile'] = 'energy_bins_gtbindef.fits'
    my_apps.evtbin.run()

    ####### Exposure Map #######
    expCube2['infile'] = f'./data/{source_name_cleaned}/LC_{time_interval_name}/ltcube/ltcube_{i}.fits'
    expCube2['cmap'] = 'none'
    expCube2['outfile'] = f'./data/{source_name_cleaned}/LC_{time_interval_name}/expmap/BinnedExpMap_{i}.fits'
    expCube2['irfs'] = 'P8R3_SOURCE_V3'
    expCube2['evtype'] = '3'
    expCube2['nxpix'] = 1800
    expCube2['nypix'] = 900
    expCube2['binsz'] = 0.2
    expCube2['coordsys'] = 'CEL'
    expCube2['xref'] = ra
    expCube2['yref'] = dec
    expCube2['axisrot'] = 0
    expCube2['proj'] = 'AIT'
    expCube2['ebinalg'] = 'FILE'
    expCube2['ebinfile'] = 'energy_bins_gtbindef.fits'
    expCube2.run()

    ####### Make model #######
    ##### Run make4FGLxml Command #####
    make4FGLxml_command = [f'make4FGLxml ./data/gll_psc_v32.xml --event_file ./data/{source_name_cleaned}/LC_{time_interval_name}/{time_interval_name}_{i}.fits -o ./data/{source_name_cleaned}/LC_{time_interval_name}/models/input_model_{i}.xml --free_radius 5.0 --norms_free_only True --sigma_to_free 25 --variable_free True']
    
    # Run the command using subprocess
    subprocess.run(make4FGLxml_command, shell=True, check=True, executable='/bin/bash')
    # Load the specific XML file
    tree = ET.parse(f'./data/{source_name_cleaned}/LC_{time_interval_name}/models/input_model_{i}.xml')  # Replace with the actual path to your XML file
    root = tree.getroot()

    # Look for the specific source by name
    source = root.find(f".//source[@name='{source_name}']")

    if source is not None:
        print(f"Modifying source: {source.get('name')}")

        # Find the 'spectrum' tag within this source
        spectrum = source.find('spectrum')
        
        if spectrum is not None:
            # Loop through the parameters within the spectrum
            for param in spectrum.findall('parameter'):
                param_name = param.get('name')  # Get the parameter name
                
                # Only modify 'alpha' and 'beta' parameters
                if param_name in ['alpha']:
                    print(f"Changing 'free' attribute for {param_name}")
                    param.set('free', '1')  # Set 'free' attribute to '1' 
        
        # Save the modified XML back to the file
        tree.write(f'./data/{source_name_cleaned}/LC_{time_interval_name}/models/input_model_{i}.xml', encoding='utf-8', xml_declaration=True)
    pass

def source_maps(vars):
    i, source_name, time_interval_name, ra, dec, short_name = vars
    source_name_cleaned = source_name.replace(" ", "").replace(".", "dot").replace("+", "plus").replace("-", "minus")
    ####### Source Map #######
    my_apps.srcMaps['expcube'] = f'./data/{source_name_cleaned}/LC_{time_interval_name}/ltcube/ltcube_{i}.fits'
    my_apps.srcMaps['cmap'] = f'./data/{source_name_cleaned}/LC_{time_interval_name}/ccube/ccube_{i}.fits'
    my_apps.srcMaps['srcmdl'] = f'./data/{source_name_cleaned}/LC_{time_interval_name}/models/input_model_{i}.xml'
    my_apps.srcMaps['bexpmap'] = f'./data/{source_name_cleaned}/LC_{time_interval_name}/expmap/BinnedExpMap_{i}.fits'
    my_apps.srcMaps['outfile'] = f'./data/{source_name_cleaned}/LC_{time_interval_name}/srcmap/srcmap_{i}.fits'
    my_apps.srcMaps['irfs'] = 'P8R3_SOURCE_V3'
    my_apps.srcMaps['evtype'] = '3'
    my_apps.srcMaps.run()
    pass

def run_binned_likelihood(vars):
    i, source_name, short_name, time_interval_name, minimal_energy, maximal_energy = vars
    source_name_cleaned = source_name.replace(" ", "").replace(".", "dot").replace("+", "plus").replace("-", "minus")
    ####### Binned Likelihood Analysis #######
    obs = BinnedObs(
        srcMaps=f'./data/{source_name_cleaned}/LC_{time_interval_name}/srcmap/srcmap_{i}.fits',
        binnedExpMap=f'./data/{source_name_cleaned}/LC_{time_interval_name}/expmap/BinnedExpMap_{i}.fits',
        expCube=f'./data/{source_name_cleaned}/LC_{time_interval_name}/ltcube/ltcube_{i}.fits',
        irfs='CALDB'
    )
    
    like = BinnedAnalysis(obs, f'./data/{source_name_cleaned}/LC_{time_interval_name}/models/input_model_{i}.xml', optimizer='NewMinuit')
    likeobj = pyLikelihood.NewMinuit(like.logLike)
    like.fit(verbosity=0, covar=True, optObject=likeobj)
    
    log_likelihood = like.logLike.value()
    TS = like.Ts(source_name)
    convergence = likeobj.getRetCode()

    like.writeCountsSpectra(f"./data/{source_name_cleaned}/LC_{time_interval_name}/CountsSpectra/spectra_{i}.fits") 
    like.logLike.writeXml(f'./data/{source_name_cleaned}/LC_{time_interval_name}/fit_params/fit_{i}.xml')
    tree = ET.parse(f'./data/{source_name_cleaned}/LC_{time_interval_name}/fit_params/fit_{i}.xml')
    root = tree.getroot()

    # Look for the specific source by name
    source = root.find(f".//source[@name='{source_name}']")

    if source is not None:
        print(f"Source: {source.get('name')}")

        # Initialize a dictionary to store parameter values and errors
        param_data = {}

        # Find the 'spectrum' tag within this source
        spectrum = source.find('spectrum')
        
        if spectrum is not None:
            # Loop through the parameters within the spectrum
            for param in spectrum.findall('parameter'):
                param_name = param.get('name')  # Get the parameter name
                param_value = param.get('value')  # Get the parameter value
                param_error = param.get('error')  # Get the parameter error (if available)
                param_scale = param.get('scale')
                
                if param_name in ['norm','alpha', 'beta', 'Eb']:  # Check if it's 'alpha' or 'beta'
                    param_data[f'{param_name}_value'] = param_value  # Store value in the dictionary
                    param_data[f'{param_name}_error'] = param_error  # Store error in the dictionary
                    param_data[f'{param_name}_scale'] = param_scale
    else:
        print("Source not found in the XML file.")

    # Assuming 'like.flux' and 'like.fluxError' provide flux and flux error for the source per full time period:
    flux_tot_value = like.flux(source_name, emin=minimal_energy, emax=maximal_energy) #before just maximal_energy
    flux_tot_error = like.fluxError(source_name, emin=minimal_energy, emax=maximal_energy)
    N0 = like.model[source_name].funcs['Spectrum'].getParam('norm').value()
    N0_err = like.model[source_name].funcs['Spectrum'].getParam('norm').error()
    #N0_scale = like.model[source_name].funcs['Spectrum'].getParam('norm').scale()
    alpha = like.model[source_name].funcs['Spectrum'].getParam('alpha').value()
    alpha_err = like.model[source_name].funcs['Spectrum'].getParam('alpha').error()

    #Flux count per energy:
    flux_per_bin = []
    flux_error_per_bin = []
    emin_edges = []
    emax_edges = []
    with open('energy_bins_gtbindef.txt', 'r') as file:
        for line in file:
            # Strip leading/trailing whitespace
            line = line.strip()
            
            # Skip empty lines or lines with fewer than two values
            if not line or len(line.split()) != 2:
                continue  # Skip this line and go to the next
            
            # Parse the minimum and maximum energy values from each line
            emin, emax = map(float, line.split())
            emin_edges.append(emin)
            emax_edges.append(emax)

            # Call the like.flux function with the source_name, emin, and emax
            flux_value = float(like.flux(source_name, emin=emin, emax=emax))
            flux_error_value = float(like.fluxError(source_name, emin=emin, emax=emax))

            # Append the flux value to the flux_per_bin list
            flux_per_bin.append(flux_value)
            flux_error_per_bin.append(flux_error_value)
 
    E = (like.energies[:-1] + like.energies[1:])/2.
    nobs = like.nobs
    # Calculate the bin widths
    E_bin_widths = like.energies[1:] - like.energies[:-1]
    E_bin_errors = E_bin_widths / 2

    #calculate the differential flux
    differential_flux_per_bin = flux_per_bin/(emax_edges - emin_edges)
    differential_flux_per_bin_error = flux_error_per_bin/(emax_edges - emin_edges)

    # Save the flux data along with alpha and beta
    fit_data = {
    f'{time_interval_name}': i,
    'flux_tot_value': float(flux_tot_value),  # Ensure this is a float
    'flux_tot_error': float(flux_tot_error),  
    'norm': float(N0),  
    'norm_error': float(N0_err),  
    'norm_scale': param_data.get('norm_scale', None),  
    'alpha_value': float(alpha),  
    'alpha_error': float(alpha_err),  
    'convergence': convergence,
    'emin_egdes': emin_edges,
    'emax_edges': emax_edges,
    'E_points': E.tolist() if isinstance(E, np.ndarray) else list(E),  # Handle ndarray or tuple
    'E_bin_errors': E_bin_errors.tolist() if isinstance(E_bin_errors, np.ndarray) else list(E_bin_errors),
    'flux_per_bin': flux_per_bin,
    'flux_error_per_bin': flux_error_per_bin,
    'nobs': list(nobs),  # Convert tuple to list
    'dFdE': differential_flux_per_bin,
    'dFdE_error': differential_flux_per_bin_error
}


    print(f"Saving flux data: {i}")
    with open(f'./data/{source_name_cleaned}/LC_{time_interval_name}/likeresults/flux_{time_interval_name}_{i}.json', 'w') as f:
        json.dump(fit_data, f, indent=4) 
    
    return (i, log_likelihood, TS, convergence)
    
def run_analysis(source_name, short_name, num_workers, num_time_intervals,
                time_interval_name, start_month, minimal_energy, maximal_energy, ra, dec):

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
        running_args.append((i, source_name, time_interval_name, ra, dec, short_name))
        running_args_likelihood.append((i, source_name, short_name, time_interval_name, minimal_energy, maximal_energy))

    # Main analysis loop
    with Pool(num_workers) as p:
        list(tqdm(p.map(generate_files, running_args), total=num_time_intervals))
        list(tqdm(p.map(source_maps, running_args), total=num_time_intervals))
        list(tqdm(p.map(run_binned_likelihood, running_args_likelihood), total=num_time_intervals))

    print("Analysis done!")

