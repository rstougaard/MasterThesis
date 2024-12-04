import subprocess
import time
from multiprocessing import Pool
import pyLikelihood
from tqdm import tqdm
import json
import gt_apps as my_apps
from GtApp import GtApp
expCube2 = GtApp('gtexpcube2', 'Likelihood')
from BinnedAnalysis import *
import os
import glob
import xml.etree.ElementTree as ET
import numpy as np
import logging

# Function to ensure paths exist
def check_paths(source_name, time_interval_name):
    source_name_cleaned = source_name.replace(" ", "").replace(".", "dot").replace("+", "plus").replace("-", "minus")
    paths = [
        f'./data/{source_name_cleaned}/LC_{time_interval_name}/ltcube/',
        f'./data/{source_name_cleaned}/LC_{time_interval_name}/ccube/',
        f'./data/{source_name_cleaned}/LC_{time_interval_name}/expcube/',
        f'./data/{source_name_cleaned}/LC_{time_interval_name}/expmap/',
        f'./data/{source_name_cleaned}/LC_{time_interval_name}/models/',
        f'./data/{source_name_cleaned}/LC_{time_interval_name}/srcmap/',
        f'./data/{source_name_cleaned}/LC_{time_interval_name}/CountsSpectra/',
        f'./data/{source_name_cleaned}/LC_{time_interval_name}/likeresults/',
        f'./data/{source_name_cleaned}/LC_{time_interval_name}/fit_params/',
        f'./energy_bins_def/'
    ]
    for path in paths:
        os.makedirs(path, exist_ok=True)
        print(f"Ensured existence of: {path}")

# Function to read energy bins
def get_energy_bins():
    energy_bins = []
    with open('energy_bins_gtbindef.txt', 'r') as file:
        for line in file:
            line = line.strip()
            if not line or len(line.split()) != 2:
                continue
            emin, emax = map(float, line.split())
            energy_bins.append((emin, emax))
    return energy_bins

# Function to generate livetime cube
def generate_ltcube(vars):
    i, source_name, time_interval_name, ra, dec, short_name = vars
    source_name_cleaned = source_name.replace(" ", "").replace(".", "dot").replace("+", "plus").replace("-", "minus")
    my_apps.expCube['evfile'] = f'./data/{source_name_cleaned}/LC_{time_interval_name}/{time_interval_name}_{i}.fits'
    my_apps.expCube['scfile'] = f'./data/{source_name_cleaned}/SC.fits'
    my_apps.expCube['outfile'] = f'./data/{source_name_cleaned}/LC_{time_interval_name}/ltcube/ltcube_{i}.fits'
    my_apps.expCube['zmax'] = 90
    my_apps.expCube['dcostheta'] = 0.025
    my_apps.expCube['binsz'] = 1
    my_apps.expCube.run()
    pass
# Function to generate files for full spectrum
def generate_files(vars):
    ####### Livetime Cube #######
    i, source_name, time_interval_name, ra, dec, minimal_energy, maximal_energy = vars
    source_name_cleaned = source_name.replace(" ", "").replace(".", "dot").replace("+", "plus").replace("-", "minus")
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
    my_apps.evtbin['ebinfile'] = './energy_bins_def/energy_bins_gtbindef.fits'
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
    expCube2['ebinfile'] = './energy_bins_def/energy_bins_gtbindef.fits'
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
    i, source_name, time_interval_name, ra, dec, minimal_energy, maximal_energy = vars
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
    i, source_name, time_interval_name, ra, dec, minimal_energy, maximal_energy = vars
    source_name_cleaned = source_name.replace(" ", "").replace(".", "dot").replace("+", "plus").replace("-", "minus")
    ####### Binned Likelihood Analysis #######
    try:
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
                    
                    if param_name in ['alpha', 'beta', 'Eb']:  # Check if it's 'alpha' or 'beta'
                        param_data[f'{param_name}_value'] = param_value  # Store value in the dictionary
                        param_data[f'{param_name}_error'] = param_error  # Store error in the dictionary
                        param_data[f'{param_name}_scale'] = param_scale
        else:
            print("Source not found in the XML file.")

        # Assuming 'like.flux' and 'like.fluxError' provide flux and flux error for the source per full time period:
        flux_tot_value = like.flux(source_name, emin=minimal_energy, emax=maximal_energy) #before just maximal_energy
        flux_tot_error = like.fluxError(source_name, emin=minimal_energy, emax=maximal_energy)
        alpha = like.model[source_name].funcs['Spectrum'].getParam('alpha').value()
        alpha_err = like.model[source_name].funcs['Spectrum'].getParam('alpha').error()
        beta = like.model[source_name].funcs['Spectrum'].getParam('beta').value()
        beta_err = like.model[source_name].funcs['Spectrum'].getParam('beta').error()
        Eb = like.model[source_name].funcs['Spectrum'].getParam('Eb').value()
        Eb_err = like.model[source_name].funcs['Spectrum'].getParam('Eb').error()

        E = (like.energies[:-1] + like.energies[1:])/2.
        nobs = like.nobs
        # Calculate the bin widths
        
        # Save the flux data along with alpha and beta
        fit_data = {
        f'{time_interval_name}': i,
        'flux_tot_value': float(flux_tot_value),  # Ensure this is a float
        'flux_tot_error': float(flux_tot_error),   
        'alpha_value': float(alpha),  
        'alpha_error': float(alpha_err), 
        'beta_value': float(beta),  
        'beta_error': float(beta_err), 
        'Eb_value': float(Eb),  
        'Eb_error': float(Eb_err),  
        'convergence': convergence,
        'E_points': E.tolist() if isinstance(E, np.ndarray) else list(E),  # Handle ndarray or tuple
        'nobs': list(nobs),  # Convert tuple to list
    }

        
        with open(f'./data/{source_name_cleaned}/LC_{time_interval_name}/likeresults/flux_{time_interval_name}_{i}.json', 'w') as f:
            json.dump(fit_data, f, indent=4) 

    except Exception as e:
        # Catch any exception and print relevant information
        print(f"Error in iteration: time_interval {i}.")
        print(f"Exception: {str(e)}")

    print(f"Saved flux data for full spectrum {i}!")
    
    

def save_flux_fit_data(source_name_cleaned, time_interval_name, num_intervals):
    combined_fit_data = {}

    # Iterate over all time intervals to load the flux fit data
    for i in range(num_intervals):
        flux_file = f'./data/{source_name_cleaned}/LC_{time_interval_name}/likeresults/flux_{time_interval_name}_{i}.json'

        if os.path.exists(flux_file):
            # Load the existing JSON data
            with open(flux_file, 'r') as f:
                fit_data = json.load(f)

                # Add the loaded fit_data to the combined dictionary
                combined_fit_data[f"time_interval_{i}"] = fit_data

            # Delete the JSON file after loading
            os.remove(flux_file)
            print(f"Deleted JSON file: {flux_file}")

    # Save the combined fit data as a .json file
    output_directory = f'./data/{source_name_cleaned}/LC_{time_interval_name}/npy_files/'
    os.makedirs(output_directory, exist_ok=True)

    combined_json_path = os.path.join(output_directory, f'combined_fit_data_{time_interval_name}.json')
    with open(combined_json_path, 'w') as f:
        json.dump(combined_fit_data, f, indent=4)

    # Save the combined fit data as a .npy file
    combined_npy_path = os.path.join(output_directory, f'combined_fit_data_{time_interval_name}.npy')
    np.save(combined_npy_path, combined_fit_data)

    print(f"Combined fit data saved as JSON: {combined_json_path}")
    print(f"Combined fit data saved as NPY: {combined_npy_path}")

    return combined_fit_data
###############################################################################################################################################################
###############################################################################################################################################################
# Function to generate other necessary files (per energy bin)
def generate_files_per_bin(vars):
    i, source_name, time_interval_name, ra, dec, short_name, emin, emax, energy_bin_index = vars
    source_name_cleaned = source_name.replace(" ", "").replace(".", "dot").replace("+", "plus").replace("-", "minus")

    # Create energy bin definition file for this bin
    energy_bin_txt = f'./energy_bins_def/energy_bin_{i}_{energy_bin_index}.txt'
    with open(energy_bin_txt, 'w') as f:
        f.write(f'{emin}   {emax}\n')

    # Create the energy bin FITS file
    gtbindef_energy_command = [
        'gtbindef',
        'E',
        energy_bin_txt,
        f'./energy_bins_def/energy_bin_{i}_{energy_bin_index}.fits',
        'MeV']
    subprocess.run(gtbindef_energy_command, check=True)

    ####### Counts Cube #######
    my_apps.evtbin['evfile'] = f'./data/{source_name_cleaned}/LC_{time_interval_name}/{time_interval_name}_{i}.fits'
    my_apps.evtbin['outfile'] = f'./data/{source_name_cleaned}/LC_{time_interval_name}/ccube/ccube_{i}_bin_{energy_bin_index}.fits'
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
    my_apps.evtbin['ebinfile'] = f'./energy_bins_def/energy_bin_{i}_{energy_bin_index}.fits'
    my_apps.evtbin.run()

    ####### Exposure Map #######
    expCube2['infile'] = f'./data/{source_name_cleaned}/LC_{time_interval_name}/ltcube/ltcube_{i}.fits'
    expCube2['cmap'] = 'none'
    expCube2['outfile'] = f'./data/{source_name_cleaned}/LC_{time_interval_name}/expmap/BinnedExpMap_{i}_bin_{energy_bin_index}.fits'
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
    expCube2['ebinfile'] = f'./energy_bins_def/energy_bin_{i}_{energy_bin_index}.fits'
    expCube2.run()
    

    ####### Make model #######
    ##### Run make4FGLxml Command #####
    make4FGLxml_command = [
        f'make4FGLxml ./data/gll_psc_v32.xml --event_file ./data/{source_name_cleaned}/LC_{time_interval_name}/{time_interval_name}_{i}.fits '
        f'-o ./data/{source_name_cleaned}/LC_{time_interval_name}/models/input_model_{i}_bin_{energy_bin_index}.xml --free_radius 5.0 --norms_free_only True '
        f'--sigma_to_free 25 --variable_free True']
    
    subprocess.run(make4FGLxml_command, shell=True, check=True, executable='/bin/bash')

    # Modify the generated XML file
    tree = ET.parse(f'./data/{source_name_cleaned}/LC_{time_interval_name}/models/input_model_{i}_bin_{energy_bin_index}.xml')
    root = tree.getroot()

    # Look for the specific source by name
    source = root.find(f".//source[@name='{source_name}']")
    if source is not None:
        print(f"Modifying source: {source.get('name')}")

        # Find the 'spectrum' tag within this source
        spectrum = source.find('spectrum')
        if spectrum is not None:
            for param in spectrum.findall('parameter'):
                param_name = param.get('name')
                # Only modify 'alpha' parameter
                if param_name in ['alpha']:
                    print(f"Changing 'free' attribute for {param_name}")
                    param.set('free', '0') #Has to be fixed for likelihood per bin, equal 0

        # Save the modified XML back to the file
        tree.write(f'./data/{source_name_cleaned}/LC_{time_interval_name}/models/input_model_{i}_bin_{energy_bin_index}.xml', encoding='utf-8', xml_declaration=True)
    pass 

# Function to generate source maps
def source_maps_per_bin(vars):
    i, source_name, time_interval_name, ra, dec, short_name, emin, emax, energy_bin_index = vars
    source_name_cleaned = source_name.replace(" ", "").replace(".", "dot").replace("+", "plus").replace("-", "minus")
    ####### Source Map #######
    my_apps.srcMaps['expcube'] = f'./data/{source_name_cleaned}/LC_{time_interval_name}/ltcube/ltcube_{i}.fits'
    my_apps.srcMaps['cmap'] = f'./data/{source_name_cleaned}/LC_{time_interval_name}/ccube/ccube_{i}_bin_{energy_bin_index}.fits'
    my_apps.srcMaps['srcmdl'] = f'./data/{source_name_cleaned}/LC_{time_interval_name}/models/input_model_{i}_bin_{energy_bin_index}.xml'
    my_apps.srcMaps['bexpmap'] = f'./data/{source_name_cleaned}/LC_{time_interval_name}/expmap/BinnedExpMap_{i}_bin_{energy_bin_index}.fits'
    my_apps.srcMaps['outfile'] = f'./data/{source_name_cleaned}/LC_{time_interval_name}/srcmap/srcmap_{i}_bin_{energy_bin_index}.fits'
    my_apps.srcMaps['irfs'] = 'P8R3_SOURCE_V3'
    my_apps.srcMaps['evtype'] = '3'
    my_apps.srcMaps.run()
    pass

# Function to run binned likelihood analysis
def run_binned_likelihood_per_bin(vars):
    i, source_name, time_interval_name, ra, dec, short_name, emin, emax, energy_bin_index = vars
    source_name_cleaned = source_name.replace(" ", "").replace(".", "dot").replace("+", "plus").replace("-", "minus")

    try:
        obs = BinnedObs(
            srcMaps=f'./data/{source_name_cleaned}/LC_{time_interval_name}/srcmap/srcmap_{i}_bin_{energy_bin_index}.fits',
            binnedExpMap=f'./data/{source_name_cleaned}/LC_{time_interval_name}/expmap/BinnedExpMap_{i}_bin_{energy_bin_index}.fits',
            expCube=f'./data/{source_name_cleaned}/LC_{time_interval_name}/ltcube/ltcube_{i}.fits',
            irfs='CALDB'
        )

        like = BinnedAnalysis(obs, f'./data/{source_name_cleaned}/LC_{time_interval_name}/models/input_model_{i}_bin_{energy_bin_index}.xml', optimizer='NewMinuit')
        likeobj = pyLikelihood.NewMinuit(like.logLike)

        # Perform the likelihood fitting
        like.fit(verbosity=0, covar=True, optObject=likeobj)

        # Write Counts Spectra and XML files
        like.writeCountsSpectra(f"./data/{source_name_cleaned}/LC_{time_interval_name}/CountsSpectra/spectra_{i}_bin_{energy_bin_index}.fits")
        like.logLike.writeXml(f'./data/{source_name_cleaned}/LC_{time_interval_name}/fit_params/fit_{i}_bin_{energy_bin_index}.xml')

        # Calculate differential flux per energy bin
        flux_value = like.flux(source_name, emin=emin, emax=emax)
        flux_error = like.fluxError(source_name, emin=emin, emax=emax)

        # Get nobs values
        nobs = like.nobs

        # Calculate differential flux and errors
        E_bin_width = emax - emin
        dFdE = flux_value / E_bin_width
        dFdE_error = flux_error / E_bin_width
        E_av = (emin * emax) ** 0.5

        E_minus_error = E_av - emin
        E_plus_error = emax - E_av

        # Save the extracted data
        fit_data = {
            'time_interval': i,
            'int_flux': flux_value,
            'int_flux_error': flux_error,
            'emin': emin,
            'emax': emax,
            'E_av': E_av,
            'E_minus_error': E_minus_error,
            'E_plus_error': E_plus_error,
            'dFdE': dFdE,
            'dFdE_error': dFdE_error,
            'nobs': list(nobs),
        }

        output_file = f'./data/{source_name_cleaned}/LC_{time_interval_name}/likeresults/flux_{i}_bin_{energy_bin_index}.json'
        with open(output_file, 'w') as f:
            json.dump(fit_data, f, indent=4)

    except Exception as e:
        # Catch any exception and print relevant information
        print(f"Error in iteration: time_interval {i}, bin {energy_bin_index}")
        print(f"Exception: {str(e)}")


def combine_flux_data_per_time_interval(source_name_cleaned, time_interval_name, num_intervals, num_bins):
    all_intervals_combined_data = []

    # Iterate over all time intervals
    for i in range(num_intervals):
        combined_data_for_interval = []

        # Iterate over all bins for a given time interval to load data
        for bin_num in range(num_bins):
            flux_file = f'./data/{source_name_cleaned}/LC_{time_interval_name}/likeresults/flux_{i}_bin_{bin_num}.json'

            if os.path.exists(flux_file):
                with open(flux_file, 'r') as f:
                    data = json.load(f)

                    # If any crucial value is None, replace it with zeroes
                    if data.get("int_flux") is None or data.get("dFdE") is None:
                        print(f"Missing data for time interval {i} bin {bin_num}. Filling with zeroes.")
                        data = {
                            "time_interval": i,
                            "int_flux": 0.0,
                            "int_flux_error": 0.0,
                            "emin": data.get("emin", 0.0),
                            "emax": data.get("emax", 0.0),
                            'E_av': 0.0,
                            'E_minus_error': 0.0,
                            'E_plus_error': 0.0,
                            "dFdE": 0.0,
                            "dFdE_error": 0.0,
                            "nobs": 0.0
                        }

                    # Extract relevant fields from JSON and append to combined data for that interval
                    combined_data_for_interval.append({
                        "time_interval": data.get("time_interval"),
                        "int_flux": data.get("int_flux"),
                        "int_flux_error": data.get("int_flux_error"),
                        "emin": data.get("emin"),
                        "emax": data.get("emax"),
                        'E_av': data.get("E_av"),
                        'E_minus_error': data.get("E_minus_error"),
                        'E_plus_error': data.get("E_plus_error"),
                        "dFdE": data.get("dFdE"),
                        "dFdE_error": data.get("dFdE_error"),
                        "nobs": data.get("nobs") if isinstance(data.get("nobs"), (int, float)) else data.get("nobs", [0])[0]
                        })


                # Delete the JSON file after reading
                os.remove(flux_file)
                print(f"Deleted JSON file: {flux_file}")

        # Add the interval data if it has values
        if combined_data_for_interval:
            all_intervals_combined_data.append(combined_data_for_interval)

    # Calculate the summed array per bin across all time intervals
    summed_array_per_bin = []

    if all_intervals_combined_data:
        # Iterate over bins to calculate the sum
        for bin_index in range(num_bins):
            num_valid_intervals = 0
            summed_bin_data = {
                "time_interval": None,
                "int_flux": 0.0,
                "int_flux_error": 0.0,
                "emin": None,
                "emax": None,
                'E_av': None,
                'E_minus_error': None,
                'E_plus_error': None,
                "dFdE": 0.0,
                "dFdE_error": 0.0,
                "nobs": 0.0,
            }

            # Iterate over each interval to sum the values for the current bin
            for interval_data in all_intervals_combined_data:
                if bin_index < len(interval_data):
                    bin_data = interval_data[bin_index]
                    num_valid_intervals += 1

                    # Retain time_interval, emin, emax from the first valid interval
                    if summed_bin_data["time_interval"] is None:
                        summed_bin_data["time_interval"] = bin_data["time_interval"]
                        summed_bin_data["emin"] = bin_data["emin"]
                        summed_bin_data["emax"] = bin_data["emax"]
                        summed_bin_data["E_av"] = bin_data["E_av"]
                        summed_bin_data["E_minus_error"] = bin_data["E_minus_error"]
                        summed_bin_data["E_plus_error"] = bin_data["E_plus_error"]

                    # Sum the numerical values
                    for key in ["int_flux", "int_flux_error", "dFdE", "dFdE_error", "nobs"]:
                        if bin_data[key] is not None:
                            summed_bin_data[key] += bin_data[key]

            # Divide summed values by number of intervals to get the average
            if num_valid_intervals > 0:
                for key in ["int_flux", "int_flux_error", "dFdE", "dFdE_error"]:
                    summed_bin_data[key] /= num_valid_intervals

            # Append the summed data for this bin
            summed_array_per_bin.append(summed_bin_data)

    # Convert summed_array_per_bin to a structured array for easier access
    dtype = [
        ("time_interval", "O"),
        ("int_flux", "f8"),
        ("int_flux_error", "f8"),
        ("emin", "f8"),
        ("emax", "f8"),
        ("E_av", "f8"),
        ("E_minus_error", "f8"),
        ("E_plus_error", "f8"),
        ("dFdE", "f8"),
        ("dFdE_error", "f8"),
        ("nobs", "f8"),
    ]
    summed_array_per_bin_np = np.array([tuple(d.values()) for d in summed_array_per_bin], dtype=dtype)

    # Save arrays as .npy files
    output_directory = f'./data/{source_name_cleaned}/LC_{time_interval_name}/npy_files/'
    os.makedirs(output_directory, exist_ok=True)

    # Save the combined data, combined array, and summed array as .npy files
    np.save(os.path.join(output_directory, 'all_intervals_combined_data.npy'), all_intervals_combined_data)
    np.save(os.path.join(output_directory, 'all_intervals_combined_array.npy'), all_intervals_combined_data, allow_pickle=True)
    np.save(os.path.join(output_directory, 'summed_array_per_bin.npy'), summed_array_per_bin_np)

    return all_intervals_combined_data, summed_array_per_bin_np

#######################################################################################################################################
def delete_fits_and_xml_files(source_name_cleaned, time_interval_name):
    # Define the folders and file types to delete
    paths_to_delete = [
        f'./data/{source_name_cleaned}/LC_{time_interval_name}/expmap/*.fits',
        f'./data/{source_name_cleaned}/LC_{time_interval_name}/srcmap/*.fits',
        f'./data/{source_name_cleaned}/LC_{time_interval_name}/ccube/*.fits',
        f'./data/{source_name_cleaned}/LC_{time_interval_name}/models/*.xml',
    ]

    # Iterate over each path pattern and delete all matching files
    for path_pattern in paths_to_delete:
        files = glob.glob(path_pattern)
        for file in files:
            try:
                os.remove(file)
                #print(f"Deleted: {file}")
            except OSError as e:
                print(f"Error deleting file {file}: {e}")

logging.basicConfig(
    filename='error_log.txt',
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(message)s'
)

# Main function to run the analysis
def run_analysis(source_name, short_name, num_workers, num_time_intervals, time_interval_name, start_month, ra, dec, minimal_energy, maximal_energy):
    # Your existing gtbindef_energy_command and subprocess call here
    
    gtbindef_energy_command = [
        'gtbindef', 
        'E',
        'energy_bins_gtbindef.txt',
        './energy_bins_def/energy_bins_gtbindef.fits' ,
        'MeV']

    subprocess.run(gtbindef_energy_command, check=True)
    energy_bins = get_energy_bins()
    source_name_cleaned = source_name.replace(" ", "").replace(".", "dot").replace("+", "plus").replace("-", "minus")
    
    running_args_ltcube = [(i, source_name, time_interval_name, ra, dec, short_name) for i in range(start_month, num_time_intervals)]
    running_args = []
    running_args_per_bin = []
    
    for i in range(start_month, num_time_intervals):
        running_args.append((i, source_name, time_interval_name, ra, dec, minimal_energy, maximal_energy))
        
        for energy_bin_index, (emin, emax) in enumerate(energy_bins):
            running_args_per_bin.append((i, source_name, time_interval_name, ra, dec, short_name, emin, emax, energy_bin_index))
            
    
    with Pool(num_workers) as p:
        list(tqdm(p.map(generate_ltcube, running_args_ltcube), total=len(running_args_ltcube)))
        list(tqdm(p.map(generate_files, running_args), total=len(running_args)))
        list(tqdm(p.map(source_maps, running_args), total=len(running_args)))
        list(tqdm(p.map(run_binned_likelihood, running_args), total=len(running_args)))

    save_flux_fit_data(source_name_cleaned, time_interval_name, num_time_intervals)
    print("Flux fit saved!")
    #fuction that deletes everything generated in generate_files and source_maps
    
    delete_fits_and_xml_files(source_name_cleaned, time_interval_name)
    
    with Pool(num_workers) as p:
        list(tqdm(p.map(generate_files_per_bin, running_args_per_bin), total=len(running_args_per_bin)))
        list(tqdm(p.map(source_maps_per_bin, running_args_per_bin), total=len(running_args_per_bin)))
        list(tqdm(p.map(run_binned_likelihood_per_bin, running_args_per_bin), total=len(running_args_per_bin)))
    
    combine_flux_data_per_time_interval(source_name_cleaned, time_interval_name, num_time_intervals, 14)
    print("Spectral points per time interval saved!")
    delete_fits_and_xml_files(source_name_cleaned, time_interval_name)


    


