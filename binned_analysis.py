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
import xml.etree.ElementTree as ET
import numpy as np

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
        f'./data/{source_name_cleaned}/LC_{time_interval_name}/fit_params/'
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

# Function to generate other necessary files (per energy bin)
def generate_files(vars):
    i, source_name, time_interval_name, ra, dec, short_name, emin, emax, energy_bin_index = vars
    source_name_cleaned = source_name.replace(" ", "").replace(".", "dot").replace("+", "plus").replace("-", "minus")

    # Create energy bin definition file for this bin
    energy_bin_txt = f'energy_bin_{i}_{energy_bin_index}.txt'
    with open(energy_bin_txt, 'w') as f:
        f.write(f'{emin}   {emax}\n')

    # Create the energy bin FITS file
    gtbindef_energy_command = [
        'gtbindef',
        'E',
        energy_bin_txt,
        f'energy_bin_{i}_{energy_bin_index}.fits',
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
    my_apps.evtbin['ebinfile'] = f'energy_bin_{i}_{energy_bin_index}.fits'
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
    expCube2['ebinfile'] = f'energy_bin_{i}_{energy_bin_index}.fits'
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
                    param.set('free', '1')

        # Save the modified XML back to the file
        tree.write(f'./data/{source_name_cleaned}/LC_{time_interval_name}/models/input_model_{i}_bin_{energy_bin_index}.xml', encoding='utf-8', xml_declaration=True)
    pass 

# Function to generate source maps
def source_maps(vars):
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
def run_binned_likelihood(vars):
    i, source_name, time_interval_name, ra, dec, short_name, emin, emax, energy_bin_index = vars
    source_name_cleaned = source_name.replace(" ", "").replace(".", "dot").replace("+", "plus").replace("-", "minus")
    obs = BinnedObs(
        srcMaps=f'./data/{source_name_cleaned}/LC_{time_interval_name}/srcmap/srcmap_{i}_bin_{energy_bin_index}.fits',
        binnedExpMap=f'./data/{source_name_cleaned}/LC_{time_interval_name}/expmap/BinnedExpMap_{i}_bin_{energy_bin_index}.fits',
        expCube=f'./data/{source_name_cleaned}/LC_{time_interval_name}/ltcube/ltcube_{i}.fits',
        irfs='CALDB'
    )
    like = BinnedAnalysis(obs, f'./data/{source_name_cleaned}/LC_{time_interval_name}/models/input_model_{i}_bin_{energy_bin_index}.xml', optimizer='NewMinuit')
    likeobj = pyLikelihood.NewMinuit(like.logLike)
    #like.fit(verbosity=0, covar=True, optObject=likeobj)

    # Print parameter values before fitting to help debug bounds issues
    for param in like.model[source_name].funcs['Spectrum'].params:
        print(f"Parameter { param }" )
    
    try:
        like.fit(verbosity=0, covar=True, optObject=likeobj)
    except RuntimeError as e:
        if "Attempt to set the value outside of existing bounds" in str(e):
            print(f"Error in time interval {i} bin {energy_bin_index} for source {source_name}: Parameter value out of bounds.")
            # Optionally, handle the error here by adjusting parameters or bounds
        raise

    # Write Counts Spectra and XML files
    like.writeCountsSpectra(f"./data/{source_name_cleaned}/LC_{time_interval_name}/CountsSpectra/spectra_{i}_bin_{energy_bin_index}.fits")
    like.logLike.writeXml(f'./data/{source_name_cleaned}/LC_{time_interval_name}/fit_params/fit_{i}_bin_{energy_bin_index}.xml')

    # Load the XML file to get alpha values
    tree = ET.parse(f'./data/{source_name_cleaned}/LC_{time_interval_name}/fit_params/fit_{i}_bin_{energy_bin_index}.xml')
    root = tree.getroot()

    # Look for the specific source by name
    source = root.find(f".//source[@name='{source_name}']")
    if source is not None:
        print(f"Source: {source.get('name')}")

        spectrum = source.find('spectrum')
        alpha_value, alpha_error = None, None
        if spectrum is not None:
            for param in spectrum.findall('parameter'):
                param_name = param.get('name')
                if param_name == 'alpha':
                    alpha_value = float(param.get('value'))
                    alpha_error = float(param.get('error'))

    # Calculate differential flux per energy bin
    flux_value = like.flux(source_name, emin=emin, emax=emax)
    flux_error = like.fluxError(source_name, emin=emin, emax=emax)

    # Get nobs values
    nobs = like.nobs

    # Calculate differential flux and errors
    E_bin_width = emax - emin
    dFdE = flux_value / E_bin_width
    dFdE_error = flux_error / E_bin_width

    # Save the extracted data
    fit_data = {
        'time_interval': i,
        'int_flux': flux_value,
        'int_flux_error': flux_error,
        'emin': emin,
        'emax': emax,
        'dFdE': dFdE,
        'dFdE_error': dFdE_error,
        'nobs': list(nobs),
        'alpha_value': alpha_value,
        'alpha_error': alpha_error  # Convert to list
    }

    output_file = f'./data/{source_name_cleaned}/LC_{time_interval_name}/likeresults/flux_{i}_bin_{energy_bin_index}.json'
    with open(output_file, 'w') as f:
        json.dump(fit_data, f, indent=4)
    pass

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
                    # Extract relevant fields from JSON and append to combined data for that interval
                    combined_data_for_interval.append({
                        "time_interval": data.get("time_interval"),
                        "int_flux": data.get("int_flux"),
                        "int_flux_error": data.get("int_flux_error"),
                        "emin": data.get("emin"),
                        "emax": data.get("emax"),
                        "dFdE": data.get("dFdE"),
                        "dFdE_error": data.get("dFdE_error"),
                        "nobs": data.get("nobs", [None])[0],  # Extract the first element of nobs (assuming it has one value)
                        "alpha_value": data.get("alpha_value"),
                        "alpha_error": data.get("alpha_error")
                    })

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
                "dFdE": 0.0,
                "dFdE_error": 0.0,
                "nobs": 0.0,
                "alpha_value": 0.0,
                "alpha_error": 0.0
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

                    # Sum the numerical values
                    for key in ["int_flux", "int_flux_error", "dFdE", "dFdE_error", "nobs", "alpha_value", "alpha_error"]:
                        if bin_data[key] is not None:
                            summed_bin_data[key] += bin_data[key]

            # Divide summed values by number of intervals to get the average
            if num_valid_intervals > 0:
                for key in ["int_flux", "int_flux_error", "dFdE", "dFdE_error", "alpha_value", "alpha_error"]:
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
        ("dFdE", "f8"),
        ("dFdE_error", "f8"),
        ("nobs", "f8"),
        ("alpha_value", "f8"),
        ("alpha_error", "f8")
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

# Main function to run the analysis
def run_analysis(source_name, short_name, num_workers, num_time_intervals, time_interval_name, start_month, ra, dec):
    energy_bins = get_energy_bins()
    source_name_cleaned = source_name.replace(" ", "").replace(".", "dot").replace("+", "plus").replace("-", "minus")
    running_args_ltcube = [(i, source_name, time_interval_name, ra, dec, short_name) for i in range(start_month, num_time_intervals)]
    running_args = []
    for i in range(start_month, num_time_intervals):
        for energy_bin_index, (emin, emax) in enumerate(energy_bins):
            running_args.append((i, source_name, time_interval_name, ra, dec, short_name, emin, emax, energy_bin_index))
            

    with Pool(num_workers) as p:
        list(tqdm(p.map(generate_ltcube, running_args_ltcube), total=len(running_args_ltcube)))
        list(tqdm(p.map(generate_files, running_args), total=len(running_args)))
        list(tqdm(p.map(source_maps, running_args), total=len(running_args)))
        list(tqdm(p.map(run_binned_likelihood, running_args), total=len(running_args)))

    # After the analysis is done
    combine_flux_data_per_time_interval(source_name_cleaned, time_interval_name, num_time_intervals, 8)

    


