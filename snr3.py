import gt_apps as my_apps
from GtApp import GtApp
expCube2 = GtApp('gtexpcube2', 'Likelihood')
from astropy.io import fits
import numpy as np
import os
import json
import subprocess
import xml.etree.ElementTree as ET
from multiprocessing import Pool
import glob
import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt
from operate_gtis import *
from astropy.table import Table
import pyLikelihood
from BinnedAnalysis import *
import multiprocessing
import shlex
import shutil
import sys
general_path_for_slurm = "/groups/pheno/sqd515/MasterThesis"
edisp_bins = -3
# Function to ensure paths exist
def check_paths(source_name, method, number_of_bins):
    source_name_cleaned = source_name.replace(" ", "").replace(".", "dot").replace("+", "plus").replace("-", "minus")
    paths = [
        f'{general_path_for_slurm}/data/{source_name_cleaned}/{method}/',
        f'{general_path_for_slurm}/data/{source_name_cleaned}/{method}/ltcube/',
        f'{general_path_for_slurm}/data/{source_name_cleaned}/{method}/ccube/',
        f'{general_path_for_slurm}/data/{source_name_cleaned}/{method}/expcube/',
        f'{general_path_for_slurm}/data/{source_name_cleaned}/{method}/expmap/',
        f'{general_path_for_slurm}/data/{source_name_cleaned}/{method}/models/',
        f'{general_path_for_slurm}/data/{source_name_cleaned}/{method}/srcmap/',
        f'{general_path_for_slurm}/data/{source_name_cleaned}/{method}/CountsSpectra/',
        f'{general_path_for_slurm}/data/{source_name_cleaned}/{method}/likeresults/',
        f'{general_path_for_slurm}/data/{source_name_cleaned}/{method}/fit_params/',
        f'{general_path_for_slurm}/energy_bins_def/{number_of_bins}/',
        f'{general_path_for_slurm}/fit_results/'
    ]
    for path in paths:
        os.makedirs(path, exist_ok=True)
        #print(f"Ensured existence of: {path}")

def filtering(vars, snrratios=None, time_intervals=None):
    # Extract variables
    source_name, ra, dec, method, specin, _, _, minimal_energy, maximal_energy = vars
    source_name_cleaned = source_name.replace(" ", "").replace(".", "dot").replace("+", "plus").replace("-", "minus")
    gt = my_apps
    evc = 128
    convt = 3
    roi = 1.
    roi_maps = 10*2**0.5
    
    #tmp_evlist = f'@./data/{source_name_cleaned}/events.list'
    tmp_gti = f'{general_path_for_slurm}/data/{source_name_cleaned}/temp_git.fits'
    gtifilter = '(DATA_QUAL>0)&&(LAT_CONFIG==1)'
    #sc = f'./data/{source_name_cleaned}/SC.fits'
    tmp_evlist = f"{general_path_for_slurm}/weekly_LAT_files/weekly/photon/lat_alldata.fits"
    sc = f"{general_path_for_slurm}/mission/spacecraft/lat_spacecraft_merged.fits" 

    gti = f'{general_path_for_slurm}/data/{source_name_cleaned}/gti.fits'
    ebinfile_txt = f'{general_path_for_slurm}/energy_7bins_gtbindef.txt '

    # Ensure either snrratios or time_intervals is provided based on method
    if method == "SNR" and not snrratios:
        raise ValueError("snrratios must be provided when method is 'SNR'")
    if method == "LIN" and not time_intervals:
        raise ValueError("time_intervals must be provided when method is 'LIN'")

    # Determine the loop items based on the method
    loop_items = snrratios if method == "SNR" else time_intervals

    for loop_item in loop_items:
        print(f"Processing {method}: {loop_item}")

        # Set the lc filename based on the method
        if method == 'SNR':
            lc = f'{general_path_for_slurm}/data/{source_name_cleaned}/{method}/lc_snr{loop_item}.fits'
            colors = ['blue', 'orange', 'green']
        elif method == 'LIN':
            lc = f'{general_path_for_slurm}/data/{source_name_cleaned}/{method}/lc_{loop_item}.fits'
            colors = ['purple', 'brown']

        # Filter selecting events by time
        if not os.path.exists(gti):
            print('GTSELECT started!')
            gt.filter['evclass'] = evc
            gt.filter['evtype'] = convt
            gt.filter['ra'] = ra
            gt.filter['dec'] = dec
            gt.filter['rad'] = roi_maps
            gt.filter['emin'] = minimal_energy
            gt.filter['emax'] = maximal_energy
            gt.filter['zmax'] = 90
            gt.filter['tmin'] = 239557417
            gt.filter['tmax'] = 754704006
            gt.filter['infile'] = tmp_evlist
            gt.filter['outfile'] = tmp_gti
            gt.filter.run()  # Run GTSELECT
            print('GTSELECT finished!')

            print('GTMKTIME start')
            gt.maketime['scfile'] = sc
            gt.maketime['filter'] = gtifilter
            gt.maketime['roicut'] = 'yes'
            gt.maketime['evfile'] = tmp_gti
            gt.maketime['outfile'] = gti
            gt.maketime.run()
            try:
                os.remove(tmp_gti)
            except Exception as e:
                print(f"Error removing tmp_gti: {e}")
            print('done!')
        else:
            print(f'{gti} file exists!')

        if(not os.path.exists( lc )):
            # Sorting events for SNR method
            if method == 'SNR':
                print('Sorting event file by time...')
                with fits.open(gti, 'update') as f:
                    data = f[1].data
                    order = np.argsort(data['TIME'])
                    for kk in data.names:
                        data[kk] = data[kk][order]
                    f[1].data = data

            
            # Create light curve
            print('Creating LC')
            always_redo_exposure = True
            gt.evtbin['evfile'] = gti
            gt.evtbin['outfile'] = lc
            gt.evtbin['scfile'] = sc
            gt.evtbin['algorithm'] = 'LC'
            gt.evtbin['tbinalg'] = method
            gt.evtbin['tstart'] = 239557417
            gt.evtbin['tstop'] = 754704006
            gt.evtbin['emin'] = minimal_energy
            gt.evtbin['emax'] = maximal_energy
            gt.evtbin['ebinalg'] = "NONE"
            gt.evtbin['ebinfile'] = "NONE"

            if method == 'LIN':
                if loop_item == "month":
                    tbin = 86400 * 30
                elif loop_item == "week":
                    tbin = 86400 * 7
                gt.evtbin['dtime'] = tbin
                gt.evtbin.run()
            elif method == 'SNR':
                gt.evtbin['snratio'] = loop_item
                gt.evtbin['lcemin'] = minimal_energy
                gt.evtbin['lcemax'] = maximal_energy
                gt.evtbin.run()

            print(f'LC created for {method}: {loop_item}')

            calc_exposure = True
            with fits.open(lc) as f:
                if('EXPOSURE' in f[1].data.names): calc_exposure=False

            if(calc_exposure or always_redo_exposure):
                print('Launching gtexposure for ',lc)
                gtexposure = my_apps.GtApp('gtexposure')
                gtexposure['infile'] = lc
                gtexposure['scfile'] = sc
                gtexposure['irfs'] = 'CALDB'
                gtexposure['specin'] = -specin
                gtexposure['apcorr'] = 'yes' #change this, if you are sure
                gtexposure['enumbins'] = 30
                gtexposure['emin'] = minimal_energy
                gtexposure['emax'] = maximal_energy
                gtexposure['ra'] = ra
                gtexposure['dec'] = dec
                gtexposure['rad'] = roi
                gtexposure.run()
            else:
                print('EXPOSURE column already exists!')
                print('If you want to re-create it, launch with always_redo_exposure=True')
        else:
            print(f'{lc} file exists!')

    initial_means = []
    final_means = []
    final_thresholds = []
    valid_intervals_list = []
    invalid_intervals_list = []

    for loop_item, color in zip(loop_items, colors):
        
        print(f"Finding flares for: {method}: {loop_item}")

        # Set the lc filename based on the method
        if method == 'SNR':
            lc = f'{general_path_for_slurm}/data/{source_name_cleaned}/{method}/lc_snr{loop_item}.fits'
            output_file_flares = f'{general_path_for_slurm}/data/{source_name_cleaned}/{method}/flare_intervals_snr{loop_item}.txt'
            plot_file = f'{general_path_for_slurm}/data/{source_name_cleaned}/{method}/lc_snr{loop_item}.png'
            tmp_gti_noflares = f'{general_path_for_slurm}/data/{source_name_cleaned}/{method}/temp_git_snr{loop_item}.fits'
            gti_noflares = f'{general_path_for_slurm}/data/{source_name_cleaned}/{method}/gti_noflares_snr{loop_item}.fits'
        elif method == 'LIN':
            lc = f'{general_path_for_slurm}/data/{source_name_cleaned}/{method}/lc_{loop_item}.fits'
            output_file_flares = f'{general_path_for_slurm}/data/{source_name_cleaned}/{method}/flare_intervals_{loop_item}.txt'
            plot_file = f'{general_path_for_slurm}/data/{source_name_cleaned}/{method}/lc_{loop_item}.png'
            tmp_gti_noflares = f'{general_path_for_slurm}/data/{source_name_cleaned}/{method}/temp_git_{loop_item}.fits'
            gti_noflares = f'{general_path_for_slurm}/data/{source_name_cleaned}/{method}/gti_noflares_{loop_item}.fits'                

        if not os.path.exists(gti_noflares):
            ###### Here the BAD TIME intervals (flares) have to be found ######
            f_bin = fits.open(lc)
            bin_data = f_bin[1].data

            # Extract data
            X_bin  = bin_data['TIME']
            exposure = bin_data['EXPOSURE']
            ε = 1e-10
            safe_exposure = np.where(exposure > 0, exposure, ε)

            Y_bin       = bin_data['COUNTS']  / safe_exposure  # Flux
            x_error_bin = bin_data['TIMEDEL'] / 2
            y_error_bin = bin_data['ERROR']   / safe_exposure

            # Initialize filtering
            filtered_Y = Y_bin.copy()
            filtered_mask = np.full(Y_bin.shape, True)
            thresholds = []
            round_means = []
            removed_points = []

            while True:
                mean = np.mean(filtered_Y[filtered_mask])
                threshold = mean + 2 * np.std(filtered_Y[filtered_mask])
                thresholds.append(threshold)
                round_means.append(mean)

                round_removed_mask = filtered_Y > threshold
                new_filtered_mask = filtered_mask & ~round_removed_mask
                removed_points.append((X_bin[round_removed_mask], filtered_Y[round_removed_mask]))

                if np.array_equal(new_filtered_mask, filtered_mask):
                    break

                filtered_mask = new_filtered_mask

            # Save final results for this file
            initial_means.append(round_means[0])
            final_means.append(round_means[-1])
            final_thresholds.append(thresholds[-1])
            valid_intervals_list.append(np.sum(filtered_mask))
            invalid_intervals_list.append(len(filtered_mask) - np.sum(filtered_mask))

            # Save flare intervals to a text file
            flare_intervals_start = X_bin[~filtered_mask] - x_error_bin[~filtered_mask]
            flare_intervals_stop = X_bin[~filtered_mask] + x_error_bin[~filtered_mask]
            flare_intervals = np.column_stack((flare_intervals_start, flare_intervals_stop))

            
            np.savetxt(output_file_flares, flare_intervals, delimiter=' ')
            print(f"File saved as: {output_file_flares}")

                # Create a new figure for this file
            plt.figure(figsize=(10, 6))
            plt.rcParams["font.family"] = "serif"
            plt.rcParams["mathtext.fontset"] = "cm"

            # Plot original data
            plt.errorbar(X_bin, Y_bin, xerr=x_error_bin, yerr=y_error_bin, fmt='o', capsize=5, color=color, alpha=0.3, label=f'Data')

            # Highlight removed points
            for i, (removed_x, removed_y) in enumerate(removed_points):
                plt.scatter(removed_x, removed_y, color='black', edgecolors='black')

            # Plot means and thresholds for each round
            plt.axhline(round_means[0], color='grey', linestyle='--', linewidth=3, alpha=1, label=f'Mean unfiltered')
            plt.axhline(thresholds[0], color='grey', linestyle='-', linewidth=3, alpha=1, label=f'Threshold unfiltered')
            plt.axhline(round_means[-1], color='black', linestyle='--', linewidth=3, alpha=1, label=f'Mean round {len(round_means)}')
            plt.axhline(thresholds[-1], color='black', linestyle='-', linewidth=3, alpha=1, label=f'Threshold round {len(round_means)}')

            # Customize plot
            plt.ylabel('Flux [photons/cm²/s]',fontsize=18)
            plt.xlabel('Time [s]', fontsize=18)
            plt.title(f'Lightcurve for {method}: {loop_item}', fontsize=20)
            plt.xscale('log')
            plt.yscale('log')
            plt.grid(True, which="both", linestyle="--", linewidth=0.5)
            plt.xticks(fontsize=16)
            plt.yticks(fontsize=16)

            # Move the legend outside the plot
            plt.legend(
                fontsize=16,
                ncol=1,  # Number of columns in the legend
                loc='upper left',  # Position the legend to the left center of the bounding box
                frameon=True,  # Add a box around the legend
            )

            # Save the plot or display it
            
            plt.tight_layout()  # Adjust layout to prevent clipping
            plt.savefig(plot_file, bbox_inches='tight', dpi=300)  # Save with adjusted bounding box
            print(f"Plot saved as: {plot_file}")


            ###### Here the BAD TIME intervals (flares) have to be removed from the original raw events ######
            
            print( 'GTSELECT started!' )
            gt.filter['evclass'] = evc
            gt.filter['evtype']=convt
            gt.filter['ra'] = ra
            gt.filter['dec'] = dec
            gt.filter['rad'] = roi_maps
            gt.filter['emin'] = minimal_energy
            gt.filter['emax'] = maximal_energy
            gt.filter['zmax'] = 90
            gt.filter['tmin'] = 239557417
            gt.filter['tmax'] = 754704006
            gt.filter['infile'] = tmp_evlist
            gt.filter['outfile'] = tmp_gti_noflares
            gt.filter.run() #run GTSELECT
            print( 'GTSELCT finished!' )
            ############################## modify GTIs ##############################
            # here apoastra was a text file with 2 columns -- start stop times. Only those intervals will be used to extract spectrum
            # you could use method='out' to exclude some intervals, i.e. extract spectrum from everything *except* these intervals
            # times in the file can be either in MJDs or in Fermi seconds (use times_in_mjds=False)

            # Check if the text file (with additional GTI intervals) is non-empty:
            if os.path.exists(output_file_flares) and os.path.getsize(output_file_flares) > 0:
                # Process the GTIs from the text file and update the FITS file
                UpdateGTIs(tmp_gti_noflares, output_file_flares, method='out', times_in_mjd=False)
                
                ############### actual filtering according to updated GTIs ###############
                print('GTMKTIME start')
                gt.maketime['scfile'] = sc
                gt.maketime['filter'] = gtifilter
                gt.maketime['roicut'] = 'no'
                gt.maketime['evfile'] = tmp_gti_noflares
                gt.maketime['outfile'] = gti_noflares
                gt.maketime.run()
                
                try:
                    os.remove(tmp_gti_noflares)
                except Exception as e:
                    print(f"Error removing tmp_gti: {e}")
                print('done!')
            else:
                # The text file is empty; flag this condition and simply use the temporary GTI file as the final file.
                print(f'The text file {output_file_flares} is empty. Using tmp_gti_noflares as gti_noflares.')
                shutil.copy(tmp_gti_noflares, gti_noflares)

        else:
            print(f'{gti_noflares} file exists!')



    # Print results for all light curves
    for item, initial_mean, final_mean, final_threshold, valid_intervals, invalid_intervals in zip(
        loop_items, initial_means, final_means, final_thresholds, valid_intervals_list, invalid_intervals_list
    ):
        print(f"{item}: Initial Mean {initial_mean:.2e} |Final Mean {final_mean:.2e} | Final Threshold {final_threshold:.2e} | "
            f"Valid intervals: {valid_intervals} | Invalid intervals: {invalid_intervals}")

#####################
def get_gti_bin(vars, snrratios=None, time_intervals=None):
    # Extract variables
    source_name, ra, dec, method, specin, _, _, minimal_energy, maximal_energy = vars
    source_name_cleaned = source_name.replace(" ", "").replace(".", "dot").replace("+", "plus").replace("-", "minus")
    gt = my_apps
    ebinfile_txt = f'{general_path_for_slurm}/energy_7bins_gtbindef.txt'
    evc = 128
    convt = 3
    roi = 10*2**0.5
    #evlist = f'@./data/{source_name_cleaned}/events.list'
    gtifilter = '(DATA_QUAL>0)&&(LAT_CONFIG==1)'
    #sc = f'./data/{source_name_cleaned}/SC.fits'
    evlist = f"{general_path_for_slurm}/weekly_LAT_files/weekly/photon/lat_alldata.fits"
    sc = f"{general_path_for_slurm}/mission/spacecraft/lat_spacecraft_merged.fits" 
    # Determine the loop items based on the method
    if method == "SNR":
        loop_items = snrratios
    elif method == "LIN":
        loop_items = time_intervals
    else:
        loop_items = "NONE"  # No looping for the "NONE" method

    # If there is nothing to loop over, handle the "NONE" method directly
    if method == "NONE":
        with open(f'{ebinfile_txt}', 'r') as file:
                for line in file:
                    line = line.strip()
                    if not line or len(line.split()) != 2:
                        continue
                    emin_float, emax_float = map(float, line.split())

                    # Convert the values to integers for the filename
                    emin = int(emin_float)
                    emax = int(emax_float)
                    temp_gti_noflares_bin = f'{general_path_for_slurm}/data/{source_name_cleaned}/{method}/temp_gti_{emin}_{emax}.fits'
                    gti_noflares_bin = f'{general_path_for_slurm}/data/{source_name_cleaned}/{method}/gti_{emin}_{emax}.fits'

                    print(f"Processing {method}: {emin_float}MeV - {emax_float}MeV")                        
                    
                    #Make spectral points per method per loop_item
                    if not os.path.exists(gti_noflares_bin):
                        print( 'Making spectral points!' )
                        print( 'GTSELECT started!' )
                        gt.filter['evclass'] = evc
                        gt.filter['evtype']=convt
                        gt.filter['ra'] = ra
                        gt.filter['dec'] = dec
                        gt.filter['rad'] = roi
                        gt.filter['emin'] = emin
                        gt.filter['emax'] = emax
                        gt.filter['zmax'] = 90
                        gt.filter['tmin'] = 239557417
                        gt.filter['tmax'] = 754704006
                        gt.filter['infile'] = evlist
                        gt.filter['outfile'] = temp_gti_noflares_bin
                        gt.filter.run() #run GTSELECT
                        print( 'GTSELCT finished!' )

                        print( 'GTMKTIME start' )
                        gt.maketime['scfile'] = sc
                        gt.maketime['filter'] = gtifilter
                        gt.maketime['roicut'] = 'no'
                        gt.maketime['evfile'] = temp_gti_noflares_bin
                        gt.maketime['outfile'] = gti_noflares_bin
                        gt.maketime.run()
                        try:
                            os.remove(temp_gti_noflares_bin)
                        except Exception as e:
                            print(f"Error removing tmp_gti: {e}")
                        print('done!')
                    else:
                        print(f'{gti_noflares_bin} file exists!')
      
       
    else:
        for loop_item in loop_items:
            with open(f'{ebinfile_txt}', 'r') as file:
                for line in file:
                    line = line.strip()
                    if not line or len(line.split()) != 2:
                        continue
                    emin_float, emax_float = map(float, line.split())

                    # Convert the values to integers for the filename
                    emin = int(emin_float)
                    emax = int(emax_float)
                    print(f"Processing {method}: {loop_item} {emin_float}MeV - {emax_float}MeV")

                    if method == 'SNR':
                        temp_gti_noflares_bin = f'{general_path_for_slurm}/data/{source_name_cleaned}/{method}/temp_gti_noflares_snr{loop_item}_{emin}_{emax}.fits'
                        gti_noflares_bin = f'{general_path_for_slurm}/data/{source_name_cleaned}/{method}/gti_noflares_snr{loop_item}_{emin}_{emax}.fits'
                        output_file_flares = f'{general_path_for_slurm}/data/{source_name_cleaned}/{method}/flare_intervals_snr{loop_item}.txt'
                        
                    elif method == 'LIN':
                        temp_gti_noflares_bin = f'{general_path_for_slurm}/data/{source_name_cleaned}/{method}/temp_gti_noflares_{loop_item}_{emin}_{emax}.fits'
                        gti_noflares_bin = f'{general_path_for_slurm}/data/{source_name_cleaned}/{method}/gti_noflares_{loop_item}_{emin}_{emax}.fits'
                        output_file_flares = f'{general_path_for_slurm}/data/{source_name_cleaned}/{method}/flare_intervals_{loop_item}.txt'
                        
                    
                    if not os.path.exists(gti_noflares_bin):
                        print( 'Making spectral points!' )
                        print( 'GTSELECT started!' )
                        gt.filter['evclass'] = evc
                        gt.filter['evtype']=convt
                        gt.filter['ra'] = ra
                        gt.filter['dec'] = dec
                        gt.filter['rad'] = roi
                        gt.filter['emin'] = emin
                        gt.filter['emax'] = emax
                        gt.filter['zmax'] = 90
                        gt.filter['tmin'] = 239557417
                        gt.filter['tmax'] = 754704006
                        gt.filter['infile'] = evlist
                        gt.filter['outfile'] = temp_gti_noflares_bin
                        gt.filter.run() #run GTSELECT
                        print( 'GTSELCT finished!' )
                        if os.path.exists(output_file_flares) and os.path.getsize(output_file_flares) > 0:
                            UpdateGTIs(temp_gti_noflares_bin, output_file_flares, method='out', times_in_mjd=False)
                        else:
                            # The text file is empty; flag this condition and simply use the temporary GTI file as the final file.
                            print(f'The text file {output_file_flares} is empty. No UpdateGTI needed.')

                        print( 'GTMKTIME start' )
                        gt.maketime['scfile'] = sc
                        gt.maketime['filter'] = gtifilter
                        gt.maketime['roicut'] = 'no'
                        gt.maketime['evfile'] = temp_gti_noflares_bin
                        gt.maketime['outfile'] = gti_noflares_bin
                        gt.maketime.run()
                        try:
                            os.remove(temp_gti_noflares_bin)
                        except Exception as e:
                            print(f"Error removing tmp_gti: {e}")
                        print('done!')
                    else:
                        print(f'{gti_noflares_bin} file exists!')

            
##################################################################################
##################################################################################
def modify_and_save(tree, source_name, method, loop_item, emin, emax):
    source_name_cleaned = source_name.replace(" ", "").replace(".", "dot").replace("+", "plus").replace("-", "minus")
    if method == "NONE":
        output_dir = f'{general_path_for_slurm}/data/{source_name_cleaned}/{method}/models/'
    elif method == "SNR":
        output_dir = f'{general_path_for_slurm}/data/{source_name_cleaned}/{method}/models/'
    elif method == "LIN":
        output_dir = f'{general_path_for_slurm}/data/{source_name_cleaned}/{method}/models/'

    """
    Modifies the XML tree to create three versions of the file: free_alpha, free_beta, free_alpha_beta.

    Args:
        tree (ElementTree): The XML tree object.
        output_dir (str): The directory path to save the modified XML files.
        source_name (str): The name of the source to modify.
    """
    root = tree.getroot()

    # Look for the specific source by name
    source = root.find(f".//source[@name='{source_name}']")

    if source is not None:
        print(f"Found source: {source.get('name')}")

        # Find the 'spectrum' tag within this source
        spectrum = source.find('spectrum')

        if spectrum is not None:
            # Define file suffixes and corresponding parameters to modify
            modifications = {
                "free_alpha": ["alpha"],
                "free_beta": ["beta"],
                "free_alpha_beta": ["alpha", "beta"]
            }

            # Create a modified XML file for each case
            for suffix, params in modifications.items():
                # Create a copy of the tree for modification
                modified_tree = ET.ElementTree(root)
                
                # Modify the copied tree
                for param in spectrum.findall('parameter'):
                    if param.get('name') in params:
                        print(f"Setting 'free' attribute for {param.get('name')} in {suffix}")
                        param.set('free', '1')  # Set 'free' attribute to '1'

                # Save the modified tree to a new file
                if method == "NONE":
                    output_path = os.path.join(output_dir, f"input_model_{suffix}_{emin}_{emax}.xml")
                elif method == "SNR":
                    output_path = os.path.join(output_dir, f"input_model_snr{loop_item}_{suffix}_{emin}_{emax}.xml")
                elif method == "LIN":
                   output_path = os.path.join(output_dir, f"input_model_{loop_item}_{suffix}_{emin}_{emax}.xml")

                modified_tree.write(output_path, encoding='utf-8', xml_declaration=True)
                print(f"Modified file saved to: {output_path}")

        else:
            print("No 'spectrum' tag found in the source.")
    else:
        print(f"Source with name '{source_name}' not found.")
##################################################################################
##################################################################################
def generate_files(vars, snrratios=None, time_intervals=None, number_of_bins=None):
    # Extract variables
    source_name, ra, dec, method, specin, _, _, minimal_energy, maximal_energy = vars
    source_name_cleaned = source_name.replace(" ", "").replace(".", "dot").replace("+", "plus").replace("-", "minus")

    general_path = f'{general_path_for_slurm}/data/{source_name_cleaned}/'
    sc = f"{general_path_for_slurm}/mission/spacecraft/lat_spacecraft_merged.fits" 
    
    ebinfile_txt = f"{general_path_for_slurm}/energy_7bins_gtbindef.txt"

    
    with open(f'{ebinfile_txt}', 'r') as file:
                for line in file:
                    line = line.strip()
                    if not line or len(line.split()) != 2:
                        continue
                    emin_float, emax_float = map(float, line.split())

                    # Convert the values to integers for the filename
                    emin = int(emin_float)
                    emax = int(emax_float)
                    ebinfile = f'/groups/pheno/sqd515/MasterThesis/energy_bins_def/{number_of_bins}/energy_bins_{emin}_{emax}.fits'
                    if not os.path.exists(ebinfile):
                        # Create energy bin definition file for this bin
                        energy_bin_txt = f'/groups/pheno/sqd515/MasterThesis/energy_bins_def/{number_of_bins}/energy_bin_{emin}_{emax}.txt'
                        with open(energy_bin_txt, 'w') as f:
                            f.write(f'{emin_float}   {emax_float}\n')

                        # Create the energy bin FITS file
                        gtbindef_energy_command = [
                            'gtbindef',
                            'E',
                            energy_bin_txt,
                            ebinfile,
                            'MeV']
                        subprocess.run(gtbindef_energy_command, check=True)
    # Determine the loop items based on the method
    if method == "SNR":
        loop_items = snrratios
    elif method == "LIN":
        loop_items = time_intervals
    else:
        loop_items = "NONE"  # No looping for the "NONE" method

    # If there is nothing to loop over, handle the "NONE" method directly
    if method == "NONE":
        with open(f'{ebinfile_txt}', 'r') as file:
                for line in file:
                    line = line.strip()
                    if not line or len(line.split()) != 2:
                        continue
                    emin_float, emax_float = map(float, line.split())

                    # Convert the values to integers for the filename
                    emin = int(emin_float)
                    emax = int(emax_float)
                    gti_noflares = general_path + f'{method}/gti_{emin}_{emax}.fits'
                    ltcube = general_path + f'{method}/ltcube/ltcube.fits'
                    ccube = general_path + f'{method}/ccube/ccube_{emin}_{emax}.fits'
                    binexpmap = general_path + f'{method}/expmap/BinnedExpMap_{emin}_{emax}.fits'
                    model = general_path + f'{method}/models/input_model_{emin}_{emax}.xml'
                    ebinfile = f'{general_path_for_slurm}/energy_bins_def/{number_of_bins}/energy_bins_{emin}_{emax}.fits'
                    print(f"Processing method {method} without looping.")
                    
                    if not os.path.exists(ltcube):
                        print(f"Creating ltcube for {method}")
                        my_apps.expCube['evfile'] = gti_noflares
                        my_apps.expCube['scfile'] = sc
                        my_apps.expCube['outfile'] = ltcube
                        my_apps.expCube['zmax'] = 90
                        my_apps.expCube['dcostheta'] = 0.025
                        my_apps.expCube['binsz'] = 1
                        my_apps.expCube.run()
                    else:
                        print(f'{ltcube} file exists!')

                    if not os.path.exists(ccube):
                        print(f"Creating ccube for {method}")
                        ####### Counts Cube #######
                        my_apps.evtbin['evfile'] = gti_noflares
                        my_apps.evtbin['outfile'] = ccube
                        my_apps.evtbin['scfile'] = 'NONE'
                        my_apps.evtbin['algorithm'] = 'CCUBE'
                        my_apps.evtbin['nxpix'] = 100
                        my_apps.evtbin['nypix'] = 100
                        my_apps.evtbin['binsz'] = 0.2
                        my_apps.evtbin['coordsys'] = 'CEL'
                        my_apps.evtbin['xref'] = ra
                        my_apps.evtbin['yref'] = dec
                        my_apps.evtbin['axisrot'] = 0
                        my_apps.evtbin['proj'] = 'CAR'
                        #my_apps.evtbin['ebinalg'] = 'FILE'
                        my_apps.evtbin['ebinfile'] = 'NONE'
                        #my_apps.evtbin.run()
                        my_apps.evtbin['emin'] = emin
                        my_apps.evtbin['emax'] = emax
                        my_apps.evtbin['enumbins'] = 3
                        my_apps.evtbin['ebinalg'] = 'LOG'
                        my_apps.evtbin.run()
                    else:
                        print(f'{ccube} file exists!')

                    if not os.path.exists(binexpmap):
                        print(f"Creating exposuremap for {method}")
                        ####### Exposure Map #######
                        expCube2['infile'] = ltcube
                        expCube2['cmap'] = 'none'
                        expCube2['outfile'] = binexpmap
                        expCube2['irfs'] = 'P8R3_SOURCE_V3'
                        expCube2['evtype'] = '3'
                        expCube2['nxpix'] = 1800
                        expCube2['nypix'] = 900
                        expCube2['binsz'] = 0.2
                        expCube2['coordsys'] = 'CEL'
                        expCube2['xref'] = ra
                        expCube2['yref'] = dec
                        expCube2['axisrot'] = 0
                        expCube2['proj'] = 'CAR'
                        #expCube2['ebinalg'] = 'FILE'
                        expCube2['ebinfile'] = 'NONE'
                        expCube2['ebinalg'] = 'LOG'
                        expCube2['emin'] = emin
                        expCube2['emax'] = emax
                        expCube2['enumbins'] = 3
                        expCube2['evtype'] = 3
                        expCube2['edisp_bins'] = edisp_bins
                        expCube2.run()
                    else:
                        print(f'{binexpmap} file exists!')
                
                    ####### Make model #######
                    ##### Run make4FGLxml Command #####
                    if not os.path.exists(model):
                        make4FGLxml_command = [f'make4FGLxml {general_path_for_slurm}/data/gll_psc_v32.xml --event_file {gti_noflares} -o {model} --free_radius 5.0 --norms_free_only True --sigma_to_free 25 --variable_free True']
                        subprocess.run(make4FGLxml_command, shell=True, check=True, executable='/bin/bash')
                    else:
                        print(f'{model} file exists!')
                    # Run the command using subprocess
                    
                    #tree = ET.parse(f'{model}')
                    #modify_and_save(tree, source_name, method, None, emin, emax)
    else:
        for loop_item in loop_items:
            with open(f'{ebinfile_txt}', 'r') as file:
                for line in file:
                    line = line.strip()
                    if not line or len(line.split()) != 2:
                        continue
                    emin_float, emax_float = map(float, line.split())

                    # Convert the values to integers for the filename
                    emin = int(emin_float)
                    emax = int(emax_float)
                    ebinfile = f'{general_path_for_slurm}/energy_bins_def/{number_of_bins}/energy_bins_{emin}_{emax}.fits'
                    if method == "SNR":
                        gti_noflares = general_path + f'{method}/gti_noflares_snr{loop_item}_{emin}_{emax}.fits'
                        ltcube = general_path + f'{method}/ltcube/ltcube_snr{loop_item}.fits'
                        ccube = general_path + f'{method}/ccube/ccube_snr{loop_item}_{emin}_{emax}.fits'
                        binexpmap = general_path + f'{method}/expmap/BinnedExpMap_snr{loop_item}_{emin}_{emax}.fits'
                        model = general_path + f'{method}/models/input_model_snr{loop_item}_{emin}_{emax}.xml'
                    elif method == "LIN":
                        gti_noflares = general_path + f'{method}/gti_noflares_{loop_item}_{emin}_{emax}.fits'
                        ltcube = general_path + f'{method}/ltcube/ltcube_{loop_item}.fits'
                        ccube = general_path + f'{method}/ccube/ccube_{loop_item}_{emin}_{emax}.fits'
                        binexpmap = general_path + f'{method}/expmap/BinnedExpMap_{loop_item}_{emin}_{emax}.fits'
                        model = general_path + f'{method}/models/input_model_{loop_item}_{emin}_{emax}.xml'

                    if not os.path.exists(ltcube):
                        print(f"Creating ltcube for {method}: {loop_item}")
                        my_apps.expCube['evfile'] = gti_noflares
                        my_apps.expCube['scfile'] = sc
                        my_apps.expCube['outfile'] = ltcube
                        my_apps.expCube['zmax'] = 90
                        my_apps.expCube['dcostheta'] = 0.025
                        my_apps.expCube['binsz'] = 1
                        my_apps.expCube.run()
                    else:
                        print(f'{ltcube} file exists!')

                    if not os.path.exists(ccube):
                        print(f"Creating ccube for {method}: {loop_item}")
                        ####### Counts Cube #######
                        my_apps.evtbin['evfile'] = gti_noflares
                        my_apps.evtbin['outfile'] = ccube
                        my_apps.evtbin['scfile'] = 'NONE'
                        my_apps.evtbin['algorithm'] = 'CCUBE'
                        my_apps.evtbin['nxpix'] = 100
                        my_apps.evtbin['nypix'] = 100
                        my_apps.evtbin['binsz'] = 0.2
                        my_apps.evtbin['coordsys'] = 'CEL'
                        my_apps.evtbin['xref'] = ra
                        my_apps.evtbin['yref'] = dec
                        my_apps.evtbin['axisrot'] = 0
                        my_apps.evtbin['proj'] = 'CAR'
                        #my_apps.evtbin['ebinalg'] = 'FILE'
                        my_apps.evtbin['ebinfile'] = "NONE"
                        my_apps.evtbin['emin'] = emin
                        my_apps.evtbin['emax'] = emax
                        my_apps.evtbin['enumbins'] = 3
                        my_apps.evtbin['ebinalg'] = 'LOG'
                        my_apps.evtbin.run()
                    else:
                        print(f'{ccube} file exists!')

                    if not os.path.exists(binexpmap):
                        print(f"Creating exposuremap for {method}: {loop_item}")
                        ####### Exposure Map #######
                        expCube2['infile'] = ltcube
                        expCube2['cmap'] = 'none'
                        expCube2['outfile'] = binexpmap
                        expCube2['irfs'] = 'P8R3_SOURCE_V3'
                        expCube2['evtype'] = '3'
                        expCube2['nxpix'] = 1800
                        expCube2['nypix'] = 900
                        expCube2['binsz'] = 0.2
                        expCube2['coordsys'] = 'CEL'
                        expCube2['xref'] = ra
                        expCube2['yref'] = dec
                        expCube2['axisrot'] = 0
                        expCube2['proj'] = 'CAR'
                        #expCube2['ebinalg'] = 'FILE'
                        expCube2['ebinfile'] = "NONE"
                        #expCube2.run()
                        expCube2['ebinalg'] = 'LOG'
                        expCube2['emin'] = emin
                        expCube2['emax'] = emax
                        expCube2['enumbins'] = 3
                        expCube2['evtype'] = 3
                        expCube2['edisp_bins'] = edisp_bins
                        expCube2.run()
                    else:
                        print(f'{binexpmap} file exists!')
                
                    ####### Make model #######
                    ##### Run make4FGLxml Command #####
                    if not os.path.exists(model):
                        make4FGLxml_command = [f'make4FGLxml {general_path_for_slurm}/data/gll_psc_v32.xml --event_file {gti_noflares} -o {model} --free_radius 5.0 --norms_free_only True --sigma_to_free 25 --variable_free True']
                        subprocess.run(make4FGLxml_command, shell=True, check=True, executable='/bin/bash')
                    else:
                        print(f'{model} file exists!')
                    # Run the command using subprocess
                    
                    #tree = ET.parse(f'{model}')
                    #modify_and_save(tree, source_name, method, loop_item, emin, emax)

    return

def source_maps(vars, snrratios=None, time_intervals=None):
    # Extract variables
    source_name, ra, dec, method, specin, _, _, minimal_energy, maximal_energy = vars
    source_name_cleaned = source_name.replace(" ", "").replace(".", "dot").replace("+", "plus").replace("-", "minus")

    general_path = f'{general_path_for_slurm}/data/{source_name_cleaned}/'
    ebinfile_txt = f'{general_path_for_slurm}/energy_7bins_gtbindef.txt'
    # Determine the loop items based on the method
    if method == "SNR":
        loop_items = snrratios
    elif method == "LIN":
        loop_items = time_intervals
    else:
        loop_items = "NONE"  # No looping for the "NONE" method
    
    #input_model = general_path + f'NONE/models/input_model.xml'
    
    # If there is nothing to loop over, handle the "NONE" method directly
    if method == "NONE":
        ltcube = general_path + f'{method}/ltcube/ltcube.fits'
        with open(f'{ebinfile_txt}', 'r') as file:
                for line in file:
                    line = line.strip()
                    if not line or len(line.split()) != 2:
                        continue
                    emin_float, emax_float = map(float, line.split())

                    # Convert the values to integers for the filename
                    emin = int(emin_float)
                    emax = int(emax_float)
                    
                    ccube = general_path + f'{method}/ccube/ccube_{emin}_{emax}.fits'
                    binexpmap = general_path + f'{method}/expmap/BinnedExpMap_{emin}_{emax}.fits'
                    srcmap = general_path + f'{method}/srcmap/srcmap_{emin}_{emax}.fits'
                    input_model = general_path + f'{method}/models/input_model_{emin}_{emax}.xml'
                    print(f"Processing method {method} without looping.")
                    
                    if not os.path.exists(srcmap):
                        ####### Source Map #######
                        print(f"Creating sourcemap for {method}: {emin_float}MeV - {emax_float}MeV")
                        my_apps.srcMaps['expcube'] = ltcube
                        my_apps.srcMaps['cmap'] = ccube
                        my_apps.srcMaps['srcmdl'] = input_model
                        my_apps.srcMaps['bexpmap'] = binexpmap
                        my_apps.srcMaps['outfile'] = srcmap
                        my_apps.srcMaps['irfs'] = 'P8R3_SOURCE_V3'
                        my_apps.srcMaps['evtype'] = '3'
                        my_apps.srcMaps['edisp_bins'] = edisp_bins
                        my_apps.srcMaps.run()
                    else:
                        print(f'{srcmap} file exists!')
    else:
        for loop_item in loop_items:
            with open(f'{ebinfile_txt}', 'r') as file:
                for line in file:
                    line = line.strip()
                    if not line or len(line.split()) != 2:
                        continue
                    emin_float, emax_float = map(float, line.split())

                    # Convert the values to integers for the filename
                    emin = int(emin_float)
                    emax = int(emax_float)

                    if method == "SNR":
                        ltcube = general_path + f'{method}/ltcube/ltcube_snr{loop_item}.fits'
                        ccube = general_path + f'{method}/ccube/ccube_snr{loop_item}_{emin}_{emax}.fits'
                        binexpmap = general_path + f'{method}/expmap/BinnedExpMap_snr{loop_item}_{emin}_{emax}.fits'
                        srcmap = general_path + f'{method}/srcmap/srcmap_snr{loop_item}_{emin}_{emax}.fits'
                        input_model = general_path + f'{method}/models/input_model_snr{loop_item}_{emin}_{emax}.xml'
                    elif method == "LIN":
                        ltcube = general_path + f'{method}/ltcube/ltcube_{loop_item}.fits'
                        ccube = general_path + f'{method}/ccube/ccube_{loop_item}_{emin}_{emax}.fits'
                        binexpmap = general_path + f'{method}/expmap/BinnedExpMap_{loop_item}_{emin}_{emax}.fits'
                        srcmap = general_path + f'{method}/srcmap/srcmap_{loop_item}_{emin}_{emax}.fits'
                        input_model = general_path + f'{method}/models/input_model_{loop_item}_{emin}_{emax}.xml'

                    if not os.path.exists(srcmap):
                        ####### Source Map #######
                        print(f"Creating sourcemap for {method}: {loop_item}: {emin_float}MeV - {emax_float}MeV")
                        my_apps.srcMaps['expcube'] = ltcube
                        my_apps.srcMaps['cmap'] = ccube
                        my_apps.srcMaps['srcmdl'] = input_model
                        my_apps.srcMaps['bexpmap'] = binexpmap
                        my_apps.srcMaps['outfile'] = srcmap
                        my_apps.srcMaps['irfs'] = 'P8R3_SOURCE_V3'
                        my_apps.srcMaps['evtype'] = '3'
                        my_apps.srcMaps['edisp_bins'] = edisp_bins
                        my_apps.srcMaps.run()
                    else:
                        print(f'{srcmap} file exists!')
    pass
##################################################################################
##################################################################################
def save_source_results_to_fits(source_name, method_results, filename):
    """
    Save a single source's fit results to a FITS file.

    Args:
        source_name (str): Name of the source.
        method_results (dict): Dictionary of method results for the source.
        filename (str): Path to save the FITS file.
    """
    hdul = fits.HDUList([fits.PrimaryHDU()])

    for method, data in method_results.items():
        table = Table(data)
        hdu = fits.BinTableHDU(table, name=method)
        hdul.append(hdu)

    hdul.writeto(filename, overwrite=True)
##################################################################################
##################################################################################
def run_binned_likelihood(vars, snrratios=None, time_intervals=None, free_params = None):
    source_name, ra, dec, method, specin, _, _, minimal_energy, maximal_energy = vars
    source_name_cleaned = source_name.replace(" ", "").replace(".", "dot").replace("+", "plus").replace("-", "minus")
    ebinfile_txt = f'{general_path_for_slurm}/energy_7bins_gtbindef.txt'
    general_path = f'{general_path_for_slurm}/data/{source_name_cleaned}/'
    method_data = {}
    irf='CALDB'
    optimizer = 'MINUIT'
    ts_min = 2 # all sources with a smaller TS will be removed                          

    def RemoveWeak(analysis, sources_to_keep ):
        names=analysis.sourceNames()
        ndel = 0 # number of deleted sources

        for nn in names:
            source_free = bool( analysis.freePars( nn )  ) # check if source has free params. We are not planiining to remove fixed sources

            # we also do not want to remove our source itself from the analysis in any case
            if( source_free and (nn not in sources_to_keep) ):
                ts = analysis.Ts( nn ) # TS with which the source was detected
                if( ts < ts_min ):
                    analysis.deleteSource( nn )
                    ndel += 1
        return analysis, ndel
    
    # Determine the loop items based on the method
    if method == "SNR":
        loop_items = snrratios
    elif method == "LIN":
        loop_items = time_intervals
    else:
        loop_items = "NONE"  # No looping for the "NONE" method

    # If there is nothing to loop over, handle the "NONE" method directly
    if method == "NONE":
        fit_data_list = []  # Accumulate all fit_data for the method
        failed_bins = []
        successful_bins = {}  # Map successful bins to their input_model 
        with open(f'{ebinfile_txt}', 'r') as file:
                for line in file:
                    line = line.strip()
                    if not line or len(line.split()) != 2:
                        continue
                    emin_float, emax_float = map(float, line.split())

                    # Convert the values to integers for the filename
                    emin = int(emin_float)
                    emax = int(emax_float)
                    print(f"Processing {method}: {emin_float}MeV - {emax_float}MeV")
                    ltcube = general_path + f'{method}/ltcube/ltcube.fits'
                    ccube = general_path + f'{method}/ccube/ccube_{emin}_{emax}.fits'
                    binexpmap = general_path + f'{method}/expmap/BinnedExpMap_{emin}_{emax}.fits'
                    srcmap = general_path + f'{method}/srcmap/srcmap_{emin}_{emax}.fits'
                    if free_params == "None":
                        input_model = general_path + f'{method}/models/input_model_{emin}_{emax}.xml'
                        cspectra = general_path + f'{method}/CountsSpectra/cspectra_{emin}_{emax}.fits'
                        writexml = general_path + f'{method}/fit_params/fit_{emin}_{emax}.xml' 
                        results_output_file = f"{source_name_cleaned}_results.fits"
                    elif free_params == "alpha":
                        input_model = general_path + f'{method}/models/input_model_free_alpha_{emin}_{emax}.xml'
                        cspectra = general_path + f'{method}/CountsSpectra/cspectra_free_alpha_{emin}_{emax}.fits'
                        writexml = general_path + f'{method}/fit_params/fit_free_alpha_{emin}_{emax}.xml' 
                        results_output_file = f"{source_name_cleaned}_free_alpha_results.fits"
                    elif free_params == "beta":
                        input_model = general_path + f'{method}/models/input_model_free_beta.xml'
                        cspectra = general_path + f'{method}/CountsSpectra/cspectra_free_beta.fits'
                        writexml = general_path + f'{method}/fit_params/fit_free_beta.xml'
                        results_output_file = f"{source_name_cleaned}_free_beta_results.fits"
                    elif free_params == "alpha and beta":
                        input_model = general_path + f'{method}/models/input_model_free_alpha_beta.xml'
                        cspectra = general_path + f'{method}/CountsSpectra/cspectra_free_alpha_beta.fits'
                        writexml = general_path + f'{method}/fit_params/fit_free_alpha_beta.xml'
                        results_output_file = f"{source_name_cleaned}_free_alpha_beta_results.fits"
                    
                    try:
                        cfg = BinnedConfig(edisp_bins=edisp_bins)
                        analysis = binnedAnalysis (config=cfg, irfs=irf,cmap=srcmap,bexpmap=binexpmap,expcube=ltcube,srcmdl=input_model, optimizer=optimizer)
                        analysis.logLike.set_edisp_flag(True)
                        iteration = 0
                        ndeleted = 1 # number of deleted sources. We want to enter the loop below at least once.

                        while( ndeleted ):
                            likeobj = pyLikelihood.Minuit( analysis.logLike )  # or pyLike.NewMinuit(analysis.logLike)
                            loglike = analysis.fit(verbosity=0, covar=True,optObject=likeobj)
                            like, ndeleted = RemoveWeak(analysis, source_name)
                            print( f'After iteration {iteration} we removed {ndeleted} sources' )
                            iteration += 1
                        like.logLike.writeXml(writexml)
                        TS = like.Ts(source_name) #also include in output file
                        convergence = likeobj.getRetCode()  #also include in output file

                        # Save successful bin details
                        successful_bins[(emin, emax)] = writexml

                        arg = pyLikelihood.dArg( (emin*emax)**0.5 ) # Emin, Emax are in MeV
                        flux = like.model.srcs[source_name].src.spectrum()(arg) *emin*emax*1.6e-6  # differential flux in erg/cm2/s ; source -- the name of the source
                        coeff = flux / like.flux(source_name,emin,emax)

                        dflux = like.fluxError(source_name,emin,emax)*coeff # flux error, erg/cm2/s

                        nobs = like.nobs
                        geometric_mean = (emin*emax)**0.5

                        print(emin, emax, geometric_mean, geometric_mean - emin, emax - geometric_mean, float(flux), float(dflux), float(sum(nobs)), TS, convergence)

                        # Add the data for this energy bin to the list
                        fit_data_list.append({
                            'emin': emin,
                            'emax': emax,
                            'geometric_mean': geometric_mean,
                            'e_lower': geometric_mean - emin,
                            'e_upper': emax - geometric_mean,
                            'flux_tot_value': float(flux),
                            'flux_tot_error': float(dflux),
                            'nobs': float(sum(nobs)),
                            'TS': TS,
                            'convergence': convergence
                        })

                    except Exception as e:
                        print(f"Error processing {method}: {emin_float}-{emax_float}MeV: {e}")
                        failed_bins.append((emin, emax))

        # Second pass: Refit failed bins
        if failed_bins:
            print(f"Refitting failed bins: {failed_bins}")
            for emin, emax in failed_bins:
                successful_bin_keys = list(successful_bins.keys())
                refit_success = False
                ltcube = general_path + f'{method}/ltcube/ltcube.fits'
                ccube = general_path + f'{method}/ccube/ccube_{emin}_{emax}.fits'
                binexpmap = general_path + f'{method}/expmap/BinnedExpMap_{emin}_{emax}.fits'
                srcmap = general_path + f'{method}/srcmap/srcmap_{emin}_{emax}.fits'
            
                for ref_bin in successful_bin_keys:
                    if refit_success:  # Stop refitting if already successful
                        break
                    ref_model = successful_bins[ref_bin]  # Model of the selected successful bin
                    writexml = general_path + f'{method}/fit_params/refit_{emin}_{emax}.xml'
                    cspectra = general_path + f'{method}/CountsSpectra/refit_cspectra_{emin}_{emax}.fits'

                    try:
                        cfg = BinnedConfig(edisp_bins=edisp_bins)
                        like = binnedAnalysis(config=cfg, irfs=irf,cmap=srcmap,bexpmap=binexpmap,expcube=ltcube,srcmdl=ref_model, optimizer=optimizer)
                        like.logLike.set_edisp_flag(True)
                        likeobj = pyLikelihood.Minuit(like.logLike)
                        like.fit(verbosity=0, covar=True,optObject=likeobj)
                        TS = like.Ts(source_name) #also include in output file
                        convergence = likeobj.getRetCode()  #also include in output file
                        arg = pyLikelihood.dArg( (emin*emax)**0.5 ) # Emin, Emax are in MeV
                        flux = like.model.srcs[source_name].src.spectrum()(arg) *emin*emax*1.6e-6  # differential flux in erg/cm2/s ; source -- the name of the source
                        coeff = flux / like.flux(source_name,emin,emax)

                        dflux = like.fluxError(source_name,emin,emax)*coeff # flux error, erg/cm2/s
                        geometric_mean = (emin * emax) ** 0.5
                        nobs = like.nobs

                        fit_data_list.append({
                            'emin': emin,
                            'emax': emax,
                            'geometric_mean': geometric_mean,
                            'e_lower': geometric_mean - emin,
                            'e_upper': emax - geometric_mean,
                            'flux_tot_value': float(flux),
                            'flux_tot_error': float(dflux),
                            'nobs': float(sum(nobs)),
                            'TS': TS,
                            'convergence': convergence
                        })
                        print(f"Refitted bin succesfully: {emin}-{emax}MeV")
                        refit_success = True
                    except Exception as e:
                        print(f"Refit failed for {emin}-{emax}MeV: {e}")

                if not refit_success:
                    print(f"All refits failed for {emin}-{emax}MeV")

        # After processing all lines, save the accumulated data to a single FITS file
        if fit_data_list:
            # Extract columns
            emin_col = [d['emin'] for d in fit_data_list]
            emax_col = [d['emax'] for d in fit_data_list]
            geometric_mean_col = [d['geometric_mean'] for d in fit_data_list]
            e_lower_col = [d['e_lower'] for d in fit_data_list]
            e_upper_col = [d['e_upper'] for d in fit_data_list]
            flux_tot_value_col = [d['flux_tot_value'] for d in fit_data_list]
            flux_tot_error_col = [d['flux_tot_error'] for d in fit_data_list]
            nobs_col = [d['nobs'] for d in fit_data_list]
            TS_col = [d['TS'] for d in fit_data_list]
            conv_col = [d['convergence'] for d in fit_data_list]

            # Create FITS columns
            cols = [
                fits.Column(name='emin', format='E', array=emin_col),
                fits.Column(name='emax', format='E', array=emax_col),
                fits.Column(name='geometric_mean', format='E', array=geometric_mean_col),
                fits.Column(name='e_lower', format='E', array=e_lower_col),
                fits.Column(name='e_upper', format='E', array=e_upper_col),
                fits.Column(name='flux_tot_value', format='E', array=flux_tot_value_col),
                fits.Column(name='flux_tot_error', format='E', array=flux_tot_error_col),
                fits.Column(name='nobs', format='E', array=nobs_col),
                fits.Column(name='TS', format='E', array=TS_col),
                fits.Column(name='convergence', format='E', array=conv_col),
            ]

            # Write to a single FITS file
            hdu = fits.BinTableHDU.from_columns(cols)
            output_fits_file = f'{general_path_for_slurm}/fit_results/{source_name_cleaned}_fit_data_{method}.fits'
            hdu.writeto(output_fits_file, overwrite=True)
            print(f"Saved single FITS file for method {method}: {output_fits_file}")
        
    else:
        for loop_item in loop_items:
            fit_data_list = []
            failed_bins = []
            successful_bins = {} 
            with open(f'{ebinfile_txt}', 'r') as file:
                for line in file:
                    line = line.strip()
                    if not line or len(line.split()) != 2:
                        continue
                    emin_float, emax_float = map(float, line.split())

                    # Convert the values to integers for the filename
                    emin = int(emin_float)
                    emax = int(emax_float)
                    print(f"Processing {method}: {loop_item} {emin_float}MeV - {emax_float}MeV")
                    if method == "SNR":
                        ltcube = general_path + f'{method}/ltcube/ltcube_snr{loop_item}.fits'
                        ccube = general_path + f'{method}/ccube/ccube_snr{loop_item}_{emin}_{emax}.fits'
                        binexpmap = general_path + f'{method}/expmap/BinnedExpMap_snr{loop_item}_{emin}_{emax}.fits'
                        srcmap = general_path + f'{method}/srcmap/srcmap_snr{loop_item}_{emin}_{emax}.fits'
                        if free_params == "None":
                            input_model = general_path + f'{method}/models/input_model_snr{loop_item}_{emin}_{emax}.xml'
                            cspectra = general_path + f'{method}/CountsSpectra/cspectra_snr{loop_item}_{emin}_{emax}.fits'
                            writexml = general_path + f'{method}/fit_params/fit_snr{loop_item}_{emin}_{emax}.xml'
                            results_output_file = f"{source_name_cleaned}_results_snr.fits"
                        elif free_params == "alpha":
                            input_model = general_path + f'{method}/models/input_model_snr{loop_item}_free_alpha_{emin}_{emax}.xml' ## HER
                            cspectra = general_path + f'{method}/CountsSpectra/cspectra_snr{loop_item}_free_alpha_{emin}_{emax}.fits'
                            writexml = general_path + f'{method}/fit_params/fit_snr{loop_item}_free_alpha_{emin}_{emax}.xml'
                            results_output_file = f"{source_name_cleaned}_snr{loop_item}_free_alpha_results.fits"
                        elif free_params == "beta":
                            input_model = general_path + f'{method}/models/input_model_snr{loop_item}_free_beta.xml'
                            cspectra = general_path + f'{method}/CountsSpectra/cspectra_snr{loop_item}_free_beta.fits'
                            writexml = general_path + f'{method}/fit_params/fit_snr{loop_item}_free_beta.xml'
                            results_output_file = f"{source_name_cleaned}_snr{loop_item}_free_beta_results.fits"
                        elif free_params == "alpha and beta":
                            input_model = general_path + f'{method}/models/input_model_snr{loop_item}_free_alpha_beta.xml'
                            cspectra = general_path + f'{method}/CountsSpectra/cspectra_snr{loop_item}_free_alpha_beta.fits'
                            writexml = general_path + f'{method}/fit_params/fit_snr{loop_item}_free_alpha_beta.xml'
                            results_output_file = f"{source_name_cleaned}_snr{loop_item}_free_alpha_beta_results.fits"
                    elif method == "LIN":
                        ltcube = general_path + f'{method}/ltcube/ltcube_{loop_item}.fits'
                        ccube = general_path + f'{method}/ccube/ccube_{loop_item}_{emin}_{emax}.fits'
                        binexpmap = general_path + f'{method}/expmap/BinnedExpMap_{loop_item}_{emin}_{emax}.fits'
                        srcmap = general_path + f'{method}/srcmap/srcmap_{loop_item}_{emin}_{emax}.fits'
                        if free_params == "None":
                            input_model = general_path + f'{method}/models/input_model_{loop_item}_{emin}_{emax}.xml'
                            cspectra = general_path + f'{method}/CountsSpectra/cspectra_{loop_item}_{emin}_{emax}.fits'
                            writexml = general_path + f'{method}/fit_params/fit_{loop_item}_{emin}_{emax}.xml'
                            results_output_file = f"{source_name_cleaned}_results_lin.fits"
                        if free_params == "alpha":
                            input_model = general_path + f'{method}/models/input_model_{loop_item}_free_alpha_{emin}_{emax}.xml' ## HER
                            cspectra = general_path + f'{method}/CountsSpectra/cspectra_{loop_item}_free_alpha_{emin}_{emax}.fits'
                            writexml = general_path + f'{method}/fit_params/fit_{loop_item}_free_alpha_{emin}_{emax}.xml'
                            results_output_file = f"{source_name_cleaned}_{loop_item}_free_alpha_results.fits"
                        elif free_params == "beta":
                            input_model = general_path + f'{method}/models/input_model_{loop_item}_free_beta.xml'
                            cspectra = general_path + f'{method}/CountsSpectra/cspectra_{loop_item}_free_beta.fits'
                            writexml = general_path + f'{method}/fit_params/fit_{loop_item}_free_beta.xml'
                            results_output_file = f"{source_name_cleaned}_{loop_item}_free_beta_results.fits"
                        elif free_params == "alpha and beta":
                            input_model = general_path + f'{method}/models/input_model_{loop_item}_free_alpha_beta.xml'
                            cspectra = general_path + f'{method}/CountsSpectra/cspectra_{loop_item}_free_alpha_beta.fits'
                            writexml = general_path + f'{method}/fit_params/fit_{loop_item}_free_alpha_beta.xml'
                            results_output_file = f"{source_name_cleaned}_{loop_item}_free_alpha_beta_results.fits"

                     ####### Binned Likelihood Analysis #######
                    try:
                        cfg = BinnedConfig(edisp_bins=edisp_bins)
                        analysis = binnedAnalysis (config=cfg, irfs=irf,cmap=srcmap,bexpmap=binexpmap,expcube=ltcube,srcmdl=input_model, optimizer=optimizer)
                        analysis.logLike.set_edisp_flag(True)
                        iteration = 0
                        ndeleted = 1 # number of deleted sources. We want to enter the loop below at least once.

                        while( ndeleted ):
                            likeobj = pyLikelihood.Minuit( analysis.logLike )  # or pyLike.NewMinuit(analysis.logLike)
                            loglike = analysis.fit(covar=True,optObject=likeobj)
                            like, ndeleted = RemoveWeak(analysis, source_name)
                            print( f'After iteration {iteration} we removed {ndeleted} sources' )
                            iteration += 1
                        like.logLike.writeXml(writexml)
                        TS = like.Ts(source_name) #also include in output file
                        convergence = likeobj.getRetCode()  #also include in output file
                        
                        # Save successful bin details
                        successful_bins[(emin, emax)] = writexml
                        
                        arg = pyLikelihood.dArg( (emin*emax)**0.5 ) # Emin, Emax are in MeV
                        flux = like.model.srcs[source_name].src.spectrum()(arg) *emin*emax*1.6e-6  # differential flux in erg/cm2/s ; source -- the name of the source
                        coeff = flux / like.flux(source_name,emin,emax)

                        dflux = like.fluxError(source_name,emin,emax)*coeff # flux error, erg/cm2/s

                        nobs = like.nobs
                        geometric_mean = (emin*emax)**0.5

                        fit_data = {
                            'loop_item': loop_item,
                            'emin': emin,
                            'emax': emax,
                            'geometric_mean': geometric_mean,
                            'e_lower': geometric_mean - emin,
                            'e_upper': emax - geometric_mean,
                            'flux_tot_value': float(flux),
                            'flux_tot_error': float(dflux),                    
                            'nobs': float(sum(nobs)),
                            'TS': TS,
                            'convergence': convergence
                        }

                        # Append data to method_data for the current method
                        if method not in method_data:
                            method_data[method] = []
                        method_data[method].append(fit_data)

                    except Exception as e:
                        print(f"Error processing {method}: {emin_float}-{emax_float}MeV: {e}")
                        failed_bins.append((emin, emax))
              # Second pass: Refit failed bins
            if failed_bins:
                print(f"Refitting failed bins {method}: {failed_bins}")
                for emin, emax in failed_bins:
                    successful_bin_keys = list(successful_bins.keys())
                    refit_success = False
                    if method == "SNR":
                            ltcube = general_path + f'{method}/ltcube/ltcube_snr{loop_item}.fits'
                            ccube = general_path + f'{method}/ccube/ccube_snr{loop_item}_{emin}_{emax}.fits'
                            binexpmap = general_path + f'{method}/expmap/BinnedExpMap_snr{loop_item}_{emin}_{emax}.fits'
                            srcmap = general_path + f'{method}/srcmap/srcmap_snr{loop_item}_{emin}_{emax}.fits'
                    elif method == "LIN":
                            ltcube = general_path + f'{method}/ltcube/ltcube_{loop_item}.fits'
                            ccube = general_path + f'{method}/ccube/ccube_{loop_item}_{emin}_{emax}.fits'
                            binexpmap = general_path + f'{method}/expmap/BinnedExpMap_{loop_item}_{emin}_{emax}.fits'
                            srcmap = general_path + f'{method}/srcmap/srcmap_{loop_item}_{emin}_{emax}.fits'
                
                    for ref_bin in successful_bin_keys:
                        if refit_success:  # Stop refitting if already successful
                            break
                        ref_model = successful_bins[ref_bin]  # Model of the selected successful bin
                        writexml = general_path + f'{method}/fit_params/refit_{emin}_{emax}.xml'
                        cspectra = general_path + f'{method}/CountsSpectra/refit_cspectra_{emin}_{emax}.fits'

                        try:
                            cfg = BinnedConfig(edisp_bins=edisp_bins)
                            like = binnedAnalysis (config=cfg, irfs=irf,cmap=srcmap,bexpmap=binexpmap,expcube=ltcube,srcmdl=ref_model, optimizer=optimizer)
                            like.logLike.set_edisp_flag(True)
                            likeobj = pyLikelihood.Minuit(like.logLike)
                            like.fit(covar=True,optObject=likeobj)
                            TS = like.Ts(source_name) #also include in output file
                            convergence = likeobj.getRetCode()  #also include in output file

                            arg = pyLikelihood.dArg( (emin*emax)**0.5 ) # Emin, Emax are in MeV
                            flux = like.model.srcs[source_name].src.spectrum()(arg) *emin*emax*1.6e-6  # differential flux in erg/cm2/s ; source -- the name of the source
                            coeff = flux / like.flux(source_name,emin,emax)

                            dflux = like.fluxError(source_name,emin,emax)*coeff # flux error, erg/cm2/s
                            geometric_mean = (emin * emax) ** 0.5
                            nobs = like.nobs

                            fit_data = {
                                'loop_item': loop_item,
                                'emin': emin,
                                'emax': emax,
                                'geometric_mean': geometric_mean,
                                'e_lower': geometric_mean - emin,
                                'e_upper': emax - geometric_mean,
                                'flux_tot_value': float(flux),
                                'flux_tot_error': float(dflux),
                                'nobs': float(sum(nobs)),
                                'TS': TS,
                                'convergence': convergence
                            }
                            print(f"Refitted bin succesfully {method}: {emin}-{emax}MeV")
                            refit_success = True
                            # Append data to method_data for the current method
                            if method not in method_data:
                                method_data[method] = []
                            method_data[method].append(fit_data)
                        except Exception as e:
                            print(f"Refit failed for {method} {emin}-{emax}MeV: {e}")

                    if not refit_success:
                        print(f"All refits failed for {method} {emin}-{emax}MeV")

    # Save one FITS file per method
    for method, data_list in method_data.items():
        # Flatten data into columns
        loop_items_col = [data['loop_item'] for data in data_list]
        emin_col = [data['emin'] for data in data_list]
        emax_col = [data['emax'] for data in data_list]
        geometric_mean_col = [data['geometric_mean'] for data in data_list]
        e_lower_col = [data['e_lower'] for data in data_list]
        e_upper_col = [data['e_upper'] for data in data_list]
        flux_tot_value_col = [data['flux_tot_value'] for data in data_list]
        flux_tot_error_col = [data['flux_tot_error'] for data in data_list]
        nobs_col = [data['nobs'] for data in data_list]
        TS_col = [data['TS'] for data in data_list]
        conv_col = [data['convergence'] for data in data_list]

        # Create FITS columns
        cols = [
            fits.Column(name='loop_item', format='20A', array=loop_items_col),
            fits.Column(name='emin', format='E', array=emin_col),
            fits.Column(name='emax', format='E', array=emax_col),
            fits.Column(name='geometric_mean', format='E', array=geometric_mean_col),
            fits.Column(name='e_lower', format='E', array=e_lower_col),
            fits.Column(name='e_upper', format='E', array=e_upper_col),
            fits.Column(name='flux_tot_value', format='E', array=flux_tot_value_col),
            fits.Column(name='flux_tot_error', format='E', array=flux_tot_error_col),
            fits.Column(name='nobs', format='E', array=nobs_col),
            fits.Column(name='TS', format='E', array=TS_col),
            fits.Column(name='convergence', format='E', array=conv_col),
        ]

        # Create HDU and write to a single FITS file for the method
        hdu = fits.BinTableHDU.from_columns(cols)
        output_fits_file = f'{general_path_for_slurm}/fit_results/{source_name_cleaned}_fit_data_{method}.fits'
        hdu.writeto(output_fits_file, overwrite=True)
        print(f"Saved FITS file for method {method}: {output_fits_file}")
    return
                   
def delete_fits_and_xml_files(source_name_cleaned, method):
    
    paths_to_delete = [
        f'{general_path_for_slurm}/data/{source_name_cleaned}/{method}/srcmap/*.fits',
        f'{general_path_for_slurm}/data/{source_name_cleaned}/{method}/models/*.xml',
        f'{general_path_for_slurm}/data/{source_name_cleaned}/{method}/ccube/*.fits',
        f'{general_path_for_slurm}/data/{source_name_cleaned}/{method}/expmap/*.fits'     
    ]

    # Iterate over each path pattern and delete all matching files
    for path_pattern in paths_to_delete:
        files = glob.glob(path_pattern)
        for file in files:
            try:
                os.remove(file)
                # print(f"Deleted: {file}")
            except OSError as e:
                print(f"Error deleting file {file}: {e}")
##################################################################################
##################################################################################

filename = f'{general_path_for_slurm}/Source_ra_dec_specin.txt'
snrratios = [3] #5, 3
time_intervals = ["week","month"]

def process_line(line):
    """Function to process a single line of the input file"""
    parts = line.strip().split()
    
    # Properly split handling quotes
    parts = shlex.split(line)

    # Extract the source name (already properly split)
    source_name = parts[0]  # No need to strip quotes, shlex handles it

    try:
        ra = float(parts[1])    # Second part: RA
        dec = float(parts[2])   # Third part: Dec
        specin = float(parts[3])  # Fourth part: spectral index
    except ValueError as e:
        print(f"Error converting values for {source_name}: {e}")
        return  # Skip this line if conversion fails

    # Run analysis steps
    check_paths(source_name, 'SNR', 7)
    check_paths(source_name, 'LIN', 7)
    check_paths(source_name, 'NONE', 7)
    

    # Construct vars_snr tuple
    vars_none = (source_name, ra, dec, "NONE", specin, None, None, 100, 1000000)
    vars_snr = (source_name, ra, dec, "SNR", specin, None, None, 100, 1000000)
    vars_lin = (source_name, ra, dec, "LIN", specin, None, None, 100, 1000000)
    source_name_cleaned = source_name.replace(" ", "").replace(".", "dot").replace("+", "plus").replace("-", "minus")

    

    filtering(vars_snr, snrratios=snrratios)
    get_gti_bin(vars_snr, snrratios=snrratios)
    generate_files(vars_snr, snrratios=snrratios, number_of_bins=7)
    source_maps(vars_snr, snrratios=snrratios)
    #print(source_name)
    #run_binned_likelihood(vars_snr, snrratios=snrratios, free_params="None")
    #print(f'Likelihood for SNR binned data done for {source_name}!')
    #delete_fits_and_xml_files(source_name_cleaned, method = "SNR")
   
    

def run_analysis():
    """Main function to use multiprocessing"""
    with open(filename, "r") as file:
        lines = file.readlines()
    num_workers = 32
    # Use multiprocessing Pool to process each line in parallel
    with multiprocessing.Pool(processes=num_workers) as pool:
        pool.map(process_line, lines)

if __name__ == "__main__":
    run_analysis()
