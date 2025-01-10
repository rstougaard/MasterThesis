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

# Function to ensure paths exist
def check_paths(source_name, method, number_of_bins):
    source_name_cleaned = source_name.replace(" ", "").replace(".", "dot").replace("+", "plus").replace("-", "minus")
    paths = [
        f'./data/{source_name_cleaned}/{method}/',
        f'./data/{source_name_cleaned}/{method}/ltcube/',
        f'./data/{source_name_cleaned}/{method}/ccube/',
        f'./data/{source_name_cleaned}/{method}/expcube/',
        f'./data/{source_name_cleaned}/{method}/expmap/',
        f'./data/{source_name_cleaned}/{method}/models/',
        f'./data/{source_name_cleaned}/{method}/srcmap/',
        f'./data/{source_name_cleaned}/{method}/CountsSpectra/',
        f'./data/{source_name_cleaned}/{method}/likeresults/',
        f'./data/{source_name_cleaned}/{method}/fit_params/',
        f'./energy_bins_def/{number_of_bins}/'
    ]
    for path in paths:
        os.makedirs(path, exist_ok=True)
        print(f"Ensured existence of: {path}")

def filtering(vars, snrratios=None, time_intervals=None):
    # Extract variables
    source_name, ra, dec, method, specin, _, _, minimal_energy, maximal_energy = vars
    source_name_cleaned = source_name.replace(" ", "").replace(".", "dot").replace("+", "plus").replace("-", "minus")
    gt = my_apps
    evc = 128
    convt = 3
    roi = 1.

    tmp_evlist = f'@./data/{source_name_cleaned}/events.list'
    tmp_gti = f'./data/{source_name_cleaned}/temp_git.fits'
    gtifilter = '(DATA_QUAL>0)&&(LAT_CONFIG==1)'
    sc = f'./data/{source_name_cleaned}/SC.fits'
    gti = f'./data/{source_name_cleaned}/gti.fits'
    ebinfile_txt = f'energy_7bins_gtbindef.txt '

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
            lc = f'./data/{source_name_cleaned}/{method}/lc_snr{loop_item}.fits'
            colors = ['blue', 'orange', 'green']
        elif method == 'LIN':
            lc = f'./data/{source_name_cleaned}/{method}/lc_{loop_item}.fits'
            colors = ['purple', 'brown']

        # Filter selecting events by time
        if not os.path.exists(gti):
            print('GTSELECT started!')
            gt.filter['evclass'] = evc
            gt.filter['evtype'] = convt
            gt.filter['ra'] = ra
            gt.filter['dec'] = dec
            gt.filter['rad'] = roi
            gt.filter['emin'] = minimal_energy
            gt.filter['emax'] = maximal_energy
            gt.filter['zmax'] = 90
            gt.filter['tmin'] = 239557417
            gt.filter['tmax'] = 435456000
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
            gt.evtbin['tstop'] = 435456000
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
            lc = f'./data/{source_name_cleaned}/{method}/lc_snr{loop_item}.fits'
            output_file_flares = f'./data/{source_name_cleaned}/{method}/flare_intervals_snr{loop_item}.txt'
            plot_file = f'./data/{source_name_cleaned}/{method}/lc_snr{loop_item}.png'
            tmp_gti_noflares = f'./data/{source_name_cleaned}/{method}/temp_git_snr{loop_item}.fits'
            gti_noflares = f'./data/{source_name_cleaned}/{method}/gti_noflares_snr{loop_item}.fits'
        elif method == 'LIN':
            lc = f'./data/{source_name_cleaned}/{method}/lc_{loop_item}.fits'
            output_file_flares = f'./data/{source_name_cleaned}/{method}/flare_intervals_{loop_item}.txt'
            plot_file = f'./data/{source_name_cleaned}/{method}/lc_{loop_item}.png'
            tmp_gti_noflares = f'./data/{source_name_cleaned}/{method}/temp_git_{loop_item}.fits'
            gti_noflares = f'./data/{source_name_cleaned}/{method}/gti_noflares_{loop_item}.fits'                

        if not os.path.exists(gti_noflares):
            ###### Here the BAD TIME intervals (flares) have to be found ######
            f_bin = fits.open(lc)
            bin_data = f_bin[1].data

            # Extract data
            X_bin = bin_data['TIME']
            Y_bin = bin_data['COUNTS'] / bin_data['EXPOSURE']  # Flux in photons/cm²/s
            time_intervals = bin_data['TIMEDEL']  # Duration of each time interval
            x_error_bin = bin_data['TIMEDEL'] / 2
            y_error_bin = bin_data['ERROR'] / bin_data['EXPOSURE']

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

            # Plot original data
            plt.errorbar(X_bin, Y_bin, xerr=x_error_bin, yerr=y_error_bin, fmt='o', capsize=5, color=color, alpha=0.3, label=f'Unfiltered {lc}')

            # Highlight removed points
            for i, (removed_x, removed_y) in enumerate(removed_points):
                plt.scatter(removed_x, removed_y, color='black', edgecolors='black')

            # Plot means and thresholds for each round
            plt.axhline(round_means[0], color='grey', linestyle='--', linewidth=3, alpha=1, label=f'Mean unfiltered')
            plt.axhline(thresholds[0], color='grey', linestyle='-', linewidth=3, alpha=1, label=f'Threshold unfiltered')
            plt.axhline(round_means[-1], color='black', linestyle='--', linewidth=3, alpha=1, label=f'Mean round {len(round_means)}')
            plt.axhline(thresholds[-1], color='black', linestyle='-', linewidth=3, alpha=1, label=f'Threshold round {len(round_means)}')

            # Customize plot
            plt.ylabel('photons/cm²/s')
            plt.xlabel('Time')
            plt.title(f'Lightcurve SNR for {lc}')
            plt.xscale('log')
            plt.yscale('log')
            plt.grid(True, which="both", linestyle="--", linewidth=0.5)

            # Move the legend outside the plot
            plt.legend(
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
            gt.filter['rad'] = roi
            gt.filter['emin'] = minimal_energy
            gt.filter['emax'] = maximal_energy
            gt.filter['zmax'] = 90
            gt.filter['tmin'] = 239557417
            gt.filter['tmax'] = 435456000
            gt.filter['infile'] = tmp_evlist
            gt.filter['outfile'] = tmp_gti_noflares
            gt.filter.run() #run GTSELECT
            print( 'GTSELCT finished!' )
            ############################## modify GTIs ##############################
            # here apoastra was a text file with 2 columns -- start stop times. Only those intervals will be used to extract spectrum
            # you could use method='out' to exclude some intervals, i.e. extract spectrum from everything *except* these intervals
            # times in the file can be either in MJDs or in Fermi seconds (use times_in_mjds=False)
        

            UpdateGTIs(tmp_gti_noflares, output_file_flares, method='out', times_in_mjd=False) #defined in operate_gtis.py, pls see also there

            ############### actual filtering according to updated GTIs ###############
            # just incase -- modified GTIs include old "standard GTIs" as well, so no need to do this twice
            
            print( 'GTMKTIME start' )
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
            print(f'{gti_noflares} file exists!')



    # Print results for all light curves
    for item, initial_mean, final_mean, final_threshold, valid_intervals, invalid_intervals in zip(
        loop_items, initial_means, final_means, final_thresholds, valid_intervals_list, invalid_intervals_list
    ):
        print(f"{item}: Initial Mean {initial_mean:.2e} |Final Mean {final_mean:.2e} | Final Threshold {final_threshold:.2e} | "
            f"Valid intervals: {valid_intervals} | Invalid intervals: {invalid_intervals}")

#####################
def get_spectral_points(vars, snrratios=None, time_intervals=None):
    # Extract variables
    source_name, ra, dec, method, specin, _, _, minimal_energy, maximal_energy = vars
    source_name_cleaned = source_name.replace(" ", "").replace(".", "dot").replace("+", "plus").replace("-", "minus")
    gt = my_apps
    ebinfile_txt = f'./energy_7bins_gtbindef.txt'
    evc = 128
    convt = 3
    roi = 1.

    gtifilter = '(DATA_QUAL>0)&&(LAT_CONFIG==1)'
    sc = f'./data/{source_name_cleaned}/SC.fits'
    # Determine the loop items based on the method
    if method == "SNR":
        loop_items = snrratios
    elif method == "LIN":
        loop_items = time_intervals
    else:
        loop_items = "NONE"  # No looping for the "NONE" method

    # If there is nothing to loop over, handle the "NONE" method directly
    if method == "NONE":
        temp_gti_noflares_bin = f'./data/{source_name_cleaned}/{method}/temp_gti_noflares_snr{loop_item}_{emin}_{emax}.fits'
        gti_noflares_bin = f'./data/{source_name_cleaned}/{method}/gti_noflares_snr{loop_item}_{emin}_{emax}.fits'
        lc_noflare_bin = f'./data/{source_name_cleaned}/{method}/lc_snr{loop_item}_{emin}_{emax}.fits'
        gti_noflares = f'./data/{source_name_cleaned}/{method}/gti_noflares_{loop_item}.fits'
       
    else:
        for loop_item in loop_items:

            with open(f'{ebinfile_txt}', 'r') as file:
                for line in file:
                    line = line.strip()
                    if not line or len(line.split()) != 2:
                        continue
                    emin_float, emax_float = map(float, line.split())

                    # Convert the values to integers for the filename
                    emin = int(emin)
                    emax = int(emax)
                    print(f"Processing {method}: {loop_item} {emin_float}MeV - {emax_float}MeV")

                    if method == 'SNR':
                        temp_gti_noflares_bin = f'./data/{source_name_cleaned}/{method}/temp_gti_noflares_snr{loop_item}_{emin}_{emax}.fits'
                        gti_noflares_bin = f'./data/{source_name_cleaned}/{method}/gti_noflares_snr{loop_item}_{emin}_{emax}.fits'
                        lc_noflare_bin = f'./data/{source_name_cleaned}/{method}/lc_snr{loop_item}_{emin}_{emax}.fits'
                        gti_noflares = f'./data/{source_name_cleaned}/{method}/gti_noflares_snr{loop_item}.fits'
                    elif method == 'LIN':
                        temp_gti_noflares_bin = f'./data/{source_name_cleaned}/{method}/temp_gti_noflares_snr{loop_item}_{emin}_{emax}.fits'
                        gti_noflares_bin = f'./data/{source_name_cleaned}/{method}/_gti_noflares_snr{loop_item}_{emin}_{emax}.fits'
                        lc_noflare_bin = f'./data/{source_name_cleaned}/{method}/lc_{loop_item}_{emin}_{emax}.fits'
                        gti_noflares = f'./data/{source_name_cleaned}/{method}/gti_noflares_{loop_item}.fits'
                    
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
                        gt.filter['tmax'] = 435456000
                        gt.filter['infile'] = gti_noflares
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

                    # Create light curve
                    print('Creating LC')
                    always_redo_exposure = True
                    gt.evtbin['evfile'] = gti_noflares_bin
                    gt.evtbin['outfile'] = lc_noflare_bin
                    gt.evtbin['scfile'] = sc
                    gt.evtbin['algorithm'] = 'LC'
                    gt.evtbin['tbinalg'] = 'LIN'
                    gt.evtbin['tstart'] = 239557417
                    gt.evtbin['tstop'] = 435456000
                    gt.evtbin['emin'] = emin
                    gt.evtbin['emax'] = emax
                    gt.evtbin['ebinalg'] = "NONE"
                    gt.evtbin['ebinfile'] = "NONE"
                    gt.evtbin['dtime'] = (435456000 - 239557417)
                    gt.evtbin.run()
                    

                    print(f'LC created for {method}: {loop_item} {emin}MeV - {emax}MeV')

                    calc_exposure = True
                    with fits.open(lc_noflare_bin) as f:
                        if('EXPOSURE' in f[1].data.names): calc_exposure=False

                    if(calc_exposure or always_redo_exposure):
                        print('Launching gtexposure for ',lc_noflare_bin)
                        gtexposure = my_apps.GtApp('gtexposure')
                        gtexposure['infile'] = gti_noflares_bin
                        gtexposure['scfile'] = sc
                        gtexposure['irfs'] = 'CALDB'
                        gtexposure['specin'] = -specin
                        gtexposure['apcorr'] = 'yes' #change this, if you are sure
                        gtexposure['enumbins'] = 30
                        gtexposure['emin'] = emin
                        gtexposure['emax'] = emax
                        gtexposure['ra'] = ra
                        gtexposure['dec'] = dec
                        gtexposure['rad'] = roi
                        gtexposure.run()
                    else:
                        print('EXPOSURE column already exists!')
                        print('If you want to re-create it, launch with always_redo_exposure=True')
                else:
                    print(f'{lc_noflare_bin} file exists!')

            
##################################################################################
##################################################################################
def modify_and_save(tree, source_name, method, loop_item):
    source_name_cleaned = source_name.replace(" ", "").replace(".", "dot").replace("+", "plus").replace("-", "minus")
    if method == "NONE":
        output_dir = f'./data/{source_name_cleaned}/{method}/models/'
    elif method == "SNR":
        output_dir = f'./data/{source_name_cleaned}/{method}/models/'
    elif method == "LIN":
        output_dir = f'./data/{source_name_cleaned}/{method}/models/'

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
                    output_path = os.path.join(output_dir, f"input_model_{suffix}.xml")
                elif method == "SNR":
                    output_path = os.path.join(output_dir, f"input_model_snr{loop_item}_{suffix}.xml")
                elif method == "LIN":
                   output_path = os.path.join(output_dir, f"input_model_{loop_item}_{suffix}.xml")

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

    general_path = f'./data/{source_name_cleaned}/'
    sc = general_path+'SC.fits'
    ebinfile = f'./energy_bins_def/{number_of_bins}/energy_bins_gtbindef.fits'

    # Determine the loop items based on the method
    if method == "SNR":
        loop_items = snrratios
    elif method == "LIN":
        loop_items = time_intervals
    else:
        loop_items = "NONE"  # No looping for the "NONE" method

    # If there is nothing to loop over, handle the "NONE" method directly
    if method == "NONE":
        gti_noflares = general_path + 'gti.fits'
        ltcube = general_path + f'{method}/ltcube/ltcube.fits'
        ccube = general_path + f'{method}/ccube/ccube.fits'
        binexpmap = general_path + f'{method}/expmap/BinnedExpMap.fits'
        model = f'./data/{source_name_cleaned}/{method}/models/input_model.xml'
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
            my_apps.evtbin['proj'] = 'AIT'
            my_apps.evtbin['ebinalg'] = 'FILE'
            my_apps.evtbin['ebinfile'] = ebinfile
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
            expCube2['proj'] = 'AIT'
            expCube2['ebinalg'] = 'FILE'
            expCube2['ebinfile'] = ebinfile
            expCube2.run()
        else:
            print(f'{binexpmap} file exists!')
    
        ####### Make model #######
        ##### Run make4FGLxml Command #####
        make4FGLxml_command = [f'make4FGLxml ./data/gll_psc_v32.xml --event_file {gti_noflares} -o {model} --free_radius 5.0 --norms_free_only True --sigma_to_free 25 --variable_free True']
    
        # Run the command using subprocess
        subprocess.run(make4FGLxml_command, shell=True, check=True, executable='/bin/bash')
        tree = ET.parse(f'{model}')
        modify_and_save(tree, source_name, method, None)
    else:
        for loop_item in loop_items:
            
            if method == "SNR":
                gti_noflares = general_path + f'{method}/gti_noflares_snr{loop_item}.fits'
                ltcube = general_path + f'{method}/ltcube/ltcube_snr{loop_item}.fits'
                ccube = general_path + f'{method}/ccube/ccube_snr{loop_item}.fits'
                binexpmap = general_path + f'{method}/expmap/BinnedExpMap_snr{loop_item}.fits'
                model = f'./data/{source_name_cleaned}/{method}/models/input_model_snr{loop_item}.xml'
            elif method == "LIN":
                gti_noflares = general_path + f'{method}/gti_noflares_{loop_item}.fits'
                ltcube = general_path + f'{method}/ltcube/ltcube_{loop_item}.fits'
                ccube = general_path + f'{method}/ccube/ccube_{loop_item}.fits'
                binexpmap = general_path + f'{method}/expmap/BinnedExpMap_{loop_item}.fits'
                model = f'./data/{source_name_cleaned}/{method}/models/input_model_{loop_item}.xml'

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
                my_apps.evtbin['proj'] = 'AIT'
                my_apps.evtbin['ebinalg'] = 'FILE'
                my_apps.evtbin['ebinfile'] = ebinfile
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
                expCube2['proj'] = 'AIT'
                expCube2['ebinalg'] = 'FILE'
                expCube2['ebinfile'] = ebinfile
                expCube2.run()
            else:
                print(f'{binexpmap} file exists!')
        
            ####### Make model #######
            ##### Run make4FGLxml Command #####
            make4FGLxml_command = [f'make4FGLxml ./data/gll_psc_v32.xml --event_file {gti_noflares} -o {model} --free_radius 5.0 --norms_free_only True --sigma_to_free 25 --variable_free True']
        
            # Run the command using subprocess
            subprocess.run(make4FGLxml_command, shell=True, check=True, executable='/bin/bash')
            tree = ET.parse(f'{model}')
            modify_and_save(tree, source_name, method, loop_item)

    return

def source_maps(vars, snrratios=None, time_intervals=None):
    # Extract variables
    source_name, ra, dec, method, specin, _, _, minimal_energy, maximal_energy = vars
    source_name_cleaned = source_name.replace(" ", "").replace(".", "dot").replace("+", "plus").replace("-", "minus")

    general_path = f'./data/{source_name_cleaned}/'
    # Determine the loop items based on the method
    if method == "SNR":
        loop_items = snrratios
    elif method == "LIN":
        loop_items = time_intervals
    else:
        loop_items = "NONE"  # No looping for the "NONE" method
    
    input_model = general_path + f'NONE/models/input_model.xml'
    
    # If there is nothing to loop over, handle the "NONE" method directly
    if method == "NONE":
        ltcube = general_path + f'{method}/ltcube/ltcube.fits'
        ccube = general_path + f'{method}/ccube/ccube.fits'
        binexpmap = general_path + f'{method}/expmap/BinnedExpMap.fits'
        srcmap = general_path + f'{method}/srcmap/srcmap.fits'
        #input_model = general_path + f'{method}/expmap/input_model.xml'
        print(f"Processing method {method} without looping.")
        
        if not os.path.exists(srcmap):
            ####### Source Map #######
            print(f"Creating sourcemap for {method}")
            my_apps.srcMaps['expcube'] = ltcube
            my_apps.srcMaps['cmap'] = ccube
            my_apps.srcMaps['srcmdl'] = input_model
            my_apps.srcMaps['bexpmap'] = binexpmap
            my_apps.srcMaps['outfile'] = srcmap
            my_apps.srcMaps['irfs'] = 'P8R3_SOURCE_V3'
            my_apps.srcMaps['evtype'] = '3'
            my_apps.srcMaps.run()
        else:
            print(f'{srcmap} file exists!')
    else:
        for loop_item in loop_items:
            if method == "SNR":
                ltcube = general_path + f'{method}/ltcube/ltcube_snr{loop_item}.fits'
                ccube = general_path + f'{method}/ccube/ccube_snr{loop_item}.fits'
                binexpmap = general_path + f'{method}/expmap/BinnedExpMap_snr{loop_item}.fits'
                srcmap = general_path + f'{method}/srcmap/srcmap_snr{loop_item}.fits'
                #input_model = general_path + f'{method}/expmap/input_model_snr{loop_item}.xml'
            elif method == "LIN":
                ltcube = general_path + f'{method}/ltcube/ltcube_{loop_item}.fits'
                ccube = general_path + f'{method}/ccube/ccube_{loop_item}.fits'
                binexpmap = general_path + f'{method}/expmap/BinnedExpMap_{loop_item}.fits'
                srcmap = general_path + f'{method}/srcmap/srcmap_{loop_item}.fits'
                #input_model = general_path + f'{method}/expmap/input_model_{loop_item}.xml'

            if not os.path.exists(srcmap):
                ####### Source Map #######
                print(f"Creating sourcemap for {method}: {loop_item}")
                my_apps.srcMaps['expcube'] = ltcube
                my_apps.srcMaps['cmap'] = ccube
                my_apps.srcMaps['srcmdl'] = input_model
                my_apps.srcMaps['bexpmap'] = binexpmap
                my_apps.srcMaps['outfile'] = srcmap
                my_apps.srcMaps['irfs'] = 'P8R3_SOURCE_V3'
                my_apps.srcMaps['evtype'] = '3'
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

    general_path = f'./data/{source_name_cleaned}/'
    method_results = {}
    # Determine the loop items based on the method
    if method == "SNR":
        loop_items = snrratios
    elif method == "LIN":
        loop_items = time_intervals
    else:
        loop_items = "NONE"  # No looping for the "NONE" method

    # If there is nothing to loop over, handle the "NONE" method directly
    if method == "NONE":
        ltcube = general_path + f'{method}/ltcube/ltcube.fits'
        ccube = general_path + f'{method}/ccube/ccube.fits'
        binexpmap = general_path + f'{method}/expmap/BinnedExpMap.fits'
        srcmap = general_path + f'{method}/srcmap/srcmap.fits'
        if free_params == "alpha":
            input_model = general_path + f'{method}/models/input_model_free_alpha.xml'
            cspectra = general_path + f'{method}/CountsSpectra/cspectra_free_alpha.fits'
            writexml = general_path + f'{method}/fit_params/fit_free_alpha.xml'
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
        print(f"Processing method {method} without looping.")
        
        #Likelihood here
    else:
        for loop_item in loop_items:
            if method == "SNR":
                ltcube = general_path + f'{method}/ltcube/ltcube_snr{loop_item}.fits'
                ccube = general_path + f'{method}/ccube/ccube_snr{loop_item}.fits'
                binexpmap = general_path + f'{method}/expmap/BinnedExpMap_snr{loop_item}.fits'
                srcmap = general_path + f'{method}/srcmap/srcmap_snr{loop_item}.fits'
                if free_params == "alpha":
                    input_model = general_path + f'{method}/models/input_model_snr{loop_item}_free_alpha.xml'
                    cspectra = general_path + f'{method}/CountsSpectra/cspectra_snr{loop_item}_free_alpha.fits'
                    writexml = general_path + f'{method}/fit_params/fit_snr{loop_item}_free_alpha.xml'
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
                ccube = general_path + f'{method}/ccube/ccube_{loop_item}.fits'
                binexpmap = general_path + f'{method}/expmap/BinnedExpMap_{loop_item}.fits'
                srcmap = general_path + f'{method}/srcmap/srcmap_{loop_item}.fits'
                if free_params == "alpha":
                    input_model = general_path + f'{method}/models/input_model_{loop_item}_free_alpha.xml'
                    cspectra = general_path + f'{method}/CountsSpectra/cspectra_{loop_item}_free_alpha.fits'
                    writexml = general_path + f'{method}/fit_params/fit_{loop_item}_free_alpha.xml'
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
                obs = BinnedObs(srcMaps=srcmap, binnedExpMap=binexpmap, expCube=ltcube, irfs='CALDB')
                like = BinnedAnalysis(obs, input_model, optimizer='NewMinuit')
                likeobj = pyLikelihood.NewMinuit(like.logLike)
                like.fit(verbosity=0, covar=True, optObject=likeobj)
                like.writeCountsSpectra(cspectra) 
                like.logLike.writeXml(writexml)
                tree = ET.parse(writexml)
                #root = tree.getroot()

                flux_tot_value = like.flux(source_name, emin=minimal_energy, emax=maximal_energy)
                flux_tot_error = like.fluxError(source_name, emin=minimal_energy, emax=maximal_energy)
                alpha = like.model[source_name].funcs['Spectrum'].getParam('alpha').value()
                alpha_err = like.model[source_name].funcs['Spectrum'].getParam('alpha').error()
                beta = like.model[source_name].funcs['Spectrum'].getParam('beta').value()
                beta_err = like.model[source_name].funcs['Spectrum'].getParam('beta').error()
                Eb = like.model[source_name].funcs['Spectrum'].getParam('Eb').value()
                Eb_err = like.model[source_name].funcs['Spectrum'].getParam('Eb').error()

                E = (like.energies[:-1] + like.energies[1:]) / 2.
                nobs = like.nobs

                fit_data = {
                    'loop_item': loop_item,
                    'flux_tot_value': float(flux_tot_value),
                    'flux_tot_error': float(flux_tot_error),
                    'alpha_value': float(alpha),
                    'alpha_error': float(alpha_err),
                    'beta_value': float(beta),
                    'beta_error': float(beta_err),
                    'Eb_value': float(Eb),
                    'Eb_error': float(Eb_err),
                    'E_points': E.tolist(),
                    'nobs': list(nobs),
                }

            except Exception as e:
                print(f"Error processing {method} loop_item {loop_item}: {e}")
                fit_data = {
                    'loop_item': loop_item,
                    'flux_tot_value': None,
                    'flux_tot_error': None,
                    'alpha_value': None,
                    'alpha_error': None,
                    'beta_value': None,
                    'beta_error': None,
                    'Eb_value': None,
                    'Eb_error': None,
                    'E_points': None,
                    'nobs': None,
                }

            # Collect results for the method
            if method not in method_results:
                method_results[method] = []
            method_results[method].append(fit_data)

        # Save results for the source to a FITS file
        save_source_results_to_fits(source_name, method_results, results_output_file)
    return
                   

##################################################################################
##################################################################################
check_paths("4FGL J0319.8+4130", 'SNR', 7)
check_paths("4FGL J0319.8+4130", 'LIN', 7)
check_paths("4FGL J0319.8+4130", 'NONE', 7)
'''
filename = "Source_ra_dec_specin.txt"

with open(filename, "r") as file:
    for line in file:
        # Strip any extra whitespace and extract components
        parts = line.strip().split()
        
        # Extract required data
        source_name = parts[0]  # First part: source name
        ra = float(parts[1])    # Second part: RA
        dec = float(parts[2])   # Third part: Dec
        specin = float(parts[3])  # Fourth part: spectral index
        
        # Construct vars_snr tuple
        vars_snr = (source_name, ra, dec, "SNR", specin, None, None, 100, 1000000)
'''
vars_snr = ("4FGL J0319.8+4130", 49.9507, 41.5117,"SNR", 2.05, None, None,  100, 1000000)
snrratios = [3, 5, 10]
vars_lin = ("4FGL J0319.8+4130", 49.9507, 41.5117,"LIN", 2.05, None, None,  100, 1000000)
time_intervals = ["week", "month"]
vars_none= ("4FGL J0319.8+4130", 49.9507, 41.5117, "NONE", 2.05, None, None,  100, 1000000)
filtering(vars_snr, snrratios=snrratios)
filtering(vars_lin, time_intervals=time_intervals)
print('Filtering done!')
get_spectral_points(vars_snr, snrratios=snrratios)
get_spectral_points(vars_lin, time_intervals=vars_lin)
get_spectral_points(vars_none)


generate_files(vars_none, number_of_bins=7)
source_maps(vars_none)

generate_files(vars_snr, snrratios=snrratios, number_of_bins=7)
# mangler input model
source_maps(vars_snr, snrratios=snrratios)

generate_files(vars_lin, time_intervals=time_intervals, number_of_bins=7)
source_maps(vars_lin, time_intervals=time_intervals)


    
'''
def run_analysis(source_name, num_workers, method, snrratio, time_interval_name, ra, dec, minimal_energy, maximal_energy, number_of_bins, bins_def_filename):
    
    #snr_arg = source_name, method, snrratio, time_interval_name, ra, dec, minimal_energy, maximal_energy
    #filtering(snr_arg)
    

# Function to read energy bins
def get_energy_bins(bins_def_filename):
    energy_bins = []
    with open(f'{bins_def_filename}.txt', 'r') as file:
        for line in file:
            line = line.strip()
            if not line or len(line.split()) != 2:
                continue
            emin, emax = map(float, line.split())
            energy_bins.append((emin, emax))
    return energy_bins

def run_binned_likelihood_free_alpha(vars):
    i, source_name, time_interval_name, ra, dec, minimal_energy, maximal_energy, number_of_bins = vars
    source_name_cleaned = source_name.replace(" ", "").replace(".", "dot").replace("+", "plus").replace("-", "minus")

    ccube = f'./data/{source_name_cleaned}/snr/ccube/ccube_{i}.fits'
    ltcube = f'./data/{source_name_cleaned}/snr/ltcube/ltcube_{i}.fits'
    expmap = f'./data/{source_name_cleaned}/snr/expmap/BinnedExpMap_{i}.fits'
    model = f'./data/{source_name_cleaned}/snr/models/input_model_{i}_free_alpha.xml'
    srcmap = f'./data/{source_name_cleaned}/snr/srcmap/srcmap_{i}_free_alpha.fits'
    ####### Binned Likelihood Analysis #######
    try:
        obs = BinnedObs(
            srcMaps=srcmap,
            binnedExpMap=expmap,
            expCube=ltcube,
            irfs='CALDB'
        )
        
        like = BinnedAnalysis(obs, model, optimizer='NewMinuit')
        likeobj = pyLikelihood.NewMinuit(like.logLike)
        like.fit(verbosity=0, covar=True, optObject=likeobj)
        
        log_likelihood = like.logLike.value()
        TS = like.Ts(source_name)
        convergence = likeobj.getRetCode()

        like.writeCountsSpectra(f"./data/{source_name_cleaned}/snr/CountsSpectra/spectra_free_alpha_{i}.fits") 
        like.logLike.writeXml(f'./data/{source_name_cleaned}/snr/fit_params/fit_free_alpha_{i}.xml')
        tree = ET.parse(f'./data/{source_name_cleaned}/snr/fit_params/fit_free_alpha_{i}.xml')
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
        fit_data_free_alpha = {
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

    except Exception as e:
        # Catch any exception and print relevant information
        print(f"Error in iteration: time_interval {i}.")
        print(f"Exception: {str(e)}")

        fit_data_free_alpha = {
        f'{time_interval_name}': i,
        'flux_tot_value': None,  
        'flux_tot_error': None,   
        'alpha_value': None,  
        'alpha_error': None, 
        'beta_value': None,  
        'beta_error': None, 
        'Eb_value': None,  
        'Eb_error': None,  
        'convergence': None,
        'E_points': None,  
        'nobs': None,  
        }

    with open(f'./data/{source_name_cleaned}/snr/likeresults/flux_time_interval_{i}_free_alpha.json', 'w') as f:
        json.dump(fit_data_free_alpha, f, indent=4)
    
    print(f"Saved flux data for free alpha full spectrum {i}!")

def run_binned_likelihood_free_beta(vars):
    i, source_name, time_interval_name, ra, dec, minimal_energy, maximal_energy, number_of_bins = vars
    source_name_cleaned = source_name.replace(" ", "").replace(".", "dot").replace("+", "plus").replace("-", "minus")

    ccube = f'./data/{source_name_cleaned}/snr/ccube/ccube_{i}.fits'
    ltcube = f'./data/{source_name_cleaned}/snr/ltcube/ltcube_{i}.fits'
    expmap = f'./data/{source_name_cleaned}/snr/expmap/BinnedExpMap_{i}.fits'
    model = f'./data/{source_name_cleaned}/snr/models/input_model_{i}_free_beta.xml'
    srcmap = f'./data/{source_name_cleaned}/snr/srcmap/srcmap_{i}_free_beta.fits'

    ####### Binned Likelihood Analysis #######
    try:
        obs = BinnedObs(
            srcMaps=srcmap,
            binnedExpMap=expmap,
            expCube=ltcube,
            irfs='CALDB'
        )
        
        like = BinnedAnalysis(obs, model, optimizer='NewMinuit')
        likeobj = pyLikelihood.NewMinuit(like.logLike)
        like.fit(verbosity=0, covar=True, optObject=likeobj)
        
        log_likelihood = like.logLike.value()
        TS = like.Ts(source_name)
        convergence = likeobj.getRetCode()

        like.writeCountsSpectra(f"./data/{source_name_cleaned}/snr/CountsSpectra/spectra_free_beta_{i}.fits") 
        like.logLike.writeXml(f'./data/{source_name_cleaned}/snr/fit_params/fit_free_beta_{i}.xml')
        tree = ET.parse(f'./data/{source_name_cleaned}/snr/fit_params/fit_free_beta_{i}.xml')
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
        fit_data_free_beta = {
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

    except Exception as e:
        # Catch any exception and print relevant information
        print(f"Error in iteration: time_interval {i}.")
        print(f"Exception: {str(e)}")

        fit_data_free_beta = {
        f'{time_interval_name}': i,
        'flux_tot_value': None,  
        'flux_tot_error': None,   
        'alpha_value': None,  
        'alpha_error': None, 
        'beta_value': None,  
        'beta_error': None, 
        'Eb_value': None,  
        'Eb_error': None,  
        'convergence': None,
        'E_points': None,  
        'nobs': None,  
        }

    with open(f'./data/{source_name_cleaned}/snr/likeresults/flux_time_interval_{i}_free_beta.json', 'w') as f:
        json.dump(fit_data_free_beta, f, indent=4)
    
    print(f"Saved flux data for free beta full spectrum {i}!")

def save_flux_fit_data(source_name_cleaned, time_interval_name, num_intervals):
    combined_fit_data_free_alpha = {}
    combined_fit_data_free_beta = {}
    # Iterate over all time intervals to load the flux fit data
    for i in range(num_intervals):
        flux_file_free_alpha = f'./data/{source_name_cleaned}/snr/likeresults/flux_time_interval_{i}_free_alpha.json'

        if os.path.exists(flux_file_free_alpha):
            # Load the existing JSON data
            with open(flux_file_free_alpha, 'r') as f:
                fit_data = json.load(f)

                # Add the loaded fit_data to the combined dictionary
                combined_fit_data_free_alpha[f"time_interval_{i}"] = fit_data

            # Delete the JSON file after loading
            os.remove(flux_file_free_alpha)
            print(f"Deleted JSON file: {flux_file_free_alpha}")
    
     # Save the combined fit data as a .json file
    output_directory = f'./data/{source_name_cleaned}/snr/npy_files/'
    os.makedirs(output_directory, exist_ok=True)

    combined_json_path_alpha = os.path.join(output_directory, f'combined_fit_data_free_alpha.json')
    with open(combined_json_path_alpha, 'w') as f:
        json.dump(combined_fit_data_free_alpha, f, indent=4)

    # Save the combined fit data as a .npy file
    combined_npy_path_alpha = os.path.join(output_directory, f'combined_fit_data_free_alpha.npy')
    np.save(combined_npy_path_alpha, combined_fit_data_free_alpha)

    print(f"Combined fit data saved as JSON: {combined_json_path_alpha}")
    print(f"Combined fit data saved as NPY: {combined_npy_path_alpha}")

    for i in range(num_intervals):
        flux_file_free_beta = f'./data/{source_name_cleaned}/snr/likeresults/flux_time_interval_{i}_free_beta.json'

        if os.path.exists(flux_file_free_beta):
            # Load the existing JSON data
            with open(flux_file_free_beta, 'r') as f:
                fit_data = json.load(f)

                # Add the loaded fit_data to the combined dictionary
                combined_fit_data_free_beta[f"time_interval_{i}"] = fit_data

            # Delete the JSON file after loading
            os.remove(flux_file_free_beta)
            print(f"Deleted JSON file: {flux_file_free_beta}")

    # Save the combined fit data as a .json file
    output_directory = f'./data/{source_name_cleaned}/LC_{time_interval_name}/npy_files/'
    os.makedirs(output_directory, exist_ok=True)

    combined_json_path_beta = os.path.join(output_directory, f'combined_fit_data_free_beta_{time_interval_name}.json')
    with open(combined_json_path_beta, 'w') as f:
        json.dump(combined_fit_data_free_beta, f, indent=4)

    # Save the combined fit data as a .npy file
    combined_npy_path_beta = os.path.join(output_directory, f'combined_fit_data_free_beta_{time_interval_name}.npy')
    np.save(combined_npy_path_beta, combined_fit_data_free_beta)

    print(f"Combined fit data saved as JSON: {combined_json_path_beta}")
    print(f"Combined fit data saved as NPY: {combined_npy_path_beta}")

    return combined_fit_data_free_alpha

###############################################################################################################################################################
###############################################################################################################################################################
# Function to generate other necessary files (per energy bin)
def generate_files_per_bin(vars):
    i, source_name, time_interval_name, ra, dec, short_name, emin, emax, energy_bin_index, number_of_bins = vars
    source_name_cleaned = source_name.replace(" ", "").replace(".", "dot").replace("+", "plus").replace("-", "minus")

    # Create energy bin definition file for this bin
    energy_bin_txt = f'./energy_bins_def/{number_of_bins}/energy_bin_{i}_{energy_bin_index}.txt'
    with open(energy_bin_txt, 'w') as f:
        f.write(f'{emin}   {emax}\n')

    # Create the energy bin FITS file
    gtbindef_energy_command = [
        'gtbindef',
        'E',
        energy_bin_txt,
        f'./energy_bins_def/{number_of_bins}/energy_bin_{i}_{energy_bin_index}.fits',
        'MeV']
    subprocess.run(gtbindef_energy_command, check=True)

    event_file = f'./data/{source_name_cleaned}/snr/time_interval_{i}.fits'
    ccube = f'./data/{source_name_cleaned}/snr/ccube/ccube_{i}_bin_{energy_bin_index}.fits'
    ltcube = f'./data/{source_name_cleaned}/snr/ltcube/ltcube_{i}.fits'
    expmap = f'./data/{source_name_cleaned}/snr/expmap/BinnedExpMap_{i}_bin_{energy_bin_index}.fits'

    ####### Counts Cube #######
    my_apps.evtbin['evfile'] = event_file
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
    my_apps.evtbin['proj'] = 'AIT'
    my_apps.evtbin['ebinalg'] = 'FILE'
    my_apps.evtbin['ebinfile'] = f'./energy_bins_def/{number_of_bins}/energy_bin_{i}_{energy_bin_index}.fits'
    my_apps.evtbin.run()

    ####### Exposure Map #######
    expCube2['infile'] = ltcube
    expCube2['cmap'] = 'none'
    expCube2['outfile'] = expmap
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
    expCube2['ebinfile'] = f'./energy_bins_def/{number_of_bins}/energy_bin_{i}_{energy_bin_index}.fits'
    expCube2.run()
    

    ####### Make model #######
    ##### Run make4FGLxml Command #####
    model = f'./data/{source_name_cleaned}/snr/models/input_model_{i}_bin_{energy_bin_index}.xml'
    make4FGLxml_command = [
        f'make4FGLxml ./data/gll_psc_v32.xml --event_file {event_file} '
        f'-o {model} --free_radius 5.0 --norms_free_only True '
        f'--sigma_to_free 25 --variable_free True']
    
    subprocess.run(make4FGLxml_command, shell=True, check=True, executable='/bin/bash')

    # Modify the generated XML file
    tree = ET.parse(f'./data/{source_name_cleaned}/snr/models/input_model_{i}_bin_{energy_bin_index}.xml')
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
                    #print(f"Changing 'free' attribute for {param_name}")
                    param.set('free', '0') #Has to be fixed for likelihood per bin, equal 0

        # Save the modified XML back to the file
        tree.write(f'./data/{source_name_cleaned}/snr/models/input_model_{i}_bin_{energy_bin_index}.xml', encoding='utf-8', xml_declaration=True)
    pass 

# Function to generate source maps
def source_maps_per_bin(vars):
    i, source_name, time_interval_name, ra, dec, short_name, emin, emax, energy_bin_index, number_of_bins = vars
    source_name_cleaned = source_name.replace(" ", "").replace(".", "dot").replace("+", "plus").replace("-", "minus")
    ltcube = f'./data/{source_name_cleaned}/snr/ltcube/ltcube_{i}.fits'
    ccube = f'./data/{source_name_cleaned}/snr/ccube/ccube_{i}_bin_{energy_bin_index}.fits'
    model = f'./data/{source_name_cleaned}/snr/models/input_model_{i}_bin_{energy_bin_index}.xml'
    expmap = f'./data/{source_name_cleaned}/snr/expmap/BinnedExpMap_{i}_bin_{energy_bin_index}.fits'
    scrmap = f'./data/{source_name_cleaned}/snr/srcmap/srcmap_{i}_bin_{energy_bin_index}.fits'
    ####### Source Map #######
    my_apps.srcMaps['expcube'] = ltcube
    my_apps.srcMaps['cmap'] = ccube
    my_apps.srcMaps['srcmdl'] = model
    my_apps.srcMaps['bexpmap'] = expmap
    my_apps.srcMaps['outfile'] = scrmap
    my_apps.srcMaps['irfs'] = 'P8R3_SOURCE_V3'
    my_apps.srcMaps['evtype'] = '3'
    my_apps.srcMaps.run()
    pass

# Function to run binned likelihood analysis
def run_binned_likelihood_per_bin(vars):
    i, source_name, time_interval_name, ra, dec, short_name, emin, emax, energy_bin_index, number_of_bins = vars
    source_name_cleaned = source_name.replace(" ", "").replace(".", "dot").replace("+", "plus").replace("-", "minus")

    ltcube = f'./data/{source_name_cleaned}/snr/ltcube/ltcube_{i}.fits'
    model = f'./data/{source_name_cleaned}/snr/models/input_model_{i}_bin_{energy_bin_index}.xml'
    expmap = f'./data/{source_name_cleaned}/snr/expmap/BinnedExpMap_{i}_bin_{energy_bin_index}.fits'
    scrmap = f'./data/{source_name_cleaned}/snr/srcmap/srcmap_{i}_bin_{energy_bin_index}.fits'

    try:
        obs = BinnedObs(
            srcMaps=scrmap,
            binnedExpMap=expmap,
            expCube=ltcube,
            irfs='CALDB'
        )

        like = BinnedAnalysis(obs, model, optimizer='NewMinuit')
        likeobj = pyLikelihood.NewMinuit(like.logLike)

        # Perform the likelihood fitting
        like.fit(verbosity=0, covar=True, optObject=likeobj)

        # Write Counts Spectra and XML files
        like.writeCountsSpectra(f"./data/{source_name_cleaned}/snr/CountsSpectra/spectra_{i}_bin_{energy_bin_index}.fits")
        like.logLike.writeXml(f'./data/{source_name_cleaned}/snr/fit_params/fit_{i}_bin_{energy_bin_index}.xml')

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


    except Exception as e:
        # Catch any exception and print relevant information
        print(f"Error in iteration: time_interval {i}, bin {energy_bin_index}")
        print(f"Exception: {str(e)}")

        # Set fit_data with None values in case of an error
        fit_data = {
            'time_interval': i,
            'int_flux': None,
            'int_flux_error': None,
            'emin': emin,
            'emax': emax,
            'E_av': None,
            'E_minus_error': None,
            'E_plus_error': None,
            'dFdE': None,
            'dFdE_error': None,
            'nobs': None,
        }

    output_file = f'./data/{source_name_cleaned}/snr/likeresults/flux_{i}_bin_{energy_bin_index}.json'
    with open(output_file, 'w') as f:
        json.dump(fit_data, f, indent=4)


def combine_flux_data_per_time_interval(source_name_cleaned, time_interval_name, num_intervals, num_bins):
    all_intervals_combined_data = []

    # Iterate over all time intervals
    for i in range(num_intervals):
        combined_data_for_interval = []

        # Iterate over all bins for a given time interval to load data
        for bin_num in range(num_bins):
            flux_file = f'./data/{source_name_cleaned}/snr/likeresults/flux_{i}_bin_{bin_num}.json'

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
                        "nobs": data.get("nobs") if isinstance(data.get("nobs"), (int, float)) else 0.0
                    })

                # Delete the JSON file after reading
                #os.remove(flux_file)
                print(f"not Deleted JSON file: {flux_file}")

        # Add the interval data if it has values
        if combined_data_for_interval:
            all_intervals_combined_data.append(combined_data_for_interval)

    # Save the data as a pickle file to preserve the original structure
    output_directory = f'./data/{source_name_cleaned}/snr/npy_files/'
    os.makedirs(output_directory, exist_ok=True)

    output_file = os.path.join(output_directory, 'all_intervals_combined_data.pkl')
    with open(output_file, 'wb') as f:
        pickle.dump(all_intervals_combined_data, f)

    return all_intervals_combined_data

def delete_fits_and_xml_files(source_name_cleaned, time_interval_name):
    # Define the folders and file types to delete
    paths_to_delete = [
        f'./data/{source_name_cleaned}/snr/expmap/*.fits',
        f'./data/{source_name_cleaned}/snr/srcmap/*.fits',
        f'./data/{source_name_cleaned}/snr/ccube/*.fits',
        f'./data/{source_name_cleaned}/snr/models/*.xml',
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

# Main function to run the analysis
def run_analysis(source_name, short_name, num_workers, num_time_intervals, time_interval_name, start_month, ra, dec, minimal_energy, maximal_energy, number_of_bins, bins_def_filename):
    num_time_intervals = 198
    source_name_cleaned = source_name.replace(" ", "").replace(".", "dot").replace("+", "plus").replace("-", "minus")
    
    snr_arg = source_name, time_interval_name, ra, dec, minimal_energy, maximal_energy
    snr_filtering(snr_arg)
    print('SNR filtering done!')

    ('Begin Likelihood')
    gtbindef_energy_command = [
        'gtbindef', 
        'E',
        f'{bins_def_filename}.txt',
        f'./energy_bins_def/{number_of_bins}/energy_bins_gtbindef.fits' ,
        'MeV']
    
    subprocess.run(gtbindef_energy_command, check=True)
    energy_bins = get_energy_bins(bins_def_filename)
    snr_args = (source_name, time_interval_name, ra, dec)
    snr_filtering_per_bin(snr_args, energy_bins)
    print('SNR filtering per bin done!')
     
    source_name_cleaned = source_name.replace(" ", "").replace(".", "dot").replace("+", "plus").replace("-", "minus")
    
    running_args_ltcube = [(i, source_name, time_interval_name, ra, dec, short_name) for i in range(start_month, num_time_intervals)]
    running_args = []
    running_args_xml_free_alpha = []
    running_args_xml_free_beta = []
    running_args_per_bin = []
    
    for i in range(198): #get the number of timeintervals from somewhere
        running_args.append((i, source_name, time_interval_name, ra, dec, minimal_energy, maximal_energy, number_of_bins))
        running_args_xml_free_alpha.append((i, source_name, source_name_cleaned, time_interval_name, "alpha"))
        running_args_xml_free_beta.append((i, source_name, source_name_cleaned, time_interval_name, "beta"))

        for energy_bin_index, (emin, emax) in enumerate(energy_bins):
            running_args_per_bin.append((i, source_name, time_interval_name, ra, dec, short_name, emin, emax, energy_bin_index, number_of_bins))
    
    with Pool(num_workers) as p:
        list(tqdm(p.map(generate_ltcube, running_args_ltcube), total=len(running_args_ltcube)))
        list(tqdm(p.map(generate_files, running_args), total=len(running_args)))
        list(tqdm(p.map(call_xml_modifier, running_args_xml_free_alpha), total=len(running_args_xml_free_alpha)))
        list(tqdm(p.map(source_maps_free_alpha, running_args), total=len(running_args)))
        list(tqdm(p.map(run_binned_likelihood_free_alpha, running_args), total=len(running_args)))
        list(tqdm(p.map(call_xml_modifier, running_args_xml_free_beta), total=len(running_args_xml_free_beta)))
        list(tqdm(p.map(source_maps_free_beta, running_args), total=len(running_args)))
        list(tqdm(p.map(run_binned_likelihood_free_beta, running_args), total=len(running_args)))

    save_flux_fit_data(source_name_cleaned, time_interval_name, num_time_intervals)
    print("Flux fit saved!")
    #fuction that deletes everything generated in generate_files and source_maps
    
    delete_fits_and_xml_files(source_name_cleaned, time_interval_name)

    
    with Pool(num_workers) as p:
        list(tqdm(p.map(generate_files_per_bin, running_args_per_bin), total=len(running_args_per_bin)))
        list(tqdm(p.map(source_maps_per_bin, running_args_per_bin), total=len(running_args_per_bin)))
        list(tqdm(p.map(run_binned_likelihood_per_bin, running_args_per_bin), total=len(running_args_per_bin)))
    
    combine_flux_data_per_time_interval(source_name_cleaned, time_interval_name, num_time_intervals, number_of_bins)
    print("Spectral points per time interval saved!")
    delete_fits_and_xml_files(source_name_cleaned, time_interval_name)
    '''