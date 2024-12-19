import gt_apps as my_apps
from astropy.io import fits
import numpy

# Function to ensure paths exist
def check_paths(source_name, time_interval_name, number_of_bins):
    source_name_cleaned = source_name.replace(" ", "").replace(".", "dot").replace("+", "plus").replace("-", "minus")
    paths = [
        f'./data/{source_name_cleaned}/snr/',
    ]
    for path in paths:
        os.makedirs(path, exist_ok=True)
        print(f"Ensured existence of: {path}")

def snr_filtering(vars):
    ####### Livetime Cube #######
    source_name, time_interval_name = vars
    source_name_cleaned = source_name.replace(" ", "").replace(".", "dot").replace("+", "plus").replace("-", "minus")

    effective_area = 7000  # in cm^2
    average_photon_flux = 3.3525944e-07  # photon flux in ph / cm^2 / s
    time_years = 14
    months_in_14_years = 68  # given
    seconds_in_a_year = 365.25 * 24 * 3600  # accounting for leap years

    # Calculate the total number of photons over 14 years
    total_seconds = time_years * seconds_in_a_year
    total_num_photons = average_photon_flux * total_seconds
    # Recalculate the total number of photons considering the effective area
    total_num_photons_with_area = total_num_photons * effective_area

    # Recalculate the counts (per month) considering the effective area
    counts_with_area = total_num_photons_with_area / months_in_14_years

    print('Sorting event file by time...')
    with fits.open(f'./data/{source_name_cleaned}/filtered_gti.fits','update') as f:
        data = f[1].data
        order = numpy.argsort( data['TIME'] )

        for kk in data.names:
            data[kk] = data[kk][order]

        f[1].data = data

    print('done!')
    print()
    print('Creating LC')
    ### SNR
    my_apps.evtbin['evfile'] = f'./data/{source_name_cleaned}/filtered_gti.fits'
    my_apps.evtbin['outfile'] = f'./data/{source_name_cleaned}/snr/lc.fits'
    my_apps.evtbin['scfile'] = f'./data/{source_name_cleaned}/SC.fits'
    my_apps.evtbin['algorithm'] = 'LC'
    my_apps.evtbin['tbinalg'] = 'SNR'
    my_apps.evtbin['tstart'] = 239557417
    my_apps.evtbin['tstop'] = 435456000
    my_apps.evtbin['emin'] = 100
    my_apps.evtbin['emax'] = 1000000
    my_apps.evtbin['ebinalg'] = "NONE"
    my_apps.evtbin['ebinfile'] = "NONE"
    my_apps.evtbin['snratio'] = float(counts_with_area**0.5)
    my_apps.evtbin['lcemin'] = 100
    my_apps.evtbin['lcemax'] = 1000000
    my_apps.evtbin.run()

# Main function to run the analysis
def run_analysis(source_name, short_name, num_workers, num_time_intervals, time_interval_name, start_month, ra, dec, minimal_energy, maximal_energy, number_of_bins, bins_def_filename):
   
    source_name_cleaned = source_name.replace(" ", "").replace(".", "dot").replace("+", "plus").replace("-", "minus")
    
    snr_arg = source_name, time_interval_name
    snr_filtering(snr_arg)
    print('SNR filtering done!')