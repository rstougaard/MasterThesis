import os
from astropy.io import fits

# Configuration
general_path_for_slurm = "/groups/pheno/sqd515/MasterThesis"
methods = ['NONE', 'LIN', 'SNR']
snrratios = [10, 5, 3]
time_intervals = ['week', 'month']
source_name = '4FGL J0319.8+4130'
source_name_cleaned = (
    source_name.replace(' ', '')
    .replace('.', 'dot')
    .replace('+', 'plus')
    .replace('-', 'minus')
    .replace('"', '')  # Ensure no extra quotes remain
)

# Helper to inspect FITS content

def inspect_fits(path):
    if not os.path.isfile(path):
        print(f"File not found: {path}")
        return
    try:
        with fits.open(path) as hdulist:
            print(f"\nContents of {path}:")
            hdulist.info()
    except Exception as e:
        print(f"Error reading {path}: {e}")

# Loop through methods
for method in methods:
    if method == 'NONE':
        # Only one GTI file for NONE
        gti = os.path.join(
            general_path_for_slurm,
            'data',
            source_name_cleaned,
            'gti.fits'
        )
        inspect_fits(gti)

    elif method == 'SNR':
        # Loop over SNR ratios for SNR method
        for snrratio in snrratios:
            loop_item = str(snrratio)
            gti_noflares = os.path.join(
                general_path_for_slurm,
                'data',
                source_name_cleaned,
                method,
                f'gti_noflares_snr{loop_item}.fits'
            )
            inspect_fits(gti_noflares)

    elif method == 'LIN':
        # Loop over time intervals for LIN method
        for interval in time_intervals:
            loop_item = interval
            gti_noflares = os.path.join(
                general_path_for_slurm,
                'data',
                source_name_cleaned,
                method,
                f'gti_noflares_{loop_item}.fits'
            )
            inspect_fits(gti_noflares)
