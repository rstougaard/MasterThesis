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

#─ BUILD LIST OF GTI FILES ───────────────────────────────────────────
gti_files = []
for method in methods:
    if method == 'NONE':
        gti_files.append(
            os.path.join(general_path_for_slurm,
                         'data', source_name_cleaned,
                         'gti.fits')
        )
    elif method == 'SNR':
        for snr in snrratios:
            gti_files.append(
                os.path.join(general_path_for_slurm,
                             'data', source_name_cleaned,
                             method,
                             f'gti_noflares_snr{snr}.fits')
            )
    elif method == 'LIN':
        for interval in time_intervals:
            gti_files.append(
                os.path.join(general_path_for_slurm,
                             'data', source_name_cleaned,
                             method,
                             f'gti_noflares_{interval}.fits')
            )

#─ PROCESS EACH FILE ─────────────────────────────────────────────────
for path in gti_files:
    if not os.path.isfile(path):
        print(f"[MISSING] {path}")
        continue

    with fits.open(path) as hdul:
        data = hdul['GTI'].data
        starts = data['START']   # array of doubles
        stops  = data['STOP']    # array of doubles

        # durations of each good-time interval
        durations = stops - starts

        # total exposure time
        total = durations.sum()
        print(f"→ {os.path.basename(path)} total duration = {total:.3f}\n")
