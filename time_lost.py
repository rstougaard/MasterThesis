import os
import numpy as np
from astropy.io import fits

def secs_to_human(seconds):
    mins = seconds / 60
    hrs = mins / 60
    days = hrs / 24
    yrs = days / 365.25
    return yrs, days, hrs

# CONFIG
general_path_for_slurm = "/groups/pheno/sqd515/MasterThesis"
methods = ['LIN', 'SNR']
snrratios = ['snr10', 'snr5', 'snr3']
time_intervals = ['week', 'month']
source_name = '4FGL J0319.8+4130'
source_clean = (source_name.replace(' ', '')
                        .replace('.', 'dot')
                        .replace('+', 'plus')
                        .replace('-', 'minus')
                        .replace('"', ''))

# Build loopitems per method
def get_loopitems(method):
    if method == 'SNR':
        return [str(s) for s in snrratios]
    elif method == 'LIN':
        return time_intervals
    return []

# Process each method/loopitem pair
for method in methods:
    for item in get_loopitems(method):
        # paths
        lc_file = os.path.join(
            general_path_for_slurm, 'data', source_clean,
            method, f'lc_{item}.fits'
        )
        flare_txt = os.path.join(
            general_path_for_slurm, 'data', source_clean,
            method, f'flare_interval_{item}.txt'
        )

        print(f"\n--- {method} {item} ---")
        # Check existence
        if not os.path.isfile(lc_file):
            print(f"LC file missing: {lc_file}")
            continue
        if not os.path.isfile(flare_txt):
            print(f"Flare-interval file missing: {flare_txt}")
            continue

        # Sum total LC exposure
        with fits.open(lc_file) as fb:
            data = fb[1].data
            total_lc = np.sum(data['TIMEDEL'])

        # Sum total flare intervals
        intervals = np.loadtxt(flare_txt)
        # ensure 2D
        if intervals.ndim == 1:
            intervals = intervals.reshape(1, -1)
        durations = intervals[:,1] - intervals[:,0]
        total_flare = np.sum(durations)

        # Compute net
        net = total_lc - total_flare
        frac_lost = total_flare / total_lc * 100 if total_lc>0 else np.nan

        # Print summary
        yrs_lc, days_lc, hrs_lc = secs_to_human(total_lc)
        yrs_fl, days_fl, hrs_fl = secs_to_human(total_flare)
        yrs_net, days_net, hrs_net = secs_to_human(net)

        print(f"Total LC time      : {total_lc:.1f} s ({days_lc:.2f} d)")
        print(f"Total flare time   : {total_flare:.1f} s ({days_fl:.2f} d) = {frac_lost:.2f}% lost")
        print(f"Net good time      : {net:.1f} s ({days_net:.2f} d)\n")