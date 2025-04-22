import os
import numpy as np
from astropy.io import fits
from datetime import datetime, timezone, timedelta

# ──────────────────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────────────────

def secs_to_human(seconds):
    """
    Convert time in seconds to (years, days, hours).
    """
    mins = seconds / 60
    hrs  = mins / 60
    days = hrs / 24
    yrs  = days / 365.25
    return yrs, days, hrs

# MET conversion: Fermi MET epoch is 2001-01-01T00:00:00 UTC
MET_EPOCH = datetime(2001, 1, 1, 0, 0, 0, tzinfo=timezone.utc)

def met_to_datetime(met_seconds):
    """
    Convert Mission Elapsed Time (MET, seconds since 2001-01-01) to UTC datetime.
    """
    return MET_EPOCH + timedelta(seconds=met_seconds)

# ──────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────
general_path_for_slurm = "/groups/pheno/sqd515/MasterThesis"
methods        = ['LIN', 'SNR']
snrratios      = ['snr10', 'snr5', 'snr3']
time_intervals = ['week', 'month']
source_name    = '4FGL J0319.8+4130'
source_clean   = (
    source_name.replace(' ', '')
               .replace('.', 'dot')
               .replace('+', 'plus')
               .replace('-', 'minus')
               .replace('"', '')
)

# Determine loop items per method

def get_loopitems(method):
    if method == 'SNR':
        return [str(s) for s in snrratios]
    elif method == 'LIN':
        return time_intervals
    return []

# ──────────────────────────────────────────────────────────────────────────
# PROCESS EACH LC AND FLARE INTERVAL
# ──────────────────────────────────────────────────────────────────────────
for method in methods:
    for item in get_loopitems(method):
        lc_file   = os.path.join(
            general_path_for_slurm, 'data', source_clean,
            method, f'lc_{item}.fit'
        )
        flare_txt = os.path.join(
            general_path_for_slurm, 'data', source_clean,
            method, f'flare_interval_{item}.txt'
        )

        print(f"\n--- {method} {item} ---")
        if not os.path.isfile(lc_file):
            print(f"LC file missing: {lc_file}")
            continue
        if not os.path.isfile(flare_txt):
            print(f"Flare-interval file missing: {flare_txt}")
            continue

        # Total LC exposure (sum of TIMEDEL)
        with fits.open(lc_file) as fb:
            data     = fb[1].data
            total_lc = np.sum(data['TIMEDEL'])
            # Example: convert first and last TIME to UTC dates
            start_met = data['TIME'][0]
            end_met   = data['TIME'][-1]

        # Total flare time (difference of start/stop in txt)
        intervals = np.loadtxt(flare_txt)
        if intervals.ndim == 1:
            intervals = intervals.reshape(1, -1)
        durations   = intervals[:,1] - intervals[:,0]
        total_flare = np.sum(durations)

        # Net good time
        net_lc    = total_lc - total_flare
        frac_lost = (total_flare / total_lc * 100) if total_lc > 0 else np.nan

        # Human-readable
        yrs_lc, days_lc, hrs_lc    = secs_to_human(total_lc)
        yrs_fl, days_fl, hrs_fl    = secs_to_human(total_flare)
        yrs_net, days_net, hrs_net = secs_to_human(net_lc)

        # Output
        print(f"Total LC time    : {total_lc:.1f} s ({days_lc:.2f} d -> {yrs_lc:.2f} y)")
        print(f"Flare time lost  : {total_flare:.1f} s ({days_fl:.2f} d -> {yrs_fl:.2f} y) = {frac_lost:.2f}%")
        print(f"Net good time    : {net_lc:.1f} s ({days_net:.2f} d -> {yrs_net:.2f} y)")

        # MET -> UTC examples
        print(f"First LC bin MET: {start_met:.1f} -> UTC {met_to_datetime(start_met)}")
        print(f"Last  LC bin MET: {end_met:.1f} -> UTC {met_to_datetime(end_met)}")

# ──────────────────────────────────────────────────────────────────────────
# INDIVIDUAL MET CONVERSION EXAMPLE
# ──────────────────────────────────────────────────────────────────────────
# Convert a standalone MET value to UTC:
#   met_time = 239557417  # seconds
#   print(met_to_datetime(met_time))
