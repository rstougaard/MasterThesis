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

# MET conversion (if you still need it elsewhere)
MET_EPOCH = datetime(2001, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
def met_to_datetime(met_seconds):
    return MET_EPOCH + timedelta(seconds=met_seconds)

# ──────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────

general_path_for_slurm = "/groups/pheno/sqd515/MasterThesis"
methods        = ['LIN'] #, 'SNR']
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
        lc_file = os.path.join(
            general_path_for_slurm, 'data', source_clean,
            method, f'lc_{item}.fits'
        )
        flare_txt = os.path.join(
            general_path_for_slurm, 'data', source_clean,
            method, f'flare_intervals_{item}.txt'
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

        # Total flare time (difference of start/stop in txt)
        intervals = np.loadtxt(flare_txt)
        if intervals.ndim == 1:
            intervals = intervals.reshape(1, -1)
        durations   = intervals[:,1] - intervals[:,0]
        total_flare = np.sum(durations)

        # Net good time
        net_lc    = total_lc - total_flare

        # Convert to years only
        yrs_flare, _, _ = secs_to_human(total_flare)
        yrs_net,   _, _ = secs_to_human(net_lc)

        # Percentage of time lost to flares
        frac_lost = (total_flare / total_lc * 100) if total_lc > 0 else np.nan

        # Simplified output
        print(f"Non‑flare duration: {yrs_net:.2f} years")
        print(f"Flare duration    : {yrs_flare:.2f} years")
        print(f"Flare fraction    : {frac_lost:.2f}%")
'''
import os
import numpy as np
import pandas as pd
import shlex
from astropy.io import fits

# CONFIG
general_path_for_slurm = "/groups/pheno/sqd515/MasterThesis/data"
methods = {
    'week':  ('LIN', 'week'),
    'month': ('LIN', 'month'),
    'snr3':  ('SNR', 'snr3'),
    'snr5':  ('SNR', 'snr5'),
    'snr10': ('SNR', 'snr10'),
}

# Read sources
results = []
with open('Source_ra_dec_specin.txt', 'r') as file:
    for line in file:
        parts = shlex.split(line.strip())
        source_name = parts[0]
        source_clean = (
            source_name.replace(' ', '')
                       .replace('.', 'dot')
                       .replace('+', 'plus')
                       .replace('-', 'minus')
                       .replace('"', '')
        )
        row = {'source': source_name}
        
        # Calculate fraction lost for each interval
        for col, (method, item) in methods.items():
            lc_file = os.path.join(general_path_for_slurm, source_clean, method, f'lc_{item}.fits')
            flare_txt = os.path.join(general_path_for_slurm, source_clean, method, f'flare_intervals_{item}.txt')
            
            if os.path.isfile(lc_file) and os.path.isfile(flare_txt):
                with fits.open(lc_file) as fb:
                    total_lc = np.sum(fb[1].data['TIMEDEL'])
                intervals = np.loadtxt(flare_txt)
                if intervals.ndim == 1:
                    intervals = intervals.reshape(1, -1)
                total_flare = np.sum(intervals[:,1] - intervals[:,0])
                frac_lost = total_flare / total_lc * 100 if total_lc > 0 else np.nan
            else:
                frac_lost = np.nan
            
            row[col] = frac_lost
        
        results.append(row)

# Create DataFrame
df = pd.DataFrame(results)
df = df[['source', 'week', 'month', 'snr3', 'snr5', 'snr10']]

# Generate LaTeX table
latex_table = df.to_latex(index=False, float_format="%.2f", caption="Flare Time Lost Percentage", label="tab:flare_loss")

print(latex_table)
outpath = 'flare_loss_table.tex'
with open(outpath, 'w') as f:
    f.write(latex_table)

print(f"Saved LaTeX table to {outpath}")

'''