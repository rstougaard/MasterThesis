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
    'snr10': ('SNR', 'snr10')
}

# Read sources & build DataFrame
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
        for col, (method, item) in methods.items():
            lc_file   = os.path.join(general_path_for_slurm, source_clean, method, f'lc_{item}.fits')
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

df = pd.DataFrame(results)
df = df[['source', 'week', 'month', 'snr3', 'snr5', 'snr10']]

# Compute averages for each numeric column
avg = df[['week', 'month', 'snr3', 'snr5', 'snr10']].mean(skipna=True)
# Build a one-row DataFrame for the averages
avg_row = {'source': 'Average', **avg.to_dict()}
avg_df = pd.DataFrame([avg_row])

# Concatenate the average row to the bottom
df = pd.concat([df, avg_df], ignore_index=True)

# Generate LaTeX table
latex_table = df.to_latex(
    index=False,
    float_format="%.2f",
    caption="Flare Time Lost Percentage",
    label="tab:flare_loss"
)

print(latex_table)

outpath = 'flare_loss_table.tex'
with open(outpath, 'w') as f:
    f.write(latex_table)

print(f"Saved LaTeX table to {outpath}")
