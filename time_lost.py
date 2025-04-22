import os
from astropy.io import fits
def compute_time_units(seconds):
    """
    Convert time in seconds to years, months (approx.), and days.
    """
    minutes = seconds / 60
    hours = minutes / 60
    days = hours / 24
    years = days / 365.25
    months = years * 12
    return years, months, days
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
gti_info = []  # tuples of (label, path)
for method in methods:
    if method == 'NONE':
        label = 'NONE'
        path = os.path.join(general_path_for_slurm, 'data', source_name_cleaned, 'gti.fits')
        gti_info.append((label, path))
    elif method == 'SNR':
        for snr in snrratios:
            label = f'SNR_{snr}'
            path = os.path.join(
                general_path_for_slurm, 'data', source_name_cleaned,
                method, f'gti_noflares_snr{snr}.fits'
            )
            gti_info.append((label, path))
    elif method == 'LIN':
        for interval in time_intervals:
            label = f'LIN_{interval}'
            path = os.path.join(
                general_path_for_slurm, 'data', source_name_cleaned,
                method, f'gti_noflares_{interval}.fits'
            )
            gti_info.append((label, path))

# Read GTI durations
results = {}
for label, path in gti_info:
    if not os.path.isfile(path):
        print(f"[MISSING] {label}: {path}")
        continue
    with fits.open(path) as hdul:
        gti = hdul['GTI'].data
        durations = gti['STOP'] - gti['START']
        total = durations.sum()
        results[label] = total

# Baseline NONE total
total_none = results.get('NONE')
if total_none is None:
    raise RuntimeError("Baseline NONE GTI not found or could not be read.")

# Compare and report
print(f"Baseline (NONE) total duration: {total_none:.1f} s")
years_none, months_none, days_none = compute_time_units(total_none)
print(f"  → {years_none:.2f} years ({months_none:.1f} months; {days_none:.1f} days)")

for label, total in results.items():
    if label == 'NONE':
        continue
    lost = total_none - total
    frac = lost / total_none * 100
    yrs, mons, dys = compute_time_units(lost)
    print(f"\n{label} total duration: {total:.1f} s")
    print(f"  Lost time: {lost:.1f} s  ({frac:.2f}% of NONE)")
    print(f"    → {yrs:.2f} years ({mons:.1f} months; {dys:.1f} days)")
