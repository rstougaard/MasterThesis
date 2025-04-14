import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import shlex
from astropy.io import fits as pyfits

# Define log-parabola function
def logpar_base(x, Norm, alpha_, beta_):
    E_b = 1000  # Fixed reference energy [MeV]
    return Norm * (x / E_b) ** (-(alpha_ + beta_ * np.log(x / E_b)))

# Fit function with systematic error handling
def fit_logpar(x, y, y_err, nobs, lowerb):
    # Mask out zeroes
    if nobs is not None:
        mask = nobs > 10
    elif lowerb is not None:
        mask = y >1e-13
    else:
        mask = y != 0
    x_filtered = x[mask]
    y_filtered = y[mask]
    y_err_filtered = y_err[mask]

    # Add systematic uncertainties
    if not mask[-1]:
        y_err_eff = y_err_filtered + 0.03 * y_filtered
    else:
        y_err_eff0 = y_err_filtered[:-1] + 0.03 * y_filtered[:-1]
        y_err_eff1 = y_err_filtered[-1] + 0.10 * y_filtered[-1]
        y_err_eff = np.append(y_err_eff0, y_err_eff1)

    # Fit
    bounds_base = ([1e-14, -5.0, -5.0], [1e-9, 5.0, 3.0])
    p0_base = [1e-11, 2.0, 0.001]

    popt, pcov = curve_fit(logpar_base, x_filtered, y_filtered,
                           sigma=y_err_eff, absolute_sigma=True, p0=p0_base)

    return popt, pcov, x_filtered, y_filtered, y_err_eff


# === Load catalogue spectrum ===
def GetCatalogueSpectrum(nn):
    with pyfits.open('test/gll_psc_v35.fit') as f:
        data = f[1].data
        ebounds = f[5].data
        emin = np.unique(ebounds['LowerEnergy'])
        emax = np.unique(ebounds['UpperEnergy'])

    eav = (emin * emax)**0.5
    de1 = eav - emin
    de2 = emax - eav
    names4fgl = data['Source_Name']

    ok = np.where(names4fgl == nn)
    fl = data['nuFnu_Band'][ok][0]  # erg/cm2/s

    ratio0 = data['Unc_Flux_Band'][ok][0][:, 0] / data['Flux_Band'][ok][0]
    ratio1 = data['Unc_Flux_Band'][ok][0][:, 1] / data['Flux_Band'][ok][0]

    dfl1 = -fl * ratio0
    dfl2 = fl * ratio1
    systematics = 0.03  # set your systematic error level here
    dfl = np.maximum(dfl1, dfl2)

    ok = fl > 0
    return eav[ok], fl[ok], dfl[ok], [de1[ok], de2[ok]]

output_lines = ["Source_Name\tChi2_Data\tChi2_Catalog\n"]

def compute_chi2(x, y, y_err, model, popt):
    model_vals = model(x, *popt)
    chi2 = np.sum(((y - model_vals) / y_err) ** 2)
    dof = len(y) - len(popt)
    return chi2 / dof if dof > 0 else np.nan

# === Loop over all sources ===
with open('Source_ra_dec_specin.txt', 'r') as file:
    for line in file:
        parts = shlex.split(line.strip())
        source_name = parts[0]
        ra = float(parts[1])
        dec = float(parts[2])
        specin = float(parts[3])

        source_name_cleaned = (source_name.replace(" ", "")
                                             .replace(".", "dot")
                                             .replace("+", "plus")
                                             .replace("-", "minus")
                                             .replace('"', ''))

        f_bin = pyfits.open(f'./fit_results/{source_name_cleaned}_fit_data_NONE.fits')
        bin_data = f_bin[1].data
        sorted_indices = np.argsort(bin_data['emin'])
        sorted_data_none = bin_data[sorted_indices]

        x = sorted_data_none['geometric_mean']
        y = sorted_data_none['flux_tot_value']
        y_err = sorted_data_none['flux_tot_error']
        nobs = sorted_data_none['nobs']

        chi2_data = np.nan
        chi2_cat = np.nan

        try:
            popt_data, _, x_filt_data, y_filt_data, yerr_eff_data = fit_logpar(x, y, y_err, nobs=nobs, lowerb=None)
            chi2_data = compute_chi2(x_filt_data, y_filt_data, yerr_eff_data, logpar_base, popt_data)
        except Exception as e:
            print(f"Data fit failed for {source_name}: {e}")

        try:
            eav0, f0, df0, de0 = GetCatalogueSpectrum(source_name)
            popt_cat, _, x_filt_cat, y_filt_cat, yerr_eff_cat = fit_logpar(eav0[1:], f0[1:], df0[1:], nobs=None, lowerb=None)
            chi2_cat = compute_chi2(x_filt_cat, y_filt_cat, yerr_eff_cat, logpar_base, popt_cat)
        except Exception as e:
            print(f"Catalogue fit failed for {source_name}: {e}")

        output_lines.append(f"{source_name}\t{chi2_data:.3f}\t{chi2_cat:.3f}\n")

# === Write results to txt file ===
with open("chi2_summary.txt", "w") as out_file:
    out_file.writelines(output_lines)
