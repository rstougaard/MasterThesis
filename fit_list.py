import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import shlex
from astropy.io import fits as pyfits
from matplotlib.backends.backend_pdf import PdfPages

# === Define log-parabola function ===
def logpar_base(x, Norm, alpha_, beta_):
    E_b = 1000  # Fixed reference energy [MeV]
    return Norm * (x / E_b) ** (-(alpha_ + beta_ * np.log(x / E_b)))

# === Fitting function with systematics and masking ===
def fit_logpar(x, y, y_err, nobs, lowerb):
    # Mask strategy
    if nobs is not None:
        mask = nobs > 10
    elif lowerb is not None:
        mask = y > 1e-13
    else:
        mask = y != 0
    x_filtered = x[mask]
    y_filtered = y[mask]
    y_err_filtered = y_err[mask]

    # Add systematic errors
    if not mask[-1]:
        y_err_eff = y_err_filtered + 0.03 * y_filtered
    else:
        y_err_eff0 = y_err_filtered[:-1] + 0.03 * y_filtered[:-1]
        y_err_eff1 = y_err_filtered[-1] + 0.10 * y_filtered[-1]
        y_err_eff = np.append(y_err_eff0, y_err_eff1)

    # Fit with bounds
    bounds_base = ([1e-14, -5.0, -5.0], [1e-9, 5.0, 3.0])
    p0_base = [1e-11, 2.0, 0.001]
    assert np.all(np.isfinite(x_filtered)), "x_filtered has non-finite values!"
    assert np.all(np.isfinite(y_filtered)), "y_filtered has non-finite values!"
    assert np.all(np.isfinite(y_err_filtered)), "y_err_filtered has non-finite values!"
    popt, pcov = curve_fit(logpar_base, x_filtered, y_filtered,
                           sigma=y_err_eff, absolute_sigma=True,
                           bounds=bounds_base, p0=p0_base)

    return popt, pcov, x_filtered, y_filtered, y_err_eff

# === Get catalogue spectrum ===
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
    dfl = np.maximum(dfl1, dfl2)

    ok = fl > 0
    return eav[ok], fl[ok], dfl[ok], [de1[ok], de2[ok]]

# === Compute reduced chi-squared ===
def compute_chi2(x, y, y_err, model, popt):
    model_vals = model(x, *popt)
    chi2 = np.sum(((y - model_vals) / y_err) ** 2)
    dof = len(y) - len(popt)
    return chi2 / dof if dof > 0 else np.nan

# === Output setup ===
output_lines = ["Source_Name\tChi2_Data\tChi2_Catalog\n"]
pdf = PdfPages("source_spectra_fits.pdf")

# === Main loop ===
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

        # Load source fit result
        try:
            f_bin = pyfits.open(f'./fit_results/{source_name_cleaned}_fit_data_NONE.fits')
            bin_data = f_bin[1].data
            sorted_indices = np.argsort(bin_data['emin'])
            sorted_data_none = bin_data[sorted_indices]

            x = sorted_data_none['geometric_mean']
            y = sorted_data_none['flux_tot_value']
            y_err = sorted_data_none['flux_tot_error']
            nobs = sorted_data_none['nobs']
        except Exception as e:
            print(f"Could not read SNR data for {source_name}: {e}")
            continue

        chi2_data = np.nan
        chi2_cat = np.nan

        try:
            popt_data, _, x_filt_data, y_filt_data, yerr_eff_data = fit_logpar(x, y, y_err, nobs=nobs, lowerb=None)
            chi2_data = compute_chi2(x_filt_data, y_filt_data, yerr_eff_data, logpar_base, popt_data)
        except Exception as e:
            print(f"Data fit failed for {source_name}: {e}")

        try:
            eav0, f0, df0, de0 = GetCatalogueSpectrum(source_name)
            popt_cat, _, x_filt_cat, y_filt_cat, yerr_eff_cat = fit_logpar(eav0[1:], f0[1:], df0[1:], nobs=None, lowerb=True)
            chi2_cat = compute_chi2(x_filt_cat, y_filt_cat, yerr_eff_cat, logpar_base, popt_cat)
        except Exception as e:
            print(f"Catalogue fit failed for {source_name}: {e}")

        # Save summary line
        output_lines.append(f"{source_name}\t{chi2_data:.3f}\t{chi2_cat:.3f}\n")

        # === Plotting ===
        fig, ax = plt.subplots(figsize=(6, 5))

        # Plot data
        if not np.isnan(chi2_data):
            ax.errorbar(x_filt_data, y_filt_data, yerr=yerr_eff_data, fmt='o', color='blue', label='Rikke Data')
            x_model_data = np.logspace(np.log10(x_filt_data.min()*0.8), np.log10(x_filt_data.max()*1.2), 300)
            ax.plot(x_model_data, logpar_base(x_model_data, *popt_data), color='blue', linestyle='-', label=f'Data Fit ($\\chi^2_\\nu$={chi2_data:.2f})')

        # Plot catalog
        if not np.isnan(chi2_cat):
            ax.errorbar(x_filt_cat, y_filt_cat, yerr=yerr_eff_cat, fmt='s', color='green', label='4FGL Catalog')
            x_model_cat = np.logspace(np.log10(x_filt_cat.min()*0.8), np.log10(x_filt_cat.max()*1.2), 300)
            ax.plot(x_model_cat, logpar_base(x_model_cat, *popt_cat), color='green', linestyle='--', label=f'Catalog Fit ($\\chi^2_\\nu$={chi2_cat:.2f})')

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Energy [MeV]")
        ax.set_ylabel("Flux [erg cm$^{-2}$ s$^{-1}$]")
        ax.set_title(source_name)
        ax.legend()
        ax.grid(True, which='both', ls=':')

        pdf.savefig(fig)
        plt.close(fig)

# === Save output ===
with open("chi2_summary.txt", "w") as out_file:
    out_file.writelines(output_lines)

pdf.close()
