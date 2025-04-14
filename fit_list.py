import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import shlex
from astropy.io import fits as pyfits
from matplotlib.backends.backend_pdf import PdfPages

# === Define log-parabola function ===
def logpar_base(x, Norm, alpha_, beta_):
    E_b = 1000  # Fixed reference energy [MeV]
    return Norm*1e-8 * (x / E_b) ** (-(alpha_ + beta_ * np.log(x / E_b)))

# === Fitting function with systematics and masking ===
def fit_logpar(x, y, y_err, nobs, lowerb):
    # Mask strategy
    if nobs is not None:
        mask = nobs > 10
    elif lowerb is not None:
        mask = y > 1e-30
    elif nobs == None and lowerb == None:
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
    #p0_base = [1e-11, 2.0, 0.001]
    
    p0_base = [10, 1, 0.1]
    assert np.all(np.isfinite(x_filtered)), "x_filtered has non-finite values!"
    assert np.all(np.isfinite(y_filtered)), "y_filtered has non-finite values!"
    assert np.all(np.isfinite(y_err_filtered)), "y_err_filtered has non-finite values!"
    popt, pcov = curve_fit(logpar_base, x_filtered, y_filtered,
                           sigma=y_err_eff, absolute_sigma=True, p0=p0_base)

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
    flux_band = data['Flux_Band'][ok][0]
    unc_flux_band = data['Unc_Flux_Band'][ok][0]
    alpha = data['PL_Index'][ok][0]

    # Compute relative errors safely
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio0 = np.where(flux_band > 0, unc_flux_band[:, 0] / flux_band, np.inf)
        ratio1 = np.where(flux_band > 0, unc_flux_band[:, 1] / flux_band, np.inf)

    dfl1 = -fl * ratio0
    dfl2 = fl * ratio1
    dfl = np.maximum(dfl1, dfl2)

    # Replace non-finite errors with a large default (e.g., 100% error)
    default_large_err = 1.0 * fl  # 100% uncertainty
    dfl = np.where(np.isfinite(dfl), dfl, default_large_err)

    # Keep only positive flux values (do not mask on error anymore)
    ok = fl > 0

    return eav[ok], fl[ok], dfl[ok], [de1[ok], de2[ok]], alpha
# === Compute reduced chi-squared ===
def compute_chi2(x, y, y_err, model, popt):
    model_vals = model(x, *popt)
    chi2 = np.sum(((y - model_vals) / y_err) ** 2)
    dof = len(y) - len(popt)
    return chi2 / dof if dof > 0 else np.nan

# === Output setup ===
output_lines = ["Source_Name\tChi2_Data\tChi2_Catalog\tAlpha_Catalog\tAlpha_DataFit\tAlpha_CatalogFit\n"]

pdf = PdfPages("source_spectra_fits_lowerb.pdf")

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
            popt_data, _, x_filt_data, y_filt_data, yerr_eff_data = fit_logpar(x, y/x**2, y_err/x**2, nobs=None, lowerb=True)
            chi2_data = compute_chi2(x_filt_data, y_filt_data, yerr_eff_data, logpar_base, popt_data)
        except Exception as e:
            print(f"Data fit failed for {source_name}: {e}")

        try:
            eav0, f0, df0, de0, alpha = GetCatalogueSpectrum(source_name)
            popt_cat, _, x_filt_cat, y_filt_cat, yerr_eff_cat = fit_logpar(eav0[1:], f0[1:]/eav0[1:]**2, df0[1:]/eav0[1:]**2, nobs=None, lowerb=True)
            chi2_cat = compute_chi2(x_filt_cat, y_filt_cat, yerr_eff_cat, logpar_base, popt_cat)
        except Exception as e:
            print(f"Catalogue fit failed for {source_name}: {e}")

        alpha_data = popt_data[1] if 'popt_data' in locals() else np.nan
        alpha_catfit = popt_cat[1] if 'popt_cat' in locals() else np.nan
        alpha_catalog = alpha if 'alpha' in locals() else np.nan

        output_lines.append(
            f"{source_name}\t{chi2_data:.3f}\t{chi2_cat:.3f}\t{alpha_catalog:.3f}\t{alpha_data:.3f}\t{alpha_catfit:.3f}\n"
        )

        # === Plotting ===
        fig, (ax_top, ax_bot) = plt.subplots(nrows=2, ncols=1, figsize=(6, 7), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)

        # === TOP: SPECTRUM PLOT ===

        # Data
        ax_top.errorbar(x_filt_data, y_filt_data, yerr=yerr_eff_data, fmt='o', color='blue', label='Rikke Data')
        x_model_data = np.logspace(np.log10(x_filt_data.min()*0.8), np.log10(x_filt_data.max()*1.2), 300)
        ax_top.plot(x_model_data, logpar_base(x_model_data, *popt_data), color='blue', linestyle='-', label=f'Data Fit ($\\chi^2_\\nu$={chi2_data:.2f})')

        # Catalogue
        ax_top.errorbar(x_filt_cat, y_filt_cat, yerr=yerr_eff_cat, fmt='s', color='green', label='4FGL Catalog')
        x_model_cat = np.logspace(np.log10(x_filt_cat.min()*0.8), np.log10(x_filt_cat.max()*1.2), 300)
        ax_top.plot(x_model_cat, logpar_base(x_model_cat, *popt_cat), color='green', linestyle='--', label=f'Catalog Fit ($\\chi^2_\\nu$={chi2_cat:.2f})')

        ax_top.set_xscale("log")
        ax_top.set_yscale("log")
        ax_top.set_ylabel("E$^2$ Flux [erg cm$^{-2}$ s$^{-1}$]")
        ax_top.set_title(source_name)
        ax_top.legend()
        ax_top.grid(True, which='both', ls=':')

        # === BOTTOM: RESIDUALS ===

        # Compute residuals = (data - model) / error
        resid_data = (y_filt_data - logpar_base(x_filt_data, *popt_data)) / yerr_eff_data
        resid_cat = (y_filt_cat - logpar_base(x_filt_cat, *popt_cat)) / yerr_eff_cat

        ax_bot.axhline(0, color='gray', linestyle='--', linewidth=1)
        ax_bot.errorbar(x_filt_data, resid_data, yerr=1, fmt='o', color='blue', label='Data Residuals')
        ax_bot.errorbar(x_filt_cat, resid_cat, yerr=1, fmt='s', color='green', label='Catalog Residuals')

        ax_bot.set_xscale("log")
        ax_bot.set_xlabel("Energy [MeV]")
        ax_bot.set_ylabel("Residuals")
        ax_bot.grid(True, which='both', ls=':')
        ax_bot.legend()

        # Tight layout for spacing


        pdf.savefig(fig)
        plt.close(fig)

# === Save output ===
with open("chi2_summary_lowerb.txt", "w") as out_file:
    out_file.writelines(output_lines)

pdf.close()
