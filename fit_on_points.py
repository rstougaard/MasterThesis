from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import shlex
from iminuit.cost import LeastSquares
from iminuit import Minuit
from matplotlib.backends.backend_pdf import PdfPages
import pickle
plt.rcParams.update({
    'font.family': 'serif',
    'mathtext.fontset': 'cm',
    'font.size': 18,
    'axes.titlesize': 20,
    'axes.labelsize': 20,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'legend.fontsize': 16
})

def logpar_base(x, Norm, alpha_, beta_):
    E_b = 1000  # Fixed E_b value
    return Norm * (x / E_b) ** (-(alpha_ + beta_ * np.log(x / E_b)))

def find_best_worst_fits(fits):
    # Normalize into a flat list of result‑dicts
    if isinstance(fits, dict) and "fit_result" in fits:
        results = [fits]
    elif isinstance(fits, list):
        results = fits
    elif isinstance(fits, dict):
        results = list(fits.values())
    else:
        raise ValueError("find_best_worst_fits() got unsupported type")

    best, worst = None, None
    best_delta, worst_delta = np.inf, -np.inf

    for result in results:
        delta = result["fit_result"]["DeltaChi2"]
        if delta < best_delta:
            best_delta, best = delta, result
        if delta > worst_delta:
            worst_delta, worst = delta, result

    return {"best": best, "worst": worst}


def simple_plot_fit(dataset_none, fit_results_none, source, png_naming=""):
    fig = plt.figure(figsize=(10, 6))
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["mathtext.fontset"] = "cm"

    # ——— your existing FITS loading + errorbar code ———
    f = fits.open('./test/gll_psc_v35.fit')
    data = f[1].data; ebounds = f[5].data
    emin = np.unique(ebounds['LowerEnergy'])
    emax = np.unique(ebounds['UpperEnergy'])
    eav = np.sqrt(emin * emax)
    ok = np.where(data["Source_Name"] == source)
    fl = data['Flux_Band'][ok][0]
    ratio0 = data['Unc_Flux_Band'][ok][0][:, 0] / fl
    ratio1 = data['Unc_Flux_Band'][ok][0][:, 1] / fl
    dfl0, dfl1 = fl * ratio0, fl * ratio1
    ul = np.isnan(dfl0)
    fl[ul] += 2 * dfl1[ul]

    ax1 = fig.add_subplot(1, 1, 1)
    ax1.errorbar(eav[1:], fl[1:], yerr=[-dfl0[1:], dfl1[1:]],
                 fmt='o', label="gll_psc_v35", color="pink", uplims=ul[1:], lw=2)

    for label, (x, y, y_err, emin_arr, emax_arr) in dataset_none.items():
        x, y, y_err = map(np.array, (x, y, y_err))
        ax1.errorbar(x, y, xerr=[x-emin_arr, emax_arr-x], yerr=y_err,
                     fmt='o', capsize=5, color='black', label=label)

    # ——— FIND BEST & WORST ΔChi2 ———
    best, worst = None, None
    best_delta, worst_delta = np.inf, -np.inf

    for row in fit_results_none[source]["No_Filtering"]:
        for result in row:
            delta = result["fit_result"]["DeltaChi2"]
            if delta < best_delta:
                best_delta, best = delta, result
            if delta > worst_delta:
                worst_delta, worst = delta, result
    p0_best = best["p0"]
    ec_best = best["E_c"]
    p0_worst = worst["p0"]
    ec_worst = worst["E_c"]
    # ——— PLOT BEST Base & WORST Axion ———
    def axion_func(E, Norm, alpha_, beta_, w, p0, E_c, k=2.71):
        return logpar_base(E, Norm, alpha_, beta_) * (1 - (p0 / (1 + (E_c / E) ** k)) * (1+0.2*np.tanh(w)))
    all_x = np.concatenate([np.array(vals[0]) for vals in dataset_none.values()])
    x_grid = np.logspace(np.log10(all_x.min()), np.log10(all_x.max()), 300)

    y_best_base = logpar_base(x_grid, *best["fit_result"]["Base"]["params"])
    ax1.plot(x_grid, y_best_base,
             label=f"Best Base (Δχ²={best_delta:.2f})", linewidth=2)
    
    y_worst_base = logpar_base(x_grid, *worst["fit_result"]["Base"]["params"])
    ax1.plot(x_grid, y_worst_base,
             label=f"Worst Base (Δχ²={best_delta:.2f})", linewidth=2)

    y_best_axion = axion_func(x_grid, *best["fit_result"]["Axion"]["params"], p0_best, ec_best)
    ax1.plot(x_grid, y_best_axion,
             linestyle="--", label=f"Best Axion (Δχ²={best_delta:.2f})", linewidth=2)
    
    y_worst_axion = axion_func(x_grid, *worst["fit_result"]["Axion"]["params"], p0_worst, ec_worst)
    ax1.plot(x_grid, y_worst_axion,
             linestyle="--", label=f"Worst Axion (Δχ²={worst_delta:.2f})", linewidth=2)

    print("Best p0, E_c:", best["p0"], best["E_c"])
    print("Worst p0, E_c:", worst["p0"], worst["E_c"])

    ax1.legend(loc='upper right')
    ax1.set_xscale('log'); ax1.set_yscale('log')
    ax1.set_ylabel('dN/dE [ photons/cm²/s/MeV ]')
    ax1.set_title(f'{source} - SNR Ratios')
    ax1.grid(True, which="both", linestyle="--", linewidth=0.5)
    fig.tight_layout()

    return fig



source_name = "4FGL J0319.8+4130"

source_name_cleaned = (
    source_name.replace(" ", "")
    .replace(".", "dot")
    .replace("+", "plus")
    .replace("-", "minus")
    .replace('"', '')  # Ensure no extra quotes remain
)

f_bin = fits.open(f'./fit_results/{source_name_cleaned}_fit_data_NONE.fits')
bin_data = f_bin[1].data

# Sort the data by the 'emin' column
sorted_indices = np.argsort(bin_data['emin'])  # Get sorted indices
sorted_data_none = bin_data[sorted_indices]  # Reorder the data using sorted indices
print("No filtering")
print(sorted_data_none)
print()

#print(sorted_data_snr5['geometric_mean'])
#print( sorted_data_snr5['flux_tot_value'])

datasets = {f"No_Filtering": (sorted_data_none['geometric_mean'], sorted_data_none['flux_tot_value'], sorted_data_none['flux_tot_error'], sorted_data_none['emin'], sorted_data_none['emax'] )}

print(source_name)

with open("all_results_none_31_logpar_no_sys_error.pkl", "rb") as file:
    all_results_none = pickle.load(file)

#print(all_results_none[source_name])

with PdfPages("./fit_results/best_worst_fits.pdf") as pdf:
    fig = simple_plot_fit(datasets, all_results_none, source_name, png_naming="")