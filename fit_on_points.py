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

def axion_func(E, Norm, alpha_, beta_, w):
        return logpar_base(E, Norm, alpha_, beta_) * (1 - (p0 / (1 + (E_c / E) ** k)) * (1+0.2*np.tanh(w)))

def find_best_worst_fits(fit_results):
    best, worst = None, None
    best_delta, worst_delta = np.inf, -np.inf

    for result in fit_results.values():
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

    f = fits.open('./test/gll_psc_v35.fit')
    data = f[1].data
    ebounds = f[5].data
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
    ax1.errorbar(
        eav[1:], fl[1:], yerr=[-dfl0[1:], dfl1[1:]],
        fmt='o', label="gll_psc_v35", color="pink", uplims=ul[1:], lw=2
    )

    for label, (x, y, y_err, emin_arr, emax_arr) in dataset_none.items():
        x, y, y_err = map(np.array, (x, y, y_err))
        ax1.errorbar(
            x, y,
            xerr=[x - emin_arr, emax_arr - x],
            yerr=y_err,
            fmt='o', capsize=5, color='black', label=label
        )

    # — Find & plot best/worst —
    bw = find_best_worst_fits(fit_results_none)
    all_x = np.concatenate([np.array(vals[0]) for vals in dataset_none.values()])
    x_grid = np.logspace(np.log10(all_x.min()), np.log10(all_x.max()), 300)

    for tag in ("best", "worst"):
        res = bw[tag]
        fit = res["fit_result"]
        label = f"{tag.capitalize()} ({'Base' if tag=='best' else 'Axion'}) Δχ²={fit['DeltaChi2']:.2f}"
        y = (logpar_base if tag=="best" else axion_func)(x_grid, *fit[tag.capitalize()]["params"])
        style = "-" if tag=="best" else "--"
        ax1.plot(x_grid, y, linestyle=style, linewidth=2, label=label)

        # extract p0 & E_c
        print(f"{tag.title()} p0 = {res['p0']}, E_c = {res['E_c']}")

    ax1.legend(loc='upper right')
    ax1.set_xscale('log'); ax1.set_yscale('log')
    ax1.set_ylabel('dN/dE [ photons/cm²/s/MeV ]')
    ax1.set_title(f'{source} - SNR Ratios')
    ax1.grid(True, which="both", linestyle="--", linewidth=0.5)

    fig.tight_layout()
    #fig.savefig("./fit_results/spectral_points_fit.png", dpi=600)
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

with PdfPages("./fit_results/best_worst_fits.pdf") as pdf:
    fig = simple_plot_fit(datasets, all_results_none, source_name, png_naming="")