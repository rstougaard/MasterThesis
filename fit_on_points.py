from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import shlex
from iminuit.cost import LeastSquares
from iminuit import Minuit
from matplotlib.backends.backend_pdf import PdfPages
import pickle
axion_data = np.load('./denys/Rikke/Data/scan12.npy')
ma_all = axion_data[:,0] #eV
g_all = axion_data[:,1] # GeV**-1
ec_all = axion_data[:,2]/1e6 #MeV
p0_all = axion_data[:,3]
k_all = axion_data[:,4]
k = np.mean(k_all)

n_g = 40
n_total = axion_data.shape[0]
n_mass = n_total // n_g

# For the (E_c, p₀) plot, we want the full grid.
# Reshape the columns for E_c (converted to MeV) and p₀ into a (n_mass, n_g) grid.
ec_all_full = (axion_data[:, 2] / 1e6).reshape(n_mass, n_g)
p0_all_full = axion_data[:, 3].reshape(n_mass, n_g)

# For the (mₐ, g) plot, extract the unique values.
# g is assumed to be the same for every mass, taken from the first 40 rows.
g_unique = axion_data[:n_g, 1]       # length = n_g
# mₐ is taken from every 40th row (i.e. each new mass in the outer loop)
mass_unique = axion_data[::n_g, 0]     # length = n_mass

# Define your desired start and stop values
m_start_val = 3e-10
g_start_val = 1e-13
m_stop_val  = 6e-9
g_stop_val  = 1e-11

# Find the row (mass) and column (g) indices closest to the desired start values.
row_start = np.argmin(np.abs(mass_unique - m_start_val))
col_start = np.argmin(np.abs(g_unique - g_start_val))

# Similarly, find the indices closest to the desired stop values.
row_stop  = np.argmin(np.abs(mass_unique - m_stop_val))
col_stop  = np.argmin(np.abs(g_unique - g_stop_val))

# For clarity, you can print out the selected values:
print("Selected mass start:", mass_unique[row_start])
print("Selected g start:", g_unique[col_start])
print("Selected mass stop:", mass_unique[row_stop])
print("Selected g stop:", g_unique[col_stop])

# Now filter the full grid arrays.
ec_masked = ec_all_full[row_start:row_stop+1, col_start:col_stop+1]
p0_masked = p0_all_full[row_start:row_stop+1, col_start:col_stop+1]

# Also filter the unique arrays.
m_masked = mass_unique[row_start:row_stop+1]
g_masked    = g_unique[col_start:col_stop+1]

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
             label=f"Best Base", linewidth=2)
    
    y_worst_base = logpar_base(x_grid, *worst["fit_result"]["Base"]["params"])
    ax1.plot(x_grid, y_worst_base,
             label=f"Worst Base", linewidth=2)

    y_best_axion = axion_func(x_grid, *best["fit_result"]["Axion"]["params"], p0_best, ec_best)
    ax1.plot(x_grid, y_best_axion,
             linestyle="--", label=f"Best Axion)", linewidth=2)
    
    y_worst_axion = axion_func(x_grid, *worst["fit_result"]["Axion"]["params"], p0_worst, ec_worst)
    ax1.plot(x_grid, y_worst_axion,
             linestyle="--", label=f"Worst Axion", linewidth=2)
    
    textstr = (
    f"Best Δχ² = {best_delta:.2f}\n"
    f"Worst Δχ² = {worst_delta:.2f}"
    )

    # Place a little box in the upper‑left corner of the axes
    ax1.text(
        0.05, 0.95, textstr,
        transform=ax1.transAxes,
        fontsize=10,
        verticalalignment='top',
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.7)
)

    print("Best p0, E_c:", best["p0"], best["E_c"])
    print("Worst p0, E_c:", worst["p0"], worst["E_c"])
    # Find the grid‐cell indices for best fit
    mask_best = np.isclose(p0_masked, p0_best) & np.isclose(ec_masked, ec_best)
    i_best, j_best = np.where(mask_best)
    m_best = m_masked[i_best[0]]
    g_best = g_masked[j_best[0]]

    # Find the grid‐cell indices for worst fit
    mask_worst = np.isclose(p0_masked, p0_worst) & np.isclose(ec_masked, ec_worst)
    i_worst, j_worst = np.where(mask_worst)
    m_worst = m_masked[i_worst[0]]
    g_worst = g_masked[j_worst[0]]

    print(f"Best fit → mass = {m_best}, coupling = {g_best}")
    print(f"Worst fit → mass = {m_worst}, coupling = {g_worst}")

    ax1.legend(loc='lower left')
    ax1.set_xscale('log'); ax1.set_yscale('log')
    ax1.set_ylabel('dN/dE [ photons/cm²/s/MeV ]')
    ax1.set_title(f'{source} - No filtering')
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
    pdf.savefig(fig)
    plt.close(fig)