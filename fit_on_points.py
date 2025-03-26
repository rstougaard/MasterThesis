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
g_start_val = 6e-13
m_stop_val  = 1e-8
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
    import numpy as np
    import matplotlib.pyplot as plt
    from astropy.io import fits

    # ————— Load data & plot points —————
    f = fits.open('./test/gll_psc_v35.fit')
    data = f[1].data; ebounds = f[5].data
    emin = np.unique(ebounds['LowerEnergy'])
    emax = np.unique(ebounds['UpperEnergy'])
    eav = np.sqrt(emin * emax)
    ok = np.where(data["Source_Name"] == source)
    fl = data['Flux_Band'][ok][0]
    ratio0 = data['Unc_Flux_Band'][ok][0][:,0]/fl
    ratio1 = data['Unc_Flux_Band'][ok][0][:,1]/fl
    dfl0, dfl1 = fl*ratio0, fl*ratio1
    ul = np.isnan(dfl0)
    fl[ul] += 2*dfl1[ul]

    # Pick first dataset for residuals
    first = next(iter(dataset_none.values()))
    x_data, y_data, y_err, emin_arr, emax_arr = map(np.array, first)

    # ————— Find best & worst —————
    best, worst = None, None
    best_delta, worst_delta = np.inf, -np.inf
    for row in fit_results_none[source]["No_Filtering"]:
        for r in row:
            d = r["fit_result"]["DeltaChi2"]
            if d < best_delta:
                best_delta, best = d, r
            if d > worst_delta:
                worst_delta, worst = d, r

    p0_best, ec_best = best["p0"], best["E_c"]
    p0_worst, ec_worst = worst["p0"], worst["E_c"]

    all_x = np.concatenate([np.array(vals[0]) for vals in dataset_none.values()])
    x_grid = np.logspace(np.log10(all_x.min()), np.log10(all_x.max()), 300)

    # Model functions
    def axion_func(E, Norm, alpha, beta, w, p0, E_c, k=2.71):
        return logpar_base(E, Norm, alpha, beta) * (1 - (p0/(1+(E_c/E)**k))*(1+0.2*np.tanh(w)))

    # Compute fits
    fitspec = {
        "best": {
            "base": logpar_base(x_grid, *best["fit_result"]["Base"]["params"]),
            "axion": axion_func(x_grid, *best["fit_result"]["Axion"]["params"], p0_best, ec_best),
            "delta": best_delta,
            "params": best["fit_result"]["Axion"]["params"]
        },
        "worst": {
            "base": logpar_base(x_grid, *worst["fit_result"]["Base"]["params"]),
            "axion": axion_func(x_grid, *worst["fit_result"]["Axion"]["params"], p0_worst, ec_worst),
            "delta": worst_delta,
            "params": worst["fit_result"]["Axion"]["params"]
        }
    }

    def make_figure(tag):
        fig, (ax_top, ax_bot) = plt.subplots(2,1, sharex=True, figsize=(10,8), gridspec_kw={"height_ratios":[3,1]})
        spec = fitspec[tag]

        # Upper: data + fits
        ax_top.errorbar(eav[1:], fl[1:], yerr=[-dfl0[1:],dfl1[1:]], fmt='o', uplims=ul[1:], label="gll_psc_v35")
        for label,(x,y,y_err,emin_arr,emax_arr) in dataset_none.items():
            ax_top.errorbar(x,y,xerr=[x-emin_arr,emax_arr-x], yerr=y_err, fmt='o', color="black", capsize=3, label=label)
        ax_top.plot(x_grid, spec["base"], label=f"{tag.capitalize()} Base", linewidth=2)
        ax_top.plot(x_grid, spec["axion"], linestyle="--", label=f"{tag.capitalize()} Axion", linewidth=2)
        ax_top.set_ylabel('dN/dE')
        ax_top.set_yscale('log'); ax_top.legend(loc='upper right')
        ax_top.grid(True, which='both', linestyle='--')

        
        base_params = best["fit_result"]["Base"]["params"] if tag=="best" else worst["fit_result"]["Base"]["params"]
        resid_base = (y_data - logpar_base(x_data, *base_params)) / y_err

        # Compute Axion residuals (as before)
        axion_params = spec["params"]
        resid_axion = (y_data - axion_func(x_data, *axion_params, (p0_best if tag=="best" else p0_worst), (ec_best if tag=="best" else ec_worst))) / y_err

        ax_bot.errorbar(x_data, resid_base, fmt='s', label='Base residuals')
        ax_bot.errorbar(x_data, resid_axion, fmt='o', label='Axion residuals')
        for level,style in zip([0,1,-1,2,-2], ['-','--','--',':',':']):
            ax_bot.axhline(level, linestyle=style)
        ax_bot.set_xscale('log'); ax_bot.set_ylim(-3,3)
        ax_bot.set_xlabel('Energy [MeV]'); ax_bot.set_ylabel('Residuals')
        ax_bot.grid(True, which='both', linestyle='--')
        # Pull fit‑stats
        fit = best["fit_result"] if tag=="best" else worst["fit_result"]
        base_chi2 = fit["Base"]["chi2"]
        base_dof  = fit["Base"]["dof"]
        axion_chi2= fit["Axion"]["chi2"]
        axion_dof = fit["Axion"]["dof"]
        delta     = spec["delta"]

        textstr = (
            f"Base χ²/dof = {base_chi2:.2f}/{base_dof}\n"
            f"Axion χ²/dof = {axion_chi2:.2f}/{axion_dof}\n"
            f"Δχ² = {delta:.2f}"
        )

        ax_top.text(
            0.05, 0.05, textstr,
            transform=ax_top.transAxes,
            fontsize=10,
            verticalalignment='bottom',
            horizontalalignment='left',
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.7)
        )

        ax_top.set_title(f"{source} — {tag.capitalize()} Fit (Δχ²={spec['delta']:.2f})")
        fig.tight_layout()
        return fig

    fig_best = make_figure("best")
    fig_worst = make_figure("worst")
    return fig_best, fig_worst


with open("all_results_none_31_logpar_no_sys_error.pkl", "rb") as file:
    all_results_none = pickle.load(file)
output_pdf = "./fit_results/best_worst_fits.pdf"

with PdfPages(output_pdf) as pdf:
    for source in all_results_none.keys():
        # Clean filename exactly as you already do
        cleaned = (
            source.replace(" ", "")
                  .replace(".", "dot")
                  .replace("+", "plus")
                  .replace("-", "minus")
                  .replace('"', "")
        )

        # Load & sort that source’s spectral‑points FITS
        f_bin = fits.open(f'./fit_results/{cleaned}_fit_data_NONE.fits')
        bin_data = f_bin[1].data
        sorted_idx = np.argsort(bin_data['emin'])
        sd = bin_data[sorted_idx]

        # Build the single “No_Filtering” dataset dict
        datasets = {
            "No_Filtering": (
                sd['geometric_mean'], sd['flux_tot_value'], sd['flux_tot_error'],
                sd['emin'], sd['emax']
            )
        }

        # Generate the two figures
        fig_best, fig_worst = simple_plot_fit(datasets, all_results_none, source)

        # Save each as a new page
        pdf.savefig(fig_best)
        plt.close(fig_best)
        pdf.savefig(fig_worst)
        plt.close(fig_worst)

print(f"Saved all best/worst fits into {output_pdf}")