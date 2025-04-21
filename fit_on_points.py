from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import shlex
from iminuit.cost import LeastSquares
from iminuit import Minuit
from matplotlib.backends.backend_pdf import PdfPages
import pickle
from naima.models import EblAbsorptionModel
import astropy.units as u
plt.rcParams["text.usetex"]     = True
plt.rcParams["font.family"]    = "serif"
plt.rcParams["font.serif"]     = ["Computer Modern Roman"]
plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams.update({
    # Base font size for text in figures
    "font.size":          24,   # controls default text size (e.g. axis labels)
    # Legend
    "legend.fontsize":    22,   # default legend text size
    # Title and label sizes (override font.size if you like)
    "axes.titlesize":     24,
    "axes.labelsize":     24,
    # Tick labels
    "xtick.labelsize":    22,
    "ytick.labelsize":    22,
})
plt.rcParams.update({
    # tick‐label font size
    "xtick.labelsize":   20,
    "ytick.labelsize":   20,
    # tick direction and which sides
    "xtick.direction":   "in",
    "ytick.direction":   "in",
    "xtick.top":         True,
    "ytick.right":       True,
    # tick length (points)
    "xtick.major.size":  8,
    "ytick.major.size":  8,
    "xtick.minor.size":  5,
    "ytick.minor.size":  5,
    # tick width (points)
    "xtick.major.width": 1.2,
    "ytick.major.width": 1.2,
    "xtick.minor.width": 0.8,
    "ytick.minor.width": 0.8,
})

source_name = "4FGL J0319.8+4130"

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
g_all_full =  axion_data[:, 1].reshape(n_mass, n_g)
mass_all_ful =  axion_data[:, 0].reshape(n_mass, n_g)
# For the (mₐ, g) plot, extract the unique values.
# g is assumed to be the same for every mass, taken from the first 40 rows.
g_unique = axion_data[:n_g, 1]       # length = n_g
# mₐ is taken from every 40th row (i.e. each new mass in the outer loop)
mass_unique = axion_data[::n_g, 0]     # length = n_mass

# Define your desired start and stop values
m_start_val = 1e-10
g_start_val = 7.5e-13 #1e-13
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
m_masked = mass_all_ful[row_start:row_stop+1, col_start:col_stop+1]
g_masked = g_all_full[row_start:row_stop+1, col_start:col_stop+1]


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
    with fits.open('table-4LAC-DR3-h.fits') as f:
        data1 = f[1].data
        idx = (data1['Source_Name'] == source)
        z = data1['Redshift'][idx][0]

    def logpar_base(x, Norm, alpha_, beta_):
        E_b = 1000  # Fixed E_b value
        ebl = EblAbsorptionModel(z).transmission(x * u.MeV)
        return Norm * (x / E_b) ** (-(alpha_ + beta_ * np.log(x / E_b)))*ebl
    # ————— Load data & plot points —————
    
    # Pick first dataset for residuals
    first = next(iter(dataset_none.values()))
    x_data, y_data, y_err, emin_arr, emax_arr = map(np.array, first)
    #bin_size = emax_arr - emin_arr
    
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

    mask = (y_data != 0) & (np.abs(y_data) >= 1e-13)
    #binsize_masked = bin_size[mask]
    x_masked   = x_data[mask]
    y_masked   = y_data[mask]
    yerr_masked = y_err[mask]
    if not mask[-1]:
            y_err_eff = yerr_masked + 0.03 * y_masked
    else:
        y_err_eff0 = yerr_masked[:-1] + 0.03 * y_masked[:-1]
        y_err_eff1 = yerr_masked[-1] + 0.10 * y_masked[-1]
        y_err_eff = np.append(y_err_eff0, y_err_eff1)
    y_err_eff = np.array(y_err_eff)

    print(source)
    print(y_masked)
    

    all_x = np.concatenate([np.array(vals[0]) for vals in dataset_none.values()])
    x_grid = np.logspace(np.log10(x_masked.min()), np.log10(x_masked.max()), 300)

    # Model functions
    def axion_func(E, Norm, alpha, beta, w, p0, E_c, k=2.71):
        p00 = p0*(1+0.2*np.tanh(w))
        return logpar_base(E, Norm, alpha, beta) * (1 - (p00/(1+(E_c/E)**k)))

    fitspec = {}
    for tag, res, p0, ec in [("best", best, p0_best, ec_best), ("worst", worst, p0_worst, ec_worst)]:
        fit = res["fit_result"]
        base_curve = logpar_base(x_grid, *fit["Base"]["params"])
        axion_curve = axion_func(x_grid, *fit["Axion"]["params"], p0, ec)
        # find indices in grid
        mask_grid = np.isclose(p0_masked, p0) & np.isclose(ec_masked, ec)
        i, j = np.where(mask_grid)
        m_val = m_masked[mask_grid]
        g_val = g_masked[mask_grid]
        print(m_val, g_val)
        fitspec[tag] = {
            "base": base_curve,
            "axion": axion_curve,
            "delta": res["fit_result"]["DeltaChi2"],
            "chi2_base": fit["Base"]["chi2"], "dof_base": fit["Base"]["dof"],
            "chi2_axion": fit["Axion"]["chi2"], "dof_axion": fit["Axion"]["dof"],
            "p0": p0, "ec": ec, "m": m_val, "g": g_val,
            "params_axion": fit["Axion"]["params"],
            "params_base": fit["Base"]["params"]
        }

    def make_figure(tag):
        fig, (ax_top, ax_bot) = plt.subplots(2,1, sharex=True, figsize=(10,8), gridspec_kw={"height_ratios":[3,1]})
        spec = fitspec[tag]


        # Upper: data + fits
        #ax_top.errorbar(eav[1:], fl[1:], yerr=[-dfl0[1:],dfl1[1:]], fmt='o', uplims=ul[1:], label="gll_psc_v35")
        for label,(x,y,y_err,emin_arr,emax_arr) in dataset_none.items():
            ax_top.errorbar(x,y,xerr=[x-emin_arr,emax_arr-x], yerr=y_err, fmt='o', color="black", capsize=3, label=label)
        ax_top.plot(x_grid, spec["base"], label=f"{tag.capitalize()} Base", color="orange",linewidth=2)
        ax_top.plot(x_grid, spec["axion"], linestyle="--", color="green",label=f"{tag.capitalize()} Axion", linewidth=2)
        ax_top.set_ylabel(r'E$^2$dN/dE [erg/cm$^2$/s]')
        ax_top.set_yscale('log'); ax_top.legend(loc='upper right')
        ax_top.grid(True, which='both', linestyle='--')


                # Axion residuals
        base_resid = (y_masked - logpar_base(x_masked, *spec['params_base'])) / yerr_masked
        axion_resid = (y_masked - axion_func(x_masked, *spec['params_axion'], spec['p0'], spec['ec'])) / yerr_masked

        ax_bot.errorbar(x_masked, base_resid, fmt='s', color="orange", label='Base residuals')
        ax_bot.errorbar(x_masked, axion_resid, fmt='o', color="green", label='Axion residuals')
        for level,style in zip([0,1,-1,2,-2], ['-','--','--',':',':']):
            ax_bot.axhline(level, linestyle=style)
        ax_bot.set_xscale('log'); ax_bot.set_ylim(-3,3)
        ax_bot.set_xlabel('Energy [MeV]'); ax_bot.set_ylabel('Normalized Residuals')
        ax_bot.grid(True, linestyle='--')
        base_chi2, base_dof = spec['chi2_base'], spec['dof_base']
        axion_chi2, axion_dof = spec['chi2_axion'], spec['dof_axion']
        delta = spec['delta']
        textstr = (
            f"Base $\chi ^2$/dof = {base_chi2[0]:.2f}/{base_dof}\n"
            f"Axion $\chi ^2$/dof = {axion_chi2:.2f}/{axion_dof}\n"
            f"$\Delta \chi ^2$ = {delta:.2f}\n"
            f"Base params: {', '.join(f'{v:.3g}' for v in spec['params_base'])}\n"
            f"Axion params: {', '.join(f'{v:.3g}' for v in spec['params_axion'])}\n\n"
            f"p$_0$={spec['p0']:.3f}, E$_c$={spec['ec']:.1f}\n"
            f"m={spec['m']/1e-9:.3f}, g={spec['g']:.3e}"
        )

        ax_top.text(
            0.05, 0.05, textstr,
            transform=ax_top.transAxes,
            verticalalignment='bottom',
            horizontalalignment='left',
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.7)
        )

        #ax_top.set_title(f"{source} : {tag.capitalize()} Fit ($\Delta \chi ^2$={spec['delta']:.2f})")
        fig.tight_layout()
        plt.savefig("./fit_results/NGC1275_bestfits.png", dpi=300)
        return fig

    fig_best = make_figure("best")
    #fig_worst = make_figure("worst")
    return fig_best


with open("none_new0_sys_error.pkl", "rb") as file:
    all_results_none = pickle.load(file)

# Clean filename exactly as you already do
cleaned = (
    source_name.replace(" ", "")
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
'''
if source == "4FGL J0617.7-1715":
    # Stack the arrays as columns; ensure that they are numpy arrays (or convert them if needed)
    data = np.column_stack((
        sd['geometric_mean'], 
        sd['flux_tot_value'], 
        sd['flux_tot_error'],
        sd['emin'], 
        sd['emax']
    ))
    
    # Define a header for clarity in the text file
    header = "geometric_mean flux flux_error emin emax"
    
    # Save the data to a text file. Adjust the format (here '%f') if you need different precision.
    np.savetxt("output_newmodel.txt", data, header=header, fmt='%s')
'''

# Generate the two figures
fig_best = simple_plot_fit(datasets, all_results_none, source_name)


#print(f"Saved all best/worst fits into {output_pdf}")