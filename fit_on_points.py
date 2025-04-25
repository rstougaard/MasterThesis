import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from naima.models import EblAbsorptionModel
import pickle
import astropy.units as u

# ————— Global plotting settings —————
plt.rcParams["text.usetex"]      = True
plt.rcParams["font.family"]      = "serif"
plt.rcParams["font.serif"]       = ["Computer Modern Roman"]
plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams.update({
    "font.size":        16,
    "legend.fontsize":  16,
    "axes.titlesize":   24,
    "axes.labelsize":   24,
    "xtick.labelsize":  22,
    "ytick.labelsize":  22,
    "xtick.direction":  "in",
    "ytick.direction":  "in",
    "xtick.top":        True,
    "ytick.right":      True,
    "xtick.major.size": 8,
    "ytick.major.size": 8,
    "xtick.minor.size": 5,
    "ytick.minor.size": 5,
    "xtick.major.width": 1.2,
    "ytick.major.width": 1.2,
    "xtick.minor.width": 0.8,
    "ytick.minor.width": 0.8,
})

# ————— Load grid scan data —————
axion_data = np.load('./denys/Rikke/Data/scan12.npy')
ma_all = axion_data[:,0]    # eV
g_all = axion_data[:,1]     # GeV^-1
ec_all = axion_data[:,2]/1e6 # MeV
p0_all = axion_data[:,3]

n_g     = 40
n_total = axion_data.shape[0]
n_mass  = n_total // n_g

# reshape into (n_mass, n_g)
ec_all_full   = ec_all.reshape(n_mass, n_g)
p0_all_full   = p0_all.reshape(n_mass, n_g)
mass_all_full = ma_all.reshape(n_mass, n_g)
g_all_full    = g_all.reshape(n_mass, n_g)

# unique axes
g_unique    = g_all[:n_g]
mass_unique = ma_all[::n_g]

# full grid for lookup
_ec = ec_all_full
_p0 = p0_all_full
_m  = mass_all_full
_g  = g_all_full

# ————— Load fit results —————
with open("none_new0_sys_error.pkl", "rb") as f:
    all_results_none = pickle.load(f)

# ————— Prepare data points —————
source_name = "4FGL J0319.8+4130"
cleaned = (source_name.replace(" ", "").replace(".", "dot")
                         .replace("+", "plus").replace("-", "minus").replace('"',''))
f_bin = fits.open(f'./fit_results/{cleaned}_fit_data_NONE.fits')
sd = f_bin[1].data
sorted_idx = np.argsort(sd['emin'])
sd = sd[sorted_idx]
datasets = {
    "No_Filtering": (
        sd['geometric_mean'], sd['flux_tot_value'], sd['flux_tot_error'],
        sd['emin'], sd['emax']
    )
}

# ————— Model definitions —————
def logpar_base(x, Norm, alpha_, beta_, z):
    E_b = 1000  # MeV
    ebl = EblAbsorptionModel(z).transmission(x * u.MeV)
    return Norm * (x/E_b)**(-(alpha_ + beta_*np.log(x/E_b))) * ebl

def axion_mod(E, Norm, alpha, beta, w, p0, E_c, z, k=2.71):
    p00 = p0*(1 + 0.2*np.tanh(w))
    return logpar_base(E, Norm, alpha, beta, z) * (1 - (p00/(1 + (E_c/E)**k)))

# ————— Plotting function —————
def simple_plot_fit(dataset_dict, fit_results_dict, source):
    # load redshift
    with fits.open('table-4LAC-DR3-h.fits') as f:
        data1 = f[1].data
        z = data1['Redshift'][data1['Source_Name'] == source][0]

    # data points
    x_data, y_data, y_err, emin_arr, emax_arr = map(np.array, next(iter(dataset_dict.values())))
    mask = (y_data != 0) & (np.abs(y_data) >= 1e-13)
    x_m   = x_data[mask]
    y_m   = y_data[mask]
    err_m = y_err[mask]
    emin_m = emin_arr[mask]
    emax_m = emax_arr[mask]
    # error floor for last point
    err_m[-1] += 0.10 * y_m[-1]

    # flatten all fits into a dict
    fitspec = {}
    for i_row, row in enumerate(fit_results_dict[source]["No_Filtering"]):
        for j_col, r in enumerate(row):
            mv       = r["m"]
            gv       = r["g"]
            p0v       = r["p0"]
            ecv       = r["E_c"]
            dchi      = r["fit_result"]["DeltaChi2"]
            bchi      = r["fit_result"]["Base"]["chi2"]
            bdof      = r["fit_result"]["Base"]["dof"]
            achi      = r["fit_result"]["Axion"]["chi2"]
            adof      = r["fit_result"]["Axion"]["dof"]
            params_b  = r["fit_result"]["Base"]["params"]
            params_a  = r["fit_result"]["Axion"]["params"]
            
            key = f"fit_{i_row}_{j_col}"
            fitspec[key] = {
                "p0": p0v, "ec": ecv, "delta": dchi,
                "chi2_base": bchi, "dof_base": bdof,
                "chi2_axion": achi, "dof_axion": adof,
                "params_base": params_b, "params_axion": params_a,
                "m": mv, "g": gv
            }
    target_delta = -6.578
    # find exact or nearest by Δχ²
    match = None
    for spec in fitspec.values():
        if np.isclose(spec["delta"], target_delta, atol=1e-3):
            match = spec
            print(f"Exact Δχ² match: Δχ² = {spec['delta']}")
            break
    if match is None:
        print("No exact Δχ² match. Using nearest Δχ²...")
        nearest_key = min(fitspec, key=lambda k: abs(fitspec[k]["delta"] - target_delta))
        match = fitspec[nearest_key]
        print(f"Nearest Δχ² = {match['delta']}")


    # ensure parameters present
    params_b = match["params_base"]
    params_a = match["params_axion"]

    # compute model curves
    x_grid = np.logspace(np.log10(x_m.min()), np.log10(x_m.max()), 300)
    base_c  = logpar_base(x_grid, *params_b, z)
    axion_c = axion_mod(x_grid, *params_a, match["p0"], match["ec"], z)

    # plotting
    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, sharex=True, figsize=(10,8),
        gridspec_kw={"height_ratios":[3,1]}
    )
    
    ax_top.plot(x_grid, base_c,  color='darkorange', lw=2, label='Base fit')
    ax_top.plot(x_grid, axion_c, color='blue',  ls='--', lw=2, label='ALP fit')
    ax_top.set_yscale('log')
    ax_top.set_ylabel(r'E$^2$dN/dE [erg/cm$^2$/s]')
    
    ax_top.grid(True, which='both', ls='--')
    ax_top.errorbar(
        x_m, y_m,
        xerr=[x_m-emin_m, emax_m-x_m],
        yerr=err_m,
        fmt='o', color='k', capsize=3, label='No filter data'
    )
    ax_top.legend(loc='upper right')

    # residuals
    resid_b = (y_m - logpar_base(x_m, *params_b, z)) / err_m
    resid_a = (y_m - axion_mod(x_m, *params_a, match["p0"], match["ec"], z)) / err_m
    ax_bot.errorbar(x_m, resid_b, fmt='s', color='darkorange', label='Base resid')
    ax_bot.errorbar(x_m, resid_a, fmt='o', color='blue',  label='ALP resid')
    for lvl, style in zip([0,1,-1,2,-2], ['-','--','--',':',':']):
        ax_bot.axhline(lvl, ls=style)
    ax_bot.set_xscale('log')
    ax_bot.set_ylim(-3,3)
    ax_bot.set_xlabel('Energy [MeV]')
    ax_bot.set_ylabel('Norm. Resid.')
    ax_bot.grid(True, ls='--')

    # annotation with LaTeX χ² and parameters
    textstr = (
        f"$\\chi^2_{{\\mathrm{{base}}}}/\\mathrm{{dof}} = {match['chi2_base']:.2f}/{match['dof_base']}$\n"
        f"$\\chi^2_{{\\mathrm{{axion}}}}/\\mathrm{{dof}} = {match['chi2_axion']:.2f}/{match['dof_axion']}$\n"
        f"$\\Delta\\chi^2 = {match['delta']:.2f}$\n\n"
        f"Base params: {', '.join(f'{v:.3g}' for v in params_b)}\n"
        f"Axion params: {', '.join(f'{v:.3g}' for v in params_a)}\n\n"
        f"$p_0 = {match['p0']:.2f},\\;E_c = {match['ec']:.1f}\\,$MeV\n"
        f"$m_a = {match['m']/1e-9:.2f}\\,\\mathrm{{neV}},\\;g{{a \gamma}} =$ {match['g']:.2g} GeV$^{-1}$"
    )
    ax_top.text(
        0.05, 0.05, textstr,
        transform=ax_top.transAxes,
        va='bottom', ha='left',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.7)
    )

    fig.tight_layout()
    plt.savefig(f"./fit_results/NGC_bestfits.png", dpi=300)
    return fig

if __name__ == '__main__':
    fig = simple_plot_fit(datasets, all_results_none, source_name)
