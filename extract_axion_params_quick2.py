from astropy.io import fits
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from scipy.optimize import curve_fit
import shlex
from naima.models import EblAbsorptionModel
import astropy.units as u
from iminuit.cost import LeastSquares
from iminuit import Minuit
from tqdm import tqdm
from matplotlib.backends.backend_pdf import PdfPages
import pickle
from joblib import Parallel, delayed

# Import numba and use it with default (object mode) because not all functions are jittable
from numba import jit

path_to_save_heatmap_m_g = "./fit_results/heatmaps_m_g/"
path_to_save_heatmap_Ec_p0 = "./fit_results/heatmaps_Ec_p0/"

# Load axion data
axion_data = np.load('./denys/Rikke/Data/scan12.npy')
ma_all = axion_data[:, 0]   # eV
g_all = axion_data[:, 1]    # GeV**-1
ec_all = axion_data[:, 2] / 1e6  # MeV
p0_all = axion_data[:, 3]
k_all = axion_data[:, 4]
k = np.mean(k_all)

n_g = 40
n_total = axion_data.shape[0]
n_mass = n_total // n_g

# For the (E_c, p₀) plot, we want the full grid.
ec_all_full = (axion_data[:, 2] / 1e6).reshape(n_mass, n_g)
p0_all_full = p0_all.reshape(n_mass, n_g)

# For the (mₐ, g) plot, extract the unique values.
g_unique = axion_data[:n_g, 1]   # length = n_g
mass_unique = axion_data[::n_g, 0] # length = n_mass

# Define desired start and stop values
m_start_val = 1e-10
g_start_val = 5e-13
m_stop_val  = 1e-8
g_stop_val  = 1e-11

# Find the row (mass) and column (g) indices closest to the desired start/stop values.
row_start = np.argmin(np.abs(mass_unique - m_start_val))
col_start = np.argmin(np.abs(g_unique - g_start_val))
row_stop  = np.argmin(np.abs(mass_unique - m_stop_val))
col_stop  = np.argmin(np.abs(g_unique - g_stop_val))

print("Selected mass start:", mass_unique[row_start])
print("Selected g start:", g_unique[col_start])
print("Selected mass stop:", mass_unique[row_stop])
print("Selected g stop:", g_unique[col_stop])

# Filter the full grid arrays.
ec_masked = ec_all_full[row_start:row_stop+1, col_start:col_stop+1]
p0_masked = p0_all_full[row_start:row_stop+1, col_start:col_stop+1]

# Also filter the unique arrays.
m_masked = mass_unique[row_start:row_stop+1]
g_masked = g_unique[col_start:col_stop+1]


# --- Base functions and chi-square computation ---

def logpar_base(x, Norm, alpha_, beta_):
    E_b = 1000  # Fixed E_b value
    return Norm * (x / E_b) ** (-(alpha_ + beta_ * np.log(x / E_b)))

def cutoff_base(x, Norm, l1, l2):
    E_b = 1000
    return np.piecewise(x, [x < E_b, x >= E_b],
                        [lambda x: Norm*(x/E_b)**l1, lambda x: Norm*(x/E_b)**l2])

def reduced_chi_square(y_obs, y_fit, y_err, num_params):
    residuals = (y_obs - y_fit) / y_err
    chi2 = np.sum(residuals**2)
    dof = len(y_obs) - num_params  # Degrees of freedom
    return chi2, dof


# --- The fitting function (unchanged) ---

def fit_data(x, y, y_err, emin, emax, bin_size, p0, E_c, k, source_name, dataset_label, useEBL=True, fitting_method="no_sys_error", basefunc="cutoff"):
    # Filter out points where y is zero
    mask = (y != 0) & (np.abs(y) >= 1e-13)
    x_filtered = x[mask]
    y_filtered = y[mask]
    y_err_filtered = y_err[mask]
    emin_f = emin[mask]
    emax_f = emax[mask]
    bin_size_masked = bin_size[mask]

    if fitting_method in ["with_sys_error", "sys_error"]:
        if not mask[-1]:
            y_err_eff = y_err_filtered + 0.03 * y_filtered
        else:
            y_err_eff0 = y_err_filtered[:-1] + 0.03 * y_filtered[:-1]
            y_err_eff1 = y_err_filtered[-1] + 0.10 * y_filtered[-1]
            y_err_eff = np.append(y_err_eff0, y_err_eff1)
        y_err_eff = np.array(y_err_eff)
    elif fitting_method == "no_sys_error":
        y_err_eff = y_err_filtered
    else:
        raise ValueError(f"Unknown fitting_method: {fitting_method}")

    if useEBL and basefunc == "logpar":
        with fits.open('table-4LAC-DR3-h.fits') as f:
            data1 = f[1].data
        idx = (data1['Source_Name'] == source_name)
        z = data1['Redshift'][idx][0]

        def LogPar_EBL(E, Norm, alpha_, beta_):
            ebl = EblAbsorptionModel(z).transmission(E * u.MeV)
            return logpar_base(E, Norm, alpha_, beta_) * ebl

        base = LogPar_EBL
        bounds_base = ([1e-14, -5.0, -5.0], [1e-9, 5.0, 3.0])
        p0_base = [1e-11, 2.0, 0.001]

        def axion_func(E, Norm, alpha_, beta_, w):
            return base(E, Norm, alpha_, beta_) * (1 - (p0 / (1 + (E_c / E)**k)) * (1+0.2*np.tanh(w)))
        bounds_alp = ([1e-14, -5.0, -5.0, -np.pi], [1e-9, 5.0, 5.0, np.pi])
        p0_alp = [1e-11, 2.0, 0.001, np.pi/2]
    else:
        raise ValueError("Only EBL logpar fitting is implemented in this code example.")

    popt_base, pcov_base = curve_fit(
        base, x_filtered, y_filtered, sigma=y_err_eff, p0=p0_base, bounds=bounds_base,
        absolute_sigma=True, maxfev=100000)
    y_fit_base = base(x_filtered, *popt_base)
    chi2_base, dof_base = reduced_chi_square(y_filtered, y_fit_base, y_err_eff, len(popt_base))
    perr_base = np.sqrt(np.diag(pcov_base))

    popt_axion, pcov_axion = curve_fit(
        axion_func, x_filtered, y_filtered, sigma=y_err_eff, p0=p0_alp, bounds=bounds_alp,
        absolute_sigma=True, maxfev=100000)
    y_fit_axion = axion_func(x_filtered, *popt_axion)
    chi2_axion, dof_axion = reduced_chi_square(y_filtered, y_fit_axion, y_err_eff, len(popt_axion))
    perr_axion = np.sqrt(np.diag(pcov_axion))

    delta_chi2 = chi2_axion - chi2_base

    if useEBL and basefunc == "logpar":
        print("LogPar:\n")
        for param, value, error in zip(["Norm", "alpha_", "beta_"], popt_base, perr_base):
            print(f"  {param}: {value:.2e} ± {error:.2e}\n")
        print(f'χ² / dof: {chi2_base:.2f} / {dof_base}\n\n')
        print("Axion:\n")
        for param, value, error in zip(["Norm", "alpha_", "beta_", "w"], popt_axion, perr_axion):
            print(f"  {param}: {value:.2e} ± {error:.2e}\n")
        print(f"χ² / dof: {chi2_axion:.2f} / {dof_axion}\n\n")
        print(f"Δχ² (Axion - LogPar): {delta_chi2:.2f}")
        print('--------------------------------------------------------------------------------------')

    return {
        "Base": {
            "params": popt_base,
            "errors": perr_base,
            "chi2": chi2_base,
            "dof": dof_base
        },
        "Axion": {
            "params": popt_axion,
            "errors": perr_axion,
            "chi2": chi2_axion,
            "dof": dof_axion,
        },
        "DeltaChi2": delta_chi2,
        "y_fit_LogPar": y_fit_base,
        "y_fit_Axion": y_fit_axion,
    }


# --- Process-chunk function with a Numba jit decorator ---
# We use nopython=False because the function uses unsupported objects.
@jit(nopython=False)
def process_chunk(i, j_start, j_end, x, y, y_err, emin, emax, bin_size, source_name, dataset_label, useEBL, fitting_method, basefunc):
    results_chunk = []
    # Note: p0_masked and ec_masked are globals
    p0_chunk = p0_masked[i, j_start:j_end]
    ec_chunk = ec_masked[i, j_start:j_end]
    for p0_val, ec_val in zip(p0_chunk, ec_chunk):
        fit_result = fit_data(
            x=np.array(x),
            y=np.array(y),
            y_err=np.array(y_err),
            emin=np.array(emin),
            emax=np.array(emax),
            bin_size=np.array(bin_size),
            p0=p0_val,
            E_c=ec_val,
            k=k,  # k defined globally
            source_name=source_name,
            dataset_label=dataset_label,
            useEBL=useEBL,
            fitting_method=fitting_method,
            basefunc=basefunc
        )
        results_chunk.append({
            "p0": p0_val,
            "E_c": ec_val,
            "fit_result": fit_result
        })
    return results_chunk


def nested_fits_combined(datasets, bin_size, source_name, useEBL=True, fitting_method="no_sys_error", basefunc="cutoff", chunk_size=10):
    """
    Processes each row (mass) of the masked (p0, Ec) grid.
    The row is split into chunks that are processed in parallel (via joblib),
    and the inner processing in each chunk is handled by the numba-compiled process_chunk.
    """
    results = {}
    for dataset_label, (x, y, y_err, emin, emax) in datasets.items():
        dataset_results = []
        num_rows = p0_masked.shape[0]
        num_cols = p0_masked.shape[1]
        # Process each row of the grid.
        for i in range(num_rows):
            tasks = []
            # Split the row into chunks.
            for j_start in range(0, num_cols, chunk_size):
                j_end = min(j_start + chunk_size, num_cols)
                tasks.append(delayed(process_chunk)(
                    i, j_start, j_end, 
                    np.array(x), np.array(y), np.array(y_err),
                    np.array(emin), np.array(emax), np.array(bin_size),
                    source_name, dataset_label, useEBL, fitting_method, basefunc
                ))
            row_results = []
            chunk_results = Parallel(n_jobs=-1)(tasks)
            for chunk in chunk_results:
                row_results.extend(chunk)
            dataset_results.append(row_results)
        results[dataset_label] = dataset_results
    return results


# --- Main processing loop over sources ---

all_results_none = {}
all_results_snr = {}
all_results_lin = {}
all_results_none_sys = {}
all_results_snr_sys = {}
all_results_lin_sys = {}

with open('temp_sources.txt', 'r') as file:
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
        
        f_bin = fits.open(f'./fit_results/{source_name_cleaned}_fit_data_NONE.fits')
        f_bin_snr = fits.open(f'./fit_results/{source_name_cleaned}_fit_data_SNR.fits')
        f_bin_lin = fits.open(f'./fit_results/{source_name_cleaned}_fit_data_LIN.fits')
        bin_data = f_bin[1].data
        bin_data_snr = f_bin_snr[1].data
        bin_data_lin = f_bin_lin[1].data

        sorted_indices = np.argsort(bin_data['emin'])
        sorted_data_none = bin_data[sorted_indices]

        snr3 = bin_data_snr[bin_data_snr['loop_item'] == '3']
        sorted_indices_snr3 = np.argsort(snr3['emin'])
        sorted_data_snr3 = snr3[sorted_indices_snr3]

        snr5 = bin_data_snr[bin_data_snr['loop_item'] == '5']
        sorted_indices_snr5 = np.argsort(snr5['emin'])
        sorted_data_snr5 = snr5[sorted_indices_snr5]

        snr10 = bin_data_snr[bin_data_snr['loop_item'] == '10']
        sorted_indices_snr10 = np.argsort(snr10['emin'])
        sorted_data_snr10 = snr10[sorted_indices_snr10]

        week = bin_data_lin[bin_data_lin['loop_item'] == 'week']
        sorted_indices_lin_week = np.argsort(week['emin'])
        sorted_data_lin_week = week[sorted_indices_lin_week]
        month = bin_data_lin[bin_data_lin['loop_item'] == 'month']
        sorted_indices_lin_month = np.argsort(month['emin'])
        sorted_data_lin_month = month[sorted_indices_lin_month]

        colors_snr = ['blue', 'orange', 'green']
        colors_lin = ['purple', 'brown']
        bin_size = np.array(sorted_data_none['emax']) - np.array(sorted_data_none['emin'])
        e_lowers = sorted_data_none['geometric_mean'] - sorted_data_none['emin']
        e_uppers = np.array(sorted_data_none['emax']) - np.array(sorted_data_none['geometric_mean'])
        datasets = {
            "No_Filtering": (
                sorted_data_none['geometric_mean'],
                sorted_data_none['flux_tot_value'],
                sorted_data_none['flux_tot_error'],
                sorted_data_none['emin'],
                sorted_data_none['emax']
            )
        }
        datasets_snr = {
            "snr_3": (
                sorted_data_snr3['geometric_mean'],
                sorted_data_snr3['flux_tot_value'],
                sorted_data_snr3['flux_tot_error'],
                sorted_data_snr3['emin'],
                sorted_data_snr3['emax']
            ),
            "snr_5": (
                sorted_data_snr5['geometric_mean'],
                sorted_data_snr5['flux_tot_value'],
                sorted_data_snr5['flux_tot_error'],
                sorted_data_snr5['emin'],
                sorted_data_snr5['emax']
            ),
            "snr_10": (
                sorted_data_snr10['geometric_mean'],
                sorted_data_snr10['flux_tot_value'],
                sorted_data_snr10['flux_tot_error'],
                sorted_data_snr10['emin'],
                sorted_data_snr10['emax']
            )
        }
        datasets_lin = {
            "week": (
                sorted_data_lin_week['geometric_mean'],
                sorted_data_lin_week['flux_tot_value'],
                sorted_data_lin_week['flux_tot_error'],
                sorted_data_lin_week['emin'],
                sorted_data_lin_week['emax']
            ),
            "month": (
                sorted_data_lin_month['geometric_mean'],
                sorted_data_lin_month['flux_tot_value'],
                sorted_data_lin_month['flux_tot_error'],
                sorted_data_lin_month['emin'],
                sorted_data_lin_month['emax']
            )
        }
        print(source_name)
        # Run fits without systematic errors.
        results = nested_fits_combined(
            datasets, bin_size, source_name, useEBL=True, fitting_method="no_sys_error",
            basefunc="logpar", chunk_size=30
        )
        all_results_none[source_name] = results
        with open("all_results_none_32_numba_no_sys_error.pkl", "wb") as file_out:
            pickle.dump(all_results_none, file_out)

        # Run fits with systematic errors.
        results_sys = nested_fits_combined(
            datasets, bin_size, source_name, useEBL=True, fitting_method="sys_error",
            basefunc="logpar", chunk_size=30
        )
        all_results_none_sys[source_name] = results_sys
        with open("all_results_none_numba_sys_error.pkl", "wb") as file_out:
            pickle.dump(all_results_none_sys, file_out)

        
