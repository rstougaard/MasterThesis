from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import shlex
from iminuit.cost import LeastSquares
from iminuit import Minuit
from lmfit import Model
import itertools
from matplotlib.backends.backend_pdf import PdfPages

def simple_plot(dataset_none, dataset_snr, colors_snr, dataset_lin, colors_lin, source, png_naming=""):
    # Create a new figure
    fig = plt.figure(figsize=(8, 10))

    # Top subplot: Spectrum - SNR Ratios
    ax1 = fig.add_subplot(2, 1, 1)
    
    # Plot the "none" dataset (using black)
    for dataset_label, (x, y, y_err, emin, emax) in dataset_none.items():
        x, y, y_err, emin, emax = np.array(x), np.array(y), np.array(y_err), np.array(emin), np.array(emax)
        e_lowers = x - emin
        e_uppers = emax - x
        bin_size = emax - emin
        ax1.errorbar(x, y, xerr=[e_lowers, e_uppers], yerr=y_err,
                     fmt='o', capsize=5, color='black', label=f'{dataset_label}')
    
    # Plot the SNR datasets with their corresponding colors
    for i, (dataset_label, (x, y, y_err, emin, emax)) in enumerate(dataset_snr.items()):
        x, y, y_err, emin, emax = np.array(x), np.array(y), np.array(y_err), np.array(emin), np.array(emax)
        e_lowers = x - emin
        e_uppers = emax - x
        bin_size = emax - emin
        # Get the color based on the index (defaulting to black if index is out of range)
        color = colors_snr[i] if i < len(colors_snr) else 'black'
        ax1.errorbar(x, y, xerr=[e_lowers, e_uppers], yerr=y_err,
                     fmt='o', capsize=5, color=color, label=f'{dataset_label}')

    ax1.legend(ncol=1, loc='upper right')
    ax1.set_ylabel('dN/dE [ photons/cm²/s/MeV ]')
    ax1.set_title(f'{source} - SNR Ratios')
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.grid(True, which="both", linestyle="--", linewidth=0.5)

    # Bottom subplot: Spectrum - Time Intervals
    ax2 = fig.add_subplot(2, 1, 2)
    
    # Plot the "none" dataset (again in black)
    for dataset_label, (x, y, y_err, emin, emax) in dataset_none.items():
        x, y, y_err, emin, emax = np.array(x), np.array(y), np.array(y_err), np.array(emin), np.array(emax)
        e_lowers = x - emin
        e_uppers = emax - x
        bin_size = emax - emin
        ax2.errorbar(x, y, xerr=[e_lowers, e_uppers], yerr=y_err,
                     fmt='o', capsize=5, color='black', label=f'{dataset_label}')

    # Plot the lin datasets with their corresponding colors
    for i, (dataset_label, (x, y, y_err, emin, emax)) in enumerate(dataset_lin.items()):
        x, y, y_err, emin, emax = np.array(x), np.array(y), np.array(y_err), np.array(emin), np.array(emax)
        e_lowers = x - emin
        e_uppers = emax - x
        bin_size = emax - emin
        # Get the color based on the index
        color = colors_lin[i] if i < len(colors_lin) else 'black'
        ax2.errorbar(x, y, xerr=[e_lowers, e_uppers], yerr=y_err,
                     fmt='o', capsize=5, color=color, label=f'{dataset_label}')

    ax2.legend(ncol=1, loc='upper right')
    ax2.set_ylabel('dN/dE [ photons/cm²/s/MeV ]')
    ax2.set_xlabel('Energy [ MeV ]')
    ax2.set_title(f'{source} - Time Intervals')
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.grid(True, which="both", linestyle="--", linewidth=0.5)

    # Adjust layout to prevent overlap
    fig.tight_layout()

    # Optionally save the figure as a PNG
    #png_name = f'fit_results/speactral_points/{dataset_label}{png_naming}.png'
    #fig.savefig("./hpc_results/NGC1275/spectral_points.png", dpi=600)

    return fig


# Function 1: LogPar
def LogPar(x, Norm, alpha_, beta_):
    E_b = 957.8065399  # Fixed E_b value
    return Norm * (x / E_b) ** (-(alpha_ + beta_ * np.log(x / E_b)))

# Function 2: axion_func
def axion_func(x, Norm, alpha_, beta_, p0, E_c):
    E_b = 1000#957.8065399  # Fixed E_b value
    k = 2.7
    return (Norm * (x / E_b) ** (-(alpha_ + beta_ * np.log(x / E_b)))) * (1 - p0 / (1 + (E_c / x) ** k))

def reduced_chi_square(y_obs, y_fit, y_err, num_params):
    residuals = (y_obs - y_fit) / y_err
    chi2 = np.sum(residuals**2)
    dof = len(y_obs) - num_params  # Degrees of freedom
    return chi2 , dof


def curve_fit_data(x, y, y_err):
    # Define bounds for LogPar parameters [Norm, alpha_, beta_]
    bounds_logpar = ([1e-13, -1.0, -2.0], [1e-9, 5.0, 2.0])  # Lower and upper bounds
    p0_logpar = [1e-11, 2.0, 0.1]  # Initial guesses
    popt_logpar, pcov_logpar = curve_fit(
        LogPar, x, y, sigma=y_err, p0=p0_logpar, bounds=bounds_logpar, absolute_sigma=True, method="trf"
    )
    y_fit_logpar = LogPar(x, *popt_logpar)
    chi2_logpar, dof_logpar = reduced_chi_square(y, y_fit_logpar, y_err, len(popt_logpar))

    # Extract parameter uncertainties
    perr_logpar = np.sqrt(np.diag(pcov_logpar))

    # Define bounds for axion_func parameters [Norm, alpha_, beta_, p0, E_c, k]
    bounds_axion = ([1e-13, 1.0, -2.0, 0.0, 800, 0.0], [1e-9, 5.0, 2.0, 1/3, 1000000, 5.0])
    p0_axion = [1e-11, 2.0, 0.1, 0.15, 2500, 3.0]  # Initial guesses
    popt_axion, pcov_axion = curve_fit(
        axion_func, x, y, sigma=y_err, p0=p0_axion, bounds=bounds_axion, absolute_sigma=True, method="trf"
    )
    y_fit_axion = axion_func(x, *popt_axion)
    chi2_axion, dof_axion = reduced_chi_square(y, y_fit_axion, y_err, len(popt_axion))

    # Extract parameter uncertainties
    perr_axion = np.sqrt(np.diag(pcov_axion))

    # Compute Δχ²
    delta_chi2 = chi2_axion - chi2_logpar

    # Return fit results
    return {
        "LogPar": {
            "params": popt_logpar,
            "errors": perr_logpar,
            "cov": pcov_logpar,  # Return covariance matrix
            "chi2": chi2_logpar,
            "dof": dof_logpar
        },
        "Axion": {
            "params": popt_axion,
            "errors": perr_axion,
            "cov": pcov_axion,  # Return covariance matrix
            "chi2": chi2_axion,
            "dof": dof_axion
        },
        "DeltaChi2": delta_chi2,
        "y_fit_LogPar": y_fit_logpar,
        "y_fit_Axion": y_fit_axion,
    }
def iminuit_fit_data(x, y, y_err):
    
    # Define LogPar model function
    def LogPar_model(x, Norm, alpha_, beta_):
        return LogPar(x, Norm, alpha_, beta_)

    # Define Axion model function
    def Axion_model(x, Norm, alpha_, beta_, p0, E_c):
        return axion_func(x, Norm, alpha_, beta_, p0, E_c)

    # Least Squares for LogPar
    least_squares_logpar = LeastSquares(x, y, y_err, LogPar_model)
    p0_logpar = [1e-11, 2.0, 0.1]
    bounds_logpar = [(1e-13, 1e-9), (-1.0, 5.0), (-2.0, 2.0)]

    # Minuit fit for LogPar
    m_logpar = Minuit(least_squares_logpar, Norm=p0_logpar[0], alpha_=p0_logpar[1], beta_=p0_logpar[2])

    # Set parameter limits
    for param, bound in zip(m_logpar.parameters, bounds_logpar):
        m_logpar.limits[param] = bound

    print("\n=== LogPar Fit with iminuit ===")
    m_logpar.simplex() 
    m_logpar.migrad()  # Minimize
    m_logpar.hesse()   # Compute uncertainties
    print(m_logpar)

    # Extract results
    popt_logpar = [m_logpar.values[p] for p in m_logpar.parameters]
    perr_logpar = [m_logpar.errors[p] for p in m_logpar.parameters]
    y_fit_logpar = LogPar(x, *popt_logpar)
    chi2_logpar, dof_logpar = reduced_chi_square(y, y_fit_logpar, y_err, len(p0_logpar))

    # Least Squares for Axion
    least_squares_axion = LeastSquares(x, y, y_err, Axion_model)
    p0_axion = [1e-11, 2.0, 0.1, 0.20, 1700]
    bounds_axion = [(1e-13, 1e-9), (1.0, 5.0), (-2.0, 2.0), (0.0, 1/3), (800, 1000000)]

    # Minuit fit for Axion
    m_axion = Minuit(least_squares_axion, Norm=p0_axion[0], alpha_=p0_axion[1], beta_=p0_axion[2], 
                     p0=p0_axion[3], E_c=p0_axion[4])

    # Set parameter limits
    for param, bound in zip(m_axion.parameters, bounds_axion):
        m_axion.limits[param] = bound

    print("\n=== Axion Fit with iminuit ===")
    m_axion.simplex() 
    m_axion.migrad()  # Minimize
    m_axion.hesse()   # Compute uncertainties
    print(m_axion)

    # Extract results
    popt_axion = [m_axion.values[p] for p in m_axion.parameters]
    perr_axion = [m_axion.errors[p] for p in m_axion.parameters]
    y_fit_axion = axion_func(x, *popt_axion)
    chi2_axion, dof_axion = reduced_chi_square(y, y_fit_axion, y_err, len(p0_axion))

    # Compute Δχ²
    delta_chi2 = chi2_axion - chi2_logpar

    # Return fit results
    return {
        "LogPar": {
            "params": popt_logpar,
            "errors": perr_logpar,
            "chi2": chi2_logpar,
            "dof": dof_logpar
        },
        "Axion": {
            "params": popt_axion,
            "errors": perr_axion,
            "chi2": chi2_axion,
            "dof": dof_axion
        },
        "DeltaChi2": delta_chi2,
        "y_fit_LogPar": y_fit_logpar,
        "y_fit_Axion": y_fit_axion,
    }

def fit_data(method, datasets, png_naming="", residual_colors=None):
    results = {}

    # Select the appropriate fitting function
    if method == "curve_fit":
        fit_function = curve_fit_data
    elif method == "iminuit":
        fit_function = iminuit_fit_data
    else:
        raise ValueError("Invalid method. Choose from 'curve_fit', 'iminuit', or 'lmfit'.")

     # Apply the selected fitting function to each dataset
    for dataset_label, (x, y, y_err, emin, emax) in datasets.items():
        x, y, y_err, emin, emax = np.array(x), np.array(y), np.array(y_err), np.array(emin), np.array(emax)
        print(emin)
        print(emax)
        e_lowers = x - emin
        e_uppers = emax - x

        # Filter out points where y is zero
        mask = y != 0
        x_filtered, y_filtered, y_err_filtered, emin_filtered, emax_filtered, e_lowers_filtered, e_uppers_filtered = x[mask], y[mask], y_err[mask], emin[mask], emax[mask], e_lowers[mask], e_uppers[mask]
        bin_size = emax_filtered - emin_filtered
        y_filtered = y_filtered/bin_size
        y_err_filtered = y_err_filtered/bin_size
        print(bin_size)
        # Check if the last point was filtered out
        if not mask[-1]:  # If the last point was removed (y[-1] == 0)
            y_err_eff = y_err_filtered + 0.03 * y_filtered  # Only add 3% to errors
        else:  # Otherwise, handle the normal case
            y_err_eff0 = y_err_filtered[:-1] + 0.03 * y_filtered[:-1]
            y_err_eff1 = y_err_filtered[-1] + 0.10 * y_filtered[-1]
            y_err_eff = np.append(y_err_eff0, y_err_eff1)

        y_err_eff = np.array(y_err_eff)

        # Only fit if there are enough data points left
        if len(x_filtered) > 1:  # Ensure at least two points remain
            results[dataset_label] = fit_function(x_filtered, y_filtered, y_err_eff)
        else:
            results[dataset_label] = None  # Indicate insufficient data for fitting

            # Extract parameters and chi-squared values
        # Extract fit results
        y_fit_logpar = results[dataset_label]["y_fit_LogPar"]
        y_fit_axion = results[dataset_label]["y_fit_Axion"]
        params_logpar = results[dataset_label]["LogPar"]["params"]
        errors_logpar = results[dataset_label]["LogPar"]["errors"]
        params_axion = results[dataset_label]["Axion"]["params"]
        errors_axion = results[dataset_label]["Axion"]["errors"]
        chi2_logpar = results[dataset_label]["LogPar"]['chi2']
        chi2_axion = results[dataset_label]['Axion']['chi2']
        dof_logpar = results[dataset_label]["LogPar"]['dof']
        dof_axion = results[dataset_label]['Axion']['dof']
        chi_sq = results[dataset_label]['DeltaChi2']

        # Calculate normalized residuals
        residuals_logpar = (y_filtered - y_fit_logpar) / y_err_eff
        residuals_axion = (y_filtered - y_fit_axion) / y_err_eff

        # Default colors for residuals
        if residual_colors is None:
            residual_colors = {"LogPar": "red", "Axion": "blue"}
        # Create figure
        fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
        x_range = np.linspace(50, 1e6, int(1e4))
        x_removed = x[~mask]
        y_removed = y[~mask]
        y_err_removed = y_err[~mask]*2
        e_lowers_removed = e_lowers[~mask]
        e_uppers_removed = e_uppers[~mask]
        # Top plot: Data and fits
        axs[0].errorbar(x_filtered, y_filtered, xerr = [e_lowers_filtered, e_uppers_filtered],yerr=y_err_eff, fmt='o', label="Data", color='black')
        axs[0].plot(x_filtered, y_fit_logpar, label="LogPar Fit", linestyle='-', color=residual_colors["LogPar"])
        axs[0].plot(x_filtered, y_fit_axion, label="Axion Fit", linestyle='--', color=residual_colors["Axion"])
        #axs[0].plot(x_range, axion_func(x_range, *params_axion), label="axion_func(E)", linestyle='-', color='orange')
        #axs[0].plot(x_range, LogPar(x_range, *params_logpar), label="LogPar(E)", linestyle='-', color='green')
        axs[0].errorbar(x_removed, y_removed, xerr=[e_lowers_removed, e_uppers_removed], yerr=y_err_removed, fmt='o', 
                color='grey', alpha=0.5, label="Removed Data")
        axs[0].set_ylabel("dN/dE [ photons/cm²/s/MeV ]")
        axs[0].set_title(f"Fits for {dataset_label}")
        axs[0].set_xscale('log')
        axs[0].set_yscale('log')
        axs[0].set_ylim(1e-30, 1e-10)
        axs[0].legend()
        axs[0].grid(True, which="both", linestyle="--", linewidth=0.5)

        # Add parameter box
        textstr = "LogPar:\n"
        for param, value, error in zip(["Norm", "alpha_", "beta_"], params_logpar, errors_logpar):
            textstr += f"  {param}: {value:.2e} ± {error:.2e}\n"
        textstr += f"  $\chi^2$ / dof: {chi2_logpar:.2f} / {dof_logpar}\n\n"

        textstr += "Axion:\n"
        for param, value, error in zip(["Norm", "alpha_", "beta_", "p0", "E_c"], params_axion, errors_axion):
            textstr += f"  {param}: {value:.2e} ± {error:.2e}\n"
        textstr += f"  $\chi^2$ / dof: {chi2_axion:.2f} / {dof_axion}\n\n"

        textstr += f"Δχ² (Axion - LogPar): {chi_sq:.2f}"

        axs[0].text(0.02, 0.56, textstr, transform=axs[0].transAxes, fontsize=10,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        # Bottom plot: Residuals
        axs[1].scatter(x_filtered, residuals_logpar, label="LogPar Residuals", color=residual_colors["LogPar"], marker='o')
        axs[1].scatter(x_filtered, residuals_axion, label="Axion Residuals", color=residual_colors["Axion"], marker='x')
        axs[1].axhline(0, color='black', linestyle='--', linewidth=0.8)
        axs[1].axhline(1, color='green', linestyle='--', linewidth=0.8, label='+1σ')
        axs[1].axhline(-1, color='green', linestyle='--', linewidth=0.8)
        axs[1].axhline(2, color='orange', linestyle='--', linewidth=0.8, label='+2σ')
        axs[1].axhline(-2, color='orange', linestyle='--', linewidth=0.8)
        axs[1].set_ylim(-4, 4)
        axs[1].set_xlabel("Energy [ MeV ]")
        axs[1].set_ylabel("Normalized Residual")
        axs[1].legend(ncol=2, loc='upper right')
        axs[1].grid(True, which="both", linestyle="--", linewidth=0.5)

        # Adjust layout
        plt.tight_layout()
        # Select the appropriate fitting function
        if method == "curve_fit":
            png_name = f'hpc_results/fixed_k/{dataset_label}{png_naming}.png'
        elif method == "iminuit":
            png_name = f'hpc_results/fixed_k/minuit/{dataset_label}{png_naming}.png'
        plt.savefig(png_name, dpi=300)
        plt.close()
        # Print formatted results
    for label, result in results.items():
        print(f"Dataset: {label}")
        
        # LogPar results
        print("  LogPar Fit:")
        for param, value, error in zip(["Norm", "alpha_", "beta_"], result["LogPar"]["params"], result["LogPar"]["errors"]):
            print(f"    {param}: {value:.2e} ± {error:.2e}")
        print(f"    χ² / dof: {result['LogPar']['chi2']:.2f} / {result['LogPar']['dof']:.2f}")
        
        # Axion results
        print("  Axion Fit:")
        for param, value, error in zip(["Norm", "alpha_", "beta_", "p0", "E_c"], result["Axion"]["params"], result["Axion"]["errors"]):
            print(f"    {param}: {value:.4e} ± {error:.4e}")
        print(f"    χ² / dof: {result['Axion']['chi2']:.2f} / {result['Axion']['dof']:.2f}")

        # Δχ²
        print(f"  Δχ² (Axion - LogPar): {result['DeltaChi2']:.2f}\n")

    return results
'''
source_name = "4FGL J0319.8+4130"

source_name_cleaned = (
    source_name.replace(" ", "")
    .replace(".", "dot")
    .replace("+", "plus")
    .replace("-", "minus")
    .replace('"', '')  # Ensure no extra quotes remain
)

f_bin = fits.open(f'./hpc_results/{source_name_cleaned}_fit_data_NONE.fits')
f_bin_snr = fits.open(f'./hpc_results/{source_name_cleaned}_fit_data_SNR.fits')
f_bin_lin = fits.open(f'./hpc_results/{source_name_cleaned}_fit_data_LIN.fits')
bin_data = f_bin[1].data
bin_data_snr = f_bin_snr[1].data
bin_data_lin = f_bin_lin[1].data

# Sort the data by the 'emin' column
sorted_indices = np.argsort(bin_data['emin'])  # Get sorted indices
sorted_data_none = bin_data[sorted_indices]  # Reorder the data using sorted indices
#print(sorted_data_none)

snr3 = bin_data_snr[bin_data_snr['loop_item'] == '3']
sorted_indices_snr3 = np.argsort(snr3['emin'])  # Get sorted indices
sorted_data_snr3 = snr3[sorted_indices_snr3]
#print(sorted_data_snr3)

snr5 = bin_data_snr[bin_data_snr['loop_item'] == '5']
sorted_indices_snr5 = np.argsort(snr5['emin'])  # Get sorted indices
sorted_data_snr5 = snr5[sorted_indices_snr5]
#print(sorted_data_snr5)

snr10 = bin_data_snr[bin_data_snr['loop_item'] == '10']
sorted_indices_snr10 = np.argsort(snr10['emin'])  # Get sorted indices
sorted_data_snr10 = snr10[sorted_indices_snr10]
#print(sorted_data_snr10)

week = bin_data_lin[bin_data_lin['loop_item'] == 'week']
sorted_indices_lin_week = np.argsort(week['emin'])  # Get sorted indices
sorted_data_lin_week = week[sorted_indices_lin_week]
month = bin_data_lin[bin_data_lin['loop_item'] == 'month']
sorted_indices_lin_month = np.argsort(month['emin'])  # Get sorted indices
sorted_data_lin_month = month[sorted_indices_lin_month]

colors_snr = ['blue', 'orange', 'green']
colors_lin = ['purple', 'brown']

#print(sorted_data_snr5['geometric_mean'])
#print( sorted_data_snr5['flux_tot_value'])

datasets = {f"No_Filtering": (sorted_data_none['geometric_mean'], sorted_data_none['flux_tot_value'], sorted_data_none['flux_tot_error'], sorted_data_none['emin'], sorted_data_none['emax'] )}
datasets_snr = {f"snr_3": (sorted_data_snr3['geometric_mean'], sorted_data_snr3['flux_tot_value'], sorted_data_snr3['flux_tot_error'], sorted_data_snr3['emin'], sorted_data_snr3['emax']),
                f"snr_5": (sorted_data_snr5['geometric_mean'], sorted_data_snr5['flux_tot_value'], sorted_data_snr5['flux_tot_error'], sorted_data_snr5['emin'], sorted_data_snr5['emax']),
                f"snr_10": (sorted_data_snr10['geometric_mean'], sorted_data_snr10['flux_tot_value'], sorted_data_snr10['flux_tot_error'], sorted_data_snr10['emin'], sorted_data_snr10['emax'])}
datasets_lin = {f"week": (sorted_data_lin_week['geometric_mean'], sorted_data_lin_week['flux_tot_value'], sorted_data_lin_week['flux_tot_error'], sorted_data_lin_week['emin'], sorted_data_lin_week['emax']),
                f"month": (sorted_data_lin_month['geometric_mean'], sorted_data_lin_month['flux_tot_value'], sorted_data_lin_month['flux_tot_error'], sorted_data_lin_month['emin'], sorted_data_lin_month['emax'])}
#print(source_name)

fig = simple_plot(datasets, datasets_snr, colors_snr, datasets_lin, colors_lin, source_name, png_naming="")
'''
with PdfPages('./fit_results/spetral_points.pdf') as pdf:
    with open(f'Source_ra_dec_specin.txt', 'r') as file:
                    for line in file:
                        parts = line.strip().split()
        
                        # Properly split handling quotes
                        parts = shlex.split(line)

                        # Extract the source name (already properly split)
                        source_name = parts[0]  # No need to strip quotes, shlex handles it

                        ra = float(parts[1])    # Second part: RA
                        dec = float(parts[2])   # Third part: Dec
                        specin = float(parts[3])  # Fourth part: spectral index
                        #beta = float(parts[4])
                        
                        source_name_cleaned = (
                            source_name.replace(" ", "")
                            .replace(".", "dot")
                            .replace("+", "plus")
                            .replace("-", "minus")
                            .replace('"', '')  # Ensure no extra quotes remain
                        )

                        f_bin = fits.open(f'./fit_results/{source_name_cleaned}_fit_data_NONE.fits')
                        f_bin_snr = fits.open(f'./fit_results/{source_name_cleaned}_fit_data_SNR.fits')
                        f_bin_lin = fits.open(f'./fit_results/{source_name_cleaned}_fit_data_LIN.fits')
                        bin_data = f_bin[1].data
                        bin_data_snr = f_bin_snr[1].data
                        bin_data_lin = f_bin_lin[1].data

                        # Sort the data by the 'emin' column
                        sorted_indices = np.argsort(bin_data['emin'])  # Get sorted indices
                        sorted_data_none = bin_data[sorted_indices]  # Reorder the data using sorted indices
                        #print(sorted_data_none)

                        snr3 = bin_data_snr[bin_data_snr['loop_item'] == '3']
                        sorted_indices_snr3 = np.argsort(snr3['emin'])  # Get sorted indices
                        sorted_data_snr3 = snr3[sorted_indices_snr3]
                        #print(sorted_data_snr3)

                        snr5 = bin_data_snr[bin_data_snr['loop_item'] == '5']
                        sorted_indices_snr5 = np.argsort(snr5['emin'])  # Get sorted indices
                        sorted_data_snr5 = snr5[sorted_indices_snr5]
                        #print(sorted_data_snr5)

                        snr10 = bin_data_snr[bin_data_snr['loop_item'] == '10']
                        sorted_indices_snr10 = np.argsort(snr10['emin'])  # Get sorted indices
                        sorted_data_snr10 = snr10[sorted_indices_snr10]
                        #print(sorted_data_snr10)

                        week = bin_data_lin[bin_data_lin['loop_item'] == 'week']
                        sorted_indices_lin_week = np.argsort(week['emin'])  # Get sorted indices
                        sorted_data_lin_week = week[sorted_indices_lin_week]
                        month = bin_data_lin[bin_data_lin['loop_item'] == 'month']
                        sorted_indices_lin_month = np.argsort(month['emin'])  # Get sorted indices
                        sorted_data_lin_month = month[sorted_indices_lin_month]

                        colors_snr = ['blue', 'orange', 'green']
                        colors_lin = ['purple', 'brown']
                        
                        #print(sorted_data_snr5['geometric_mean'])
                        #print( sorted_data_snr5['flux_tot_value'])

                        datasets = {f"No_Filtering": (sorted_data_none['geometric_mean'], sorted_data_none['flux_tot_value'], sorted_data_none['flux_tot_error'], sorted_data_none['emin'], sorted_data_none['emax'] )}
                        datasets_snr = {f"snr_3": (sorted_data_snr3['geometric_mean'], sorted_data_snr3['flux_tot_value'], sorted_data_snr3['flux_tot_error'], sorted_data_snr3['emin'], sorted_data_snr3['emax']),
                                        f"snr_5": (sorted_data_snr5['geometric_mean'], sorted_data_snr5['flux_tot_value'], sorted_data_snr5['flux_tot_error'], sorted_data_snr5['emin'], sorted_data_snr5['emax']),
                                        f"snr_10": (sorted_data_snr10['geometric_mean'], sorted_data_snr10['flux_tot_value'], sorted_data_snr10['flux_tot_error'], sorted_data_snr10['emin'], sorted_data_snr10['emax'])}
                        datasets_lin = {f"week": (sorted_data_lin_week['geometric_mean'], sorted_data_lin_week['flux_tot_value'], sorted_data_lin_week['flux_tot_error'], sorted_data_lin_week['emin'], sorted_data_lin_week['emax']),
                                        f"month": (sorted_data_lin_month['geometric_mean'], sorted_data_lin_month['flux_tot_value'], sorted_data_lin_month['flux_tot_error'], sorted_data_lin_month['emin'], sorted_data_lin_month['emax'])}
                        #print(source_name)
                       
                        fig = simple_plot(datasets, datasets_snr, colors_snr, datasets_lin, colors_lin, source_name, png_naming="")
                        pdf.savefig(fig)
                        plt.close(fig)

'''  
                for when I need example walk throug of NGC 1275
                    print()
                    print(f"CURVE FIT for {source_name}")
                    results = fit_data("iminuit", datasets, png_naming =f"_{source_name_cleaned}")
                    results_snr = fit_data("iminuit", datasets_snr, png_naming =f"_{source_name_cleaned}")
                    results_lin = fit_data("iminuit", datasets_lin, png_naming =f"_{source_name_cleaned}")
                    
                    plot_fits_with_residuals(datasets, results, e_lowers, e_uppers, png_naming =f"_{source_name_cleaned}")
                    plot_fits_with_residuals(datasets_snr, results_snr, e_lowers, e_uppers, png_naming =f"_{source_name_cleaned}")
                    plot_fits_with_residuals(datasets_lin, results_lin, e_lowers, e_uppers, png_naming =f"_{source_name_cleaned}")
'''