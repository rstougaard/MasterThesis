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

path_to_save_heatmap_m_g = "./fit_results/heatmaps_m_g/"
path_to_save_heatmap_Ec_p0 = "./fit_results/heatmaps_Ec_p0/"

#p0_values = np.linspace(0, 1 / 3, 11)
#E_c_values = np.logspace(1, 4, 61)  # MeV
#k = 2.7
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
g_start_val = 7e-13
m_stop_val  = 1.5e-8
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

    
# Function 1: LogPar
def logpar_base(x, Norm, alpha_, beta_):
    E_b = 1000  # Fixed E_b value
    return Norm * (x / E_b) ** (-(alpha_ + beta_ * np.log(x / E_b)))

def cutoff_base(x, Norm, l1, l2):
    E_b = 1000
    return np.piecewise(x, [x < E_b, x >= E_b], [lambda x: Norm*(x/E_b)**l1, lambda x: Norm*(x/E_b)**l2])

def reduced_chi_square(y_obs, y_fit, y_err, num_params):
    residuals = (y_obs - y_fit) / y_err
    chi2 = np.sum(residuals**2)
    dof = len(y_obs) - num_params  # Degrees of freedom
    return chi2 , dof

def fit_data(x, y, y_err, emin, emax, p0, E_c, k, source_name, dataset_label, useEBL=True, fitting_method="iminuit", basefunc = "cutoff"):
    # Filter out points where y is zero
    mask = y != 0
    x_filtered, y_filtered, y_err_filtered, emin_f, emax_f  = x[mask], y[mask], y_err[mask], emin[mask], emax[mask]

    e_lowers = x_filtered - emin_f
    e_uppers = emax_f - x_filtered
    
    # Check if the last point was filtered out
    if not mask[-1]:  # If the last point was removed (y[-1] == 0)
        y_err_eff = y_err_filtered + 0.03 * y_filtered  # Only add 3% to errors
    else:  # Otherwise, handle the normal case
        y_err_eff0 = y_err_filtered[:-1] + 0.03 * y_filtered[:-1]
        y_err_eff1 = y_err_filtered[-1] + 0.10 * y_filtered[-1]
        y_err_eff = np.append(y_err_eff0, y_err_eff1)

    y_err_eff = np.array(y_err_eff)

    if useEBL and basefunc == "logpar":
        with fits.open('table-4LAC-DR3-h.fits') as f:
            data1 = f[1].data
        idx = (data1['Source_Name'] == source_name)
        z = data1['Redshift'][idx][0]
        ebl = EblAbsorptionModel(z).transmission(x_filtered * u.MeV)
        
        def LogPar_EBL(x, Norm, alpha_, beta_):
            return logpar_base(x, Norm, alpha_, beta_) * ebl

        base = LogPar_EBL
        bounds_base = ([1e-12, -1.0, -1.0], [1e-7, 5.0, 3.0])  # Lower and upper bounds
        p0_base = [1e-9, 2.0, 0.1]  # Initial guesses

        def axion_func(E, Norm, alpha_, beta_, w):
            return base(E, Norm, alpha_, beta_) * (1 - p0 / (1 + (E_c / E) ** k) * (1+0.2*np.tanh(w)))
        
        bounds_alp = ([1e-12, -1.0, -1.0, -np.pi], [1e-7, 5.0, 3.0, np.pi])  # Lower and upper bounds
        p0_alp= [1e-9, 2.0, 0.1, np.pi/2]

    elif useEBL and basefunc == "cutoff":
        with fits.open('table-4LAC-DR3-h.fits') as f:
            data1 = f[1].data
        idx = (data1['Source_Name'] == source_name)
        z = data1['Redshift'][idx][0]
        ebl = EblAbsorptionModel(z).transmission(x_filtered * u.MeV)

        def cutoff_EBL(x, Norm,  l1, l2):
            return cutoff_base(x, Norm, l1, l2) * ebl

        base = cutoff_EBL

        bounds_base = ([1e-12, -5.0, -5.0], [1e-7, 5.0, 5.0])  # Lower and upper bounds
        p0_base = [1e-9, 2.0, 2.0]  # Initial guesses

        def axion_func(E, Norm, l1, l2, w):
            return base(E, Norm, l1, l2) * (1 - p0 / (1 + (E_c / E) ** k) * (1+0.2*np.tanh(w)))
        
        bounds_alp = ([1e-12, -5.0, -5.0, -np.pi], [1e-7, 5.0, 5.0, np.pi])  # Lower and upper bounds
        p0_alp= [1e-9, 2.0, 2.0, np.pi/2]

    if fitting_method == "curve_fit":
        
        popt_base, pcov_base = curve_fit(
            base, x_filtered, y_filtered, sigma=y_err_eff, p0=p0_base, bounds=bounds_base, absolute_sigma=True, maxfev=100000)
        y_fit_base = base(x_filtered, *popt_base)
        chi2_base, dof_base = reduced_chi_square(y_filtered, y_fit_base, y_err_eff, len(popt_base))

        # Extract parameter uncertainties
        perr_base = np.sqrt(np.diag(pcov_base))

        popt_axion, pcov_axion = curve_fit(
            axion_func, x_filtered, y_filtered, sigma=y_err_eff, p0=p0_alp, bounds=bounds_alp, absolute_sigma=True
        )
        y_fit_axion = axion_func(x_filtered, *popt_axion)
        chi2_axion, dof_axion = reduced_chi_square(y_filtered, y_fit_axion, y_err_eff, len(popt_axion))

        # Extract parameter uncertainties
        perr_axion = np.sqrt(np.diag(pcov_axion))

        # Compute Δχ²
        delta_chi2 = chi2_axion - chi2_base
    '''
    print("LogPar:\n")
    for param, value, error in zip(["Norm", "alpha_", "beta_"], popt_logpar, perr_logpar):
        print(f"  {param}: {value:.2e} ± {error:.2e}\n")
    print(f'χ² / dof: {chi2_logpar:.2f} / {dof_logpar}\n\n')
    print("Axion:\n")
    for param, value, error in zip(["Norm", "alpha_", "beta_", "w"], popt_axion, perr_axion):
        print(f"  {param}: {value:.2e} ± {error:.2e}\n")  
    print( f"χ² / dof: {chi2_axion:.2f} / {dof_axion}\n\n")

    print(f"Δχ² (Axion - LogPar): {delta_chi2:.2f}")
    print('--------------------------------------------------------------------------------------')
    '''
    # Return fit results
    return {
        "Base": {
            "params": popt_base,
            "errors": perr_base,
            "chi2": chi2_base,
            "dof": dof_base
        },
        "Axion": {
            "params": popt_axion,
            "errors": perr_axion, # Return covariance matrix
            "chi2": chi2_axion,
            "dof": dof_axion,
        },
        "DeltaChi2": delta_chi2,
        "y_fit_LogPar": y_fit_base,
        "y_fit_Axion": y_fit_axion,
    }

def nested_fits(datasets, source_name, useEBL=True, fitting_method="iminuit"):
    results = {}
    # Here we assume p0_masked and ec_masked are defined globally (or accessible in this scope)
    # and have the same shape (n_mass_masked, n_g_masked) corresponding to your m, g grid.
    for dataset_label, (x, y, y_err, emin, emax) in datasets.items():
        dataset_results = []
        # Loop over the mass dimension (rows)
        for i in range(p0_masked.shape[0]):
            row_results = []
            # Loop over the g dimension (columns)
            for j in range(p0_masked.shape[1]):
                p0_val = p0_masked[i, j]
                ec_val = ec_masked[i, j]
                # Perform the fit for this (m, g) pair
                fit_result = fit_data(
                    x=np.array(x),
                    y=np.array(y),
                    y_err=np.array(y_err),
                    emin=np.array(emin),
                    emax=np.array(emax),
                    p0=p0_val,
                    E_c=ec_val,
                    k=k,  # Ensure k is defined in your scope
                    source_name=source_name,
                    dataset_label=dataset_label,
                    useEBL=useEBL,
                    fitting_method = fitting_method
                )
                row_results.append({
                    "p0": p0_val,
                    "E_c": ec_val,
                    "fit_result": fit_result
                })
                
        
            # Append the row (corresponding to a mass value) to the dataset results
            dataset_results.append(row_results)
        results[dataset_label] = dataset_results
    return results


def plot_delta_chi2_heatmap(results, dataset_label, png_naming):

    # Extract results for the specified dataset
    if dataset_label not in results:
        raise ValueError(f"Dataset label '{dataset_label}' not found in results.")

    dataset_results = results[dataset_label]

    # Initialize a 2D array to store DeltaChi2 values
    delta_chi2_grid = np.zeros((len(p0_all), len(ec_all)))

    # Fill the grid
    for result in dataset_results:
        p0 = result["p0"]
        E_c = result["E_c"]
        DeltaChi2 = result["fit_result"]["DeltaChi2"]

        # Find indices
        p0_index = p0_all.tolist().index(p0)  # Index of p0
        E_c_index = ec_all.tolist().index(E_c)  # Index of E_c

        # Store DeltaChi2 value
        delta_chi2_grid[p0_index, E_c_index] = DeltaChi2

    # Find the indices of the minimum DeltaChi2
    min_idx = np.unravel_index(np.argmin(delta_chi2_grid), delta_chi2_grid.shape)
    best_p0 = p0_all[min_idx[0]]
    best_Ec = ec_all[min_idx[1]]

    print(f"Best fit parameters: p0 = {best_p0:.2f}, Ec = {best_Ec:.2f}")

    vmin, vmax = -10, 25  # Color limits
    num_colors = 30  # More steps for finer detail
    boundaries = np.linspace(vmin, vmax, num_colors + 1)  # More steps

    # Use reversed colormap to make low values bright
    cmap = plt.get_cmap('gnuplot2_r', num_colors)  # '_r' reverses colors
    norm = mcolors.BoundaryNorm(boundaries=boundaries, ncolors=num_colors, clip=True)

    # Create the heatmap
    plt.figure(figsize=(10, 6))
    E_c_mesh, p0_mesh = np.meshgrid(ec_all, p0_all)
    heatmap = plt.pcolormesh(E_c_mesh, p0_mesh, delta_chi2_grid, shading='auto', cmap=cmap, norm=norm)

    # Add colorbar with fixed range
    cbar = plt.colorbar(heatmap, label=r'$\Delta \chi^2$', ticks=np.linspace(vmin, vmax, 11))
    cbar.ax.tick_params(labelsize=15)  
    cbar.set_label(r'$\Delta \chi^2$', fontsize=15)

    # Plot the best-fit point as a cross
    plt.plot(best_Ec, best_p0, marker='+', color='red', markersize=15, mew=2,
            label=f'Best Fit ($E_c$, p0) = ({best_Ec:.2f}, {best_p0:.2f})')

    # Add labels and title
    plt.xlabel(r'$E_c$ [MeV]', fontsize=15)
    plt.ylabel('p0', fontsize=15)
    plt.title(f'Heatmap of $\Delta \chi^2$ for {dataset_label}', fontsize=15)

    # Set x-axis to logarithmic scale
    plt.xscale('log')
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)

    # Add legend for the best-fit marker
    plt.legend(fontsize=12, loc='lower left')

    # Tight layout and save the plot
    plt.tight_layout()
    plt.savefig(f'{path_to_save_heatmap_Ec_p0}{png_naming}_{dataset_label}.png', dpi=300)
    plt.close()


def plot_mean_delta_chi2_heatmap(all_results, dataset_labels, png_naming):
    """Generates mean Δχ² heatmaps separately for each filtering category in both 
       (E_c, p0) space and (mₐ, gₐ) space.
       
       Assumes that:
         - p0_all and ec_all are defined globally (or available in scope) for the first plot.
         - ma_all and g_all (with ma_all = axion_data[:,0] in eV and g_all = axion_data[:,1] in GeV⁻¹)
           are defined globally (or available in scope) for the second plot.
    """
    
    # Identify filtering categories (e.g., "week", "month", etc.)
    first_source = next(iter(all_results.values()))
    filtering_methods = list(first_source.keys())

    for filter_label in filtering_methods:
        # Initialize sum and count grids for (E_c, p0) parameter space
        sum_grid = np.zeros((len(p0_all), len(ec_all)))
        count_grid = np.zeros((len(p0_all), len(ec_all)))

        # Loop over all sources and accumulate Δχ² values for this filter type
        for source_name in dataset_labels:
            if source_name not in all_results or filter_label not in all_results[source_name]:
                print(f"Warning: Missing data for {filter_label} in {source_name}. Skipping.")
                continue

            dataset_results = all_results[source_name][filter_label]
            delta_chi2_grid = np.zeros((len(p0_all), len(ec_all)))

            for result in dataset_results:
                p0 = result["p0"]
                E_c = result["E_c"]
                DeltaChi2 = result["fit_result"]["DeltaChi2"]

                # Find the indices in the parameter grids
                p0_index = p0_all.tolist().index(p0)
                E_c_index = ec_all.tolist().index(E_c)

                # Store the Δχ² value at the corresponding grid cell
                delta_chi2_grid[p0_index, E_c_index] = DeltaChi2

            # Update sum and count grids using only nonzero entries
            valid_mask = delta_chi2_grid != 0
            sum_grid += delta_chi2_grid
            count_grid += valid_mask

        # Compute the mean Δχ², taking care to avoid division by zero
        mean_delta_chi2_grid = np.divide(sum_grid, count_grid, where=(count_grid != 0))

        # Set colormap properties (common to both plots)
        vmin, vmax = -10, 25
        num_colors = 30
        boundaries = np.linspace(vmin, vmax, num_colors + 1)
        cmap = plt.get_cmap('gnuplot2_r', num_colors)
        norm = mcolors.BoundaryNorm(boundaries=boundaries, ncolors=num_colors, clip=True)

        ### Plot 1: (E_c, p0) heatmap ###
        plt.figure(figsize=(10, 6))
        # Here we assume that the x-axis should be E_c and y-axis p0.
        # Adjust the order in meshgrid if needed.
        E_c_mesh, p0_mesh = np.meshgrid(ec_all, p0_all)  # x=E_c, y=p0
        heatmap = plt.pcolormesh(E_c_mesh, p0_mesh, mean_delta_chi2_grid, shading='auto',
                                 cmap=cmap, norm=norm)

        # Add contour for Δχ² ≤ -6.2 if any
        if np.any(mean_delta_chi2_grid <= -6.2):
            contour_levels = [-6.2]
            plt.contour(E_c_mesh, p0_mesh, mean_delta_chi2_grid, levels=contour_levels,
                        colors='red', linewidths=2)

        cbar = plt.colorbar(heatmap, label=r'$\langle \Delta \chi^2 \rangle$', 
                             ticks=np.linspace(vmin, vmax, 11))
        cbar.ax.tick_params(labelsize=15)
        cbar.set_label(r'$\langle \Delta \chi^2 \rangle$', fontsize=15)
        plt.xlabel(r'$E_c$ [MeV]', fontsize=15)
        plt.ylabel('p0', fontsize=15)
        plt.ylim(0.0, 1/3)
        plt.title(f'Mean $\Delta \chi^2$ Heatmap for {filter_label} in (E_c, p0) Space', fontsize=15)
        plt.xscale('log')
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.tight_layout()
        plt.savefig(f'{path_to_save_heatmap_Ec_p0}{png_naming}_{filter_label}_ec_p0.png', dpi=300)
        plt.close()

        ### Plot 2: (mₐ, gₐ₍γ₎) heatmap ###
        plt.figure(figsize=(10, 6))
        # Create a meshgrid for mₐ and gₐ.
        # Here we assume that the grid dimensions for (mₐ, gₐ) are the same as for (E_c, p0).
        ma_mesh, g_mesh = np.meshgrid(ma_all, g_all)  # x=mₐ, y=gₐ
        heatmap = plt.pcolormesh(ma_mesh, g_mesh, mean_delta_chi2_grid, shading='auto',
                                 cmap=cmap, norm=norm)

        if np.any(mean_delta_chi2_grid <= -6.2):
            contour_levels = [-6.2]
            plt.contour(ma_mesh, g_mesh, mean_delta_chi2_grid, levels=contour_levels,
                        colors='red', linewidths=2)

        cbar = plt.colorbar(heatmap, label=r'$\langle \Delta \chi^2 \rangle$', 
                             ticks=np.linspace(vmin, vmax, 11))
        cbar.ax.tick_params(labelsize=15)
        cbar.set_label(r'$\langle \Delta \chi^2 \rangle$', fontsize=15)
        plt.xlabel(r'$m_a$ [eV]', fontsize=15)
        plt.ylabel(r'$g_{a\gamma}$ [GeV$^{-1}$]', fontsize=15)
        plt.title(f'Mean $\Delta \chi^2$ Heatmap for {filter_label} in ($m_a$, $g_{{a\gamma}}$) Space', fontsize=15)
        plt.xscale('log')  # Remove or adjust if not appropriate for mₐ
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.tight_layout()
        plt.savefig(f'{path_to_save_heatmap_m_g}{png_naming}_{filter_label}_ma_ga.png', dpi=300)
        plt.close()

def compute_mean_delta_chi2_grid(all_results, dataset_labels, filter_label,
                                 p0_masked, ec_masked):
    """
    all_results: e.g. all_results_none or all_results_snr
    dataset_labels: list of keys in that dictionary
    filter_label: which key to look for in all_results[source][filter_label]
    p0_masked, ec_masked: your global grids
    Returns: mean Δχ² grid
    """
    sum_grid = np.zeros((ec_masked.shape[0], ec_masked.shape[1]))
    count_grid = np.zeros((ec_masked.shape[0], ec_masked.shape[1]))
    
    for source_name in dataset_labels:
        if (source_name not in all_results) or (filter_label not in all_results[source_name]):
            continue
        dataset_results = all_results[source_name][filter_label]
        for row in dataset_results:  # each row is a list of dicts
            for result in row:
                p0_val = result["p0"]
                ec_val = result["E_c"]
                DeltaChi2 = result["fit_result"]["DeltaChi2"]
                
                matches = np.where(
                    np.isclose(p0_masked, p0_val) &
                    np.isclose(ec_masked, ec_val)
                )
                if matches[0].size == 1:
                    i = matches[0][0]
                    j = matches[1][0]
                    sum_grid[i, j] += DeltaChi2
                    count_grid[i, j] += 1
    
    with np.errstate(divide='ignore', invalid='ignore'):
        mean_delta_chi2_grid = np.where(count_grid != 0,
                                        sum_grid / count_grid,
                                        np.nan)
    print("Minimum mean Δχ²:", np.nanmin(mean_delta_chi2_grid))
    print("Maximum mean Δχ²:", np.nanmax(mean_delta_chi2_grid))
    return mean_delta_chi2_grid


def plot_mean_delta_chi2_heatmap3(all_results, dataset_labels, png_naming, no_filtering_grid=None):
    """
    Generates mean Δχ² heatmaps for each filtering category in two spaces:
      (E_c, p₀) and (mₐ, gₐ).
    
    Assumes that the data were generated in a nested loop over masses and g-values:
      - ec_all_full and p0_all_full (reshaped from axion_data) have shape (n_mass, n_g).
      - g_unique comes from the first n_g rows and mass_unique from every n_g-th row.
    
    Uses tolerance-based matching (via np.isclose) to assign each result (with a pair [p₀, E_c])
    into its proper grid cell.
    """
    global axion_data, ec_masked, p0_masked, m_masked, g_masked, n_mass, n_g
    
     # Extract all filter labels from the first source
    first_source = next(iter(all_results.values()))
    filtering_methods = list(first_source.keys())  # e.g. ["snr_3", "snr_5", "snr_10"] or ["week", "month"]
    
    # Build the (m_a, g_a) mesh
    ma_mesh, g_mesh = np.meshgrid(m_masked, g_masked, indexing='ij')
    
    for filter_label in filtering_methods:
        # -- 1) Compute mean Δχ² for this filter_label --
        mean_delta_chi2_grid = compute_mean_delta_chi2_grid(
            all_results=all_results,
            dataset_labels=dataset_labels,
            filter_label=filter_label,
            p0_masked=p0_masked,
            ec_masked=ec_masked
        )

        print(np.min(mean_delta_chi2_grid))
        print(np.max(mean_delta_chi2_grid))
        # Set up colormap.
        vmin, vmax = -10, 25
        num_colors = 30
        boundaries = np.linspace(vmin, vmax, num_colors + 1)
        cmap = plt.get_cmap('gnuplot2', num_colors)
        norm = mcolors.BoundaryNorm(boundaries=boundaries, ncolors=num_colors, clip=True)

        ma_mesh, g_mesh = np.meshgrid(m_masked, g_masked, indexing='ij')

        plt.figure(figsize=(10, 6))
        heatmap = plt.pcolormesh(ma_mesh / 1e-9, g_mesh, mean_delta_chi2_grid,
                                 cmap=cmap, norm=norm, shading='auto')
        
        # Overlay No_Filtering contour if provided and if we aren't plotting No_Filtering itself.
        if (no_filtering_grid is not None) and (filter_label != "No_Filtering"):
            if np.any(no_filtering_grid <= -6.2):
                plt.contour(ma_mesh / 1e-9, g_mesh, no_filtering_grid,
                            levels=[-6.2], colors='#f16913', linewidths=2)
        
        if np.any(mean_delta_chi2_grid <= -6.2):
            plt.contour(ma_mesh / 1e-9, g_mesh, mean_delta_chi2_grid,
                        levels=[-6.2], colors='#78c679', linewidths=2)
        
        if np.any(mean_delta_chi2_grid >= 6.2):
            plt.contour(ma_mesh / 1e-9, g_mesh, mean_delta_chi2_grid,
                        levels=[6.2], colors='red', linewidths=2)
                
        cbar = plt.colorbar(heatmap, ticks=np.linspace(vmin, vmax, 11))
        cbar.set_label(r'$\langle \Delta \chi^2 \rangle$', fontsize=15)
        plt.xlabel(r'$m_a$ [neV]', fontsize=15)
        plt.ylabel(r'$g_{a\gamma}$ [GeV$^{-1}$]', fontsize=15)
        plt.title(f'Mean $\Delta \chi^2$ Heatmap for {filter_label} in ($m_a$, $g_{{a\gamma}}$) Space', fontsize=15)
        plt.xscale('log')
        plt.yscale('log')
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        #plt.xlim(start_mass/1e-9, stop_mass/1e-9)
        #plt.ylim(start_g, stop_g)
        plt.tight_layout()
        plt.savefig(f'{path_to_save_heatmap_m_g}{png_naming}_{filter_label}_ma_ga.png', dpi=300)
        plt.close()
              
        print(f"Finished plotting for filter: {filter_label}")


'''
source_name = "4FGL J0319.8+4130"  # No need to strip quotes, shlex handles it

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

snr3 = bin_data_snr[bin_data_snr['loop_item'] == '3']
sorted_indices_snr3 = np.argsort(snr3['emin'])  # Get sorted indices
sorted_data_snr3 = snr3[sorted_indices_snr3]

snr5 = bin_data_snr[bin_data_snr['loop_item'] == '5']
sorted_indices_snr5 = np.argsort(snr5['emin'])  # Get sorted indices
sorted_data_snr5 = snr5[sorted_indices_snr5]

snr10 = bin_data_snr[bin_data_snr['loop_item'] == '10']
sorted_indices_snr10 = np.argsort(snr10['emin'])  # Get sorted indices
sorted_data_snr10 = snr10[sorted_indices_snr10]

week = bin_data_lin[bin_data_lin['loop_item'] == 'week']
sorted_indices_lin_week = np.argsort(week['emin'])  # Get sorted indices
sorted_data_lin_week = week[sorted_indices_lin_week]
month = bin_data_lin[bin_data_lin['loop_item'] == 'month']
sorted_indices_lin_month = np.argsort(month['emin'])  # Get sorted indices
sorted_data_lin_month = month[sorted_indices_lin_month]

colors_snr = ['blue', 'orange', 'green']
colors_lin = ['purple', 'brown']
bin_size = np.array(sorted_data_none['emax'])-np.array(sorted_data_none['emin'])
e_lowers = sorted_data_none['geometric_mean'] - sorted_data_none['emin']
e_uppers = sorted_data_none['emax'] - sorted_data_none['geometric_mean']
datasets = {f"No_Filtering": (sorted_data_none['geometric_mean'], sorted_data_none['flux_tot_value']/bin_size, sorted_data_none['flux_tot_error']/bin_size, sorted_data_none['emin'], sorted_data_none['emax'] )}
datasets_snr = {f"snr_3": (sorted_data_snr3['geometric_mean'], sorted_data_snr3['flux_tot_value']/bin_size, sorted_data_snr3['flux_tot_error']/bin_size, sorted_data_snr3['emin'], sorted_data_snr3['emax'] ),
                f"snr_5": (sorted_data_snr5['geometric_mean'], sorted_data_snr5['flux_tot_value']/bin_size, sorted_data_snr5['flux_tot_error']/bin_size, sorted_data_snr5['emin'], sorted_data_snr5['emax'] ),
                f"snr_10": (sorted_data_snr10['geometric_mean'], sorted_data_snr10['flux_tot_value']/bin_size, sorted_data_snr10['flux_tot_error']/bin_size, sorted_data_snr10['emin'], sorted_data_snr10['emax'] )}
datasets_lin = {f"week": (sorted_data_lin_week['geometric_mean'], sorted_data_lin_week['flux_tot_value']/bin_size, sorted_data_lin_week['flux_tot_error']/bin_size, sorted_data_lin_week['emin'], sorted_data_lin_week['emax'] ),
                f"month": (sorted_data_lin_month['geometric_mean'], sorted_data_lin_month['flux_tot_value']/bin_size, sorted_data_lin_month['flux_tot_error']/bin_size, sorted_data_lin_month['emin'], sorted_data_lin_month['emax'] )}
print(source_name)

results = nested_fits(datasets, source_name, useEBL=True)
results_snr = nested_fits(datasets_snr, source_name, useEBL=True)
results_lin= nested_fits(datasets_lin, source_name, useEBL=True)
'''
def process_chunk(i, j_start, j_end, x, y, y_err, emin, emax, source_name, dataset_label, useEBL, fitting_method, basefunc):
    # For row i, get a chunk of p0 and Ec values from the masked grid.
    p0_chunk = p0_masked[i, j_start:j_end]
    ec_chunk = ec_masked[i, j_start:j_end]
    
    # Vectorize the fit call over the chunk.
    # (If possible, replace np.vectorize with a truly vectorized version of fit_data.)
    vectorized_fit = np.vectorize(
        lambda p0_val, ec_val: fit_data(
            x=np.array(x),
            y=np.array(y),
            y_err=np.array(y_err),
            emin=np.array(emin),
            emax=np.array(emax),
            p0=p0_val,
            E_c=ec_val,
            k=k,  # Ensure k is defined in the scope
            source_name=source_name,
            dataset_label=dataset_label,
            useEBL=useEBL,
            fitting_method=fitting_method,
            basefunc = basefunc
        ),
        otypes=[object]  # each fit_data returns a dict (or any object)
    )
    # Apply the vectorized function to the chunk arrays.
    results_chunk = vectorized_fit(p0_chunk, ec_chunk)
    
    # Format the output as a list of dictionaries.
    return [{
        "p0": p0_chunk[idx],
        "E_c": ec_chunk[idx],
        "fit_result": results_chunk[idx]
    } for idx in range(len(p0_chunk))]

def nested_fits_combined(datasets, source_name, useEBL=True, fitting_method="iminuit", basefunc = "cutoff", chunk_size=10):
    """
    This function processes each row (mass) of the masked (p0, Ec) grid.
    Each row is split into chunks (to minimize overhead) and processed in parallel.
    Within each chunk, np.vectorize is used to apply the fitting function over the array of p0 and Ec.
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
                    np.array(emin), np.array(emax),
                    source_name, dataset_label, useEBL, fitting_method, basefunc
                ))
            # Execute the tasks in parallel for row i.
            row_results = []
            chunk_results = Parallel(n_jobs=-1)(tasks)
            # Flatten the results for row i.
            for chunk in chunk_results:
                row_results.extend(chunk)
            dataset_results.append(row_results)
        results[dataset_label] = dataset_results
    return results

all_results_none = {}
all_results_snr = {}
all_results_lin = {}
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

                    snr3 = bin_data_snr[bin_data_snr['loop_item'] == '3']
                    sorted_indices_snr3 = np.argsort(snr3['emin'])  # Get sorted indices
                    sorted_data_snr3 = snr3[sorted_indices_snr3]

                    snr5 = bin_data_snr[bin_data_snr['loop_item'] == '5']
                    sorted_indices_snr5 = np.argsort(snr5['emin'])  # Get sorted indices
                    sorted_data_snr5 = snr5[sorted_indices_snr5]

                    snr10 = bin_data_snr[bin_data_snr['loop_item'] == '10']
                    sorted_indices_snr10 = np.argsort(snr10['emin'])  # Get sorted indices
                    sorted_data_snr10 = snr10[sorted_indices_snr10]

                    week = bin_data_lin[bin_data_lin['loop_item'] == 'week']
                    sorted_indices_lin_week = np.argsort(week['emin'])  # Get sorted indices
                    sorted_data_lin_week = week[sorted_indices_lin_week]
                    month = bin_data_lin[bin_data_lin['loop_item'] == 'month']
                    sorted_indices_lin_month = np.argsort(month['emin'])  # Get sorted indices
                    sorted_data_lin_month = month[sorted_indices_lin_month]

                    colors_snr = ['blue', 'orange', 'green']
                    colors_lin = ['purple', 'brown']
                    bin_size = np.array(sorted_data_none['emax'])-np.array(sorted_data_none['emin'])
                    e_lowers = sorted_data_none['geometric_mean'] - sorted_data_none['emin']
                    e_uppers = sorted_data_none['emax'] - sorted_data_none['geometric_mean']
                    datasets = {f"No_Filtering": (sorted_data_none['geometric_mean'], sorted_data_none['flux_tot_value'], sorted_data_none['flux_tot_error'], sorted_data_none['emin'], sorted_data_none['emax'] )}
                    datasets_snr = {f"snr_3": (sorted_data_snr3['geometric_mean'], sorted_data_snr3['flux_tot_value'], sorted_data_snr3['flux_tot_error'], sorted_data_snr3['emin'], sorted_data_snr3['emax'] ),
                                    f"snr_5": (sorted_data_snr5['geometric_mean'], sorted_data_snr5['flux_tot_value'], sorted_data_snr5['flux_tot_error'], sorted_data_snr5['emin'], sorted_data_snr5['emax'] ),
                                    f"snr_10": (sorted_data_snr10['geometric_mean'], sorted_data_snr10['flux_tot_value'], sorted_data_snr10['flux_tot_error'], sorted_data_snr10['emin'], sorted_data_snr10['emax'] )}
                    datasets_lin = {f"week": (sorted_data_lin_week['geometric_mean'], sorted_data_lin_week['flux_tot_value'], sorted_data_lin_week['flux_tot_error'], sorted_data_lin_week['emin'], sorted_data_lin_week['emax'] ),
                                    f"month": (sorted_data_lin_month['geometric_mean'], sorted_data_lin_month['flux_tot_value'], sorted_data_lin_month['flux_tot_error'], sorted_data_lin_month['emin'], sorted_data_lin_month['emax'] )}
                    print(source_name)

                    results = nested_fits_combined(datasets, source_name, useEBL=True, fitting_method="curve_fit", basefunc = "cutoff", chunk_size=10)
                    results_snr = nested_fits_combined(datasets_snr, source_name, useEBL=True, fitting_method="curve_fit", basefunc = "cutoff", chunk_size=10)
                    results_lin= nested_fits_combined(datasets_lin, source_name, useEBL=True, fitting_method="curve_fit", basefunc = "cutoff", chunk_size=10)
                    

                    all_results_none[source_name] = results
                    with open("all_results_none_32_curve_fit_nodivbin_cutoff.pkl", "wb") as file:
                        pickle.dump(all_results_none, file)
                      
                    all_results_snr[source_name] = results_snr
                    with open("all_results_snr_32_curve_fit_nodivbin_cutoff.pkl", "wb") as file:
                        pickle.dump(all_results_snr, file)

                    all_results_lin[source_name] = results_lin
                    with open("all_results_lin_32_curve_fit_nodivbin_cutoff.pkl", "wb") as file:
                        pickle.dump(all_results_lin, file)

                    ''' 
                    plot_delta_chi2_heatmap(results, dataset_label="No_Filtering", png_naming =f"{source_name_cleaned}")
                    
                    plot_delta_chi2_heatmap(results_snr, dataset_label="snr_3", png_naming =f"{source_name_cleaned}")
                    plot_delta_chi2_heatmap(results_snr, dataset_label="snr_5", png_naming =f"{source_name_cleaned}")
                    plot_delta_chi2_heatmap(results_snr, dataset_label="snr_10", png_naming =f"{source_name_cleaned}")
                    
                    plot_delta_chi2_heatmap(results_lin, dataset_label="week", png_naming =f"{source_name_cleaned}")
                    plot_delta_chi2_heatmap(results_lin, dataset_label="month", png_naming =f"{source_name_cleaned}")
                    print("(p0, Ec) Heatmaps done!")
                    '''
'''
print('Plotting mean chi-squared heatmap!')
no_filtering_sources = list(all_results_none.keys())  # e.g. ["No_Filtering"] or sometimes multiple sources

no_filtering_grid = compute_mean_delta_chi2_grid(
    all_results=all_results_none,
    dataset_labels=no_filtering_sources,
    filter_label="No_Filtering",
    p0_masked=p0_masked,
    ec_masked=ec_masked
) 

plot_mean_delta_chi2_heatmap3(all_results_none, list(all_results_none.keys()), "mean")
# For LIN filtering ("week" and "month")
plot_mean_delta_chi2_heatmap3(all_results_lin, list(all_results_lin.keys()), "mean_", no_filtering_grid=no_filtering_grid)

# For SNR filtering ("snr_3", "snr_5", "snr_10")
plot_mean_delta_chi2_heatmap3(all_results_snr, list(all_results_snr.keys()), "mean_", no_filtering_grid=no_filtering_grid)



plot_delta_chi2_heatmap_m_g(results, dataset_label="No_Filtering")
plot_delta_chi2_heatmap_m_g(results_snr, dataset_label="snr_3")
plot_delta_chi2_heatmap_m_g(results_snr, dataset_label="snr_5")
plot_delta_chi2_heatmap_m_g(results_snr, dataset_label="snr_10")
plot_delta_chi2_heatmap_m_g(results_lin, dataset_label="week")
plot_delta_chi2_heatmap_m_g(results_lin, dataset_label="month")

'''