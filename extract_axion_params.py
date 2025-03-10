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
g_start_val = 5e-13
m_stop_val  = 2e-8
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

def reduced_chi_square(y_obs, y_fit, y_err, num_params):
    residuals = (y_obs - y_fit) / y_err
    chi2 = np.sum(residuals**2)
    dof = len(y_obs) - num_params  # Degrees of freedom
    return chi2 , dof

def fit_data(x, y, y_err, p0, E_c, k, source_name, useEBL=True):
    # Filter out points where y is zero
    mask = y != 0
    x_filtered, y_filtered, y_err_filtered = x[mask], y[mask], y_err[mask]
    # Check if the last point was filtered out
    if not mask[-1]:  # If the last point was removed (y[-1] == 0)
        y_err_eff = y_err_filtered + 0.03 * y_filtered  # Only add 3% to errors
    else:  # Otherwise, handle the normal case
        y_err_eff0 = y_err_filtered[:-1] + 0.03 * y_filtered[:-1]
        y_err_eff1 = y_err_filtered[-1] + 0.10 * y_filtered[-1]
        y_err_eff = np.append(y_err_eff0, y_err_eff1)

    y_err_eff = np.array(y_err_eff)

    if useEBL:
        with fits.open('table-4LAC-DR3-h.fits') as f:
            data1 = f[1].data
        idx = (data1['Source_Name'] == source_name)
        z = data1['Redshift'][idx][0]
        ebl = EblAbsorptionModel(z).transmission(x_filtered * u.MeV)
        
        def LogPar_EBL(x, Norm, alpha_, beta_):
            return logpar_base(x, Norm, alpha_, beta_) * ebl

        LogPar= LogPar_EBL
    else:
        LogPar = logpar_base
        #print('No EBL accounted for in fit.')
    
    # Least Squares for LogPar
    least_squares_logpar = LeastSquares(x_filtered, y_filtered, y_err_eff, LogPar)
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
    y_fit_logpar = LogPar(x_filtered, *popt_logpar)
    chi2_logpar, dof_logpar = reduced_chi_square(y_filtered, y_fit_logpar, y_err_eff, len(p0_logpar))

    def LogPar_axion_func(x, Norm, alpha_, beta_, w):
        return LogPar(x, Norm, alpha_, beta_) * (1 - p0 / (1 + (E_c / x) ** k) * (1+0.2*np.tanh(w)))

    # Least Squares for Axion
    least_squares_axion = LeastSquares(x_filtered, y_filtered, y_err_eff, LogPar_axion_func)
    p0_axion = [1e-11, 2.0, 0.1, np.pi]
    bounds_axion = [(1e-13, 1e-9), (1.0, 5.0), (-2.0, 2.0), (0, 2*np.pi)]

    # Minuit fit for Axion
    m_axion = Minuit(least_squares_axion, Norm=p0_axion[0], alpha_=p0_axion[1], beta_=p0_axion[2], w=p0_axion[3])

    # Set parameter limits
    for param, bound in zip(m_axion.parameters, bounds_axion):
        m_axion.limits[param] = bound

    print("\n=== Axion Fit with iminuit ===")
    m_axion.simplex() 
    m_axion.migrad()  # Minimize
    m_axion.hesse()   # Compute uncertainties
    #print(m_axion)

    # Extract results
    popt_axion = [m_axion.values[p] for p in m_axion.parameters]
    perr_axion = [m_axion.errors[p] for p in m_axion.parameters]
    y_fit_axion = LogPar_axion_func(x_filtered, *popt_axion)
    chi2_axion, dof_axion = reduced_chi_square(y_filtered, y_fit_axion, y_err_eff, len(p0_axion))

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

'''
    # Define bounds for LogPar parameters [Norm, alpha_, beta_]
    bounds_logpar = ([1e-13, -1.0, -2.0], [1e-9, 5.0, 2.0])  # Lower and upper bounds
    p0_logpar = [1e-11, 2.0, 0.1]  # Initial guesses
    popt_logpar, pcov_logpar = curve_fit(
        LogPar, x_filtered, y_filtered, sigma=y_err_eff, p0=p0_logpar, bounds=bounds_logpar, absolute_sigma=True
    )
    y_fit_logpar = LogPar(x_filtered, *popt_logpar)
    chi2_logpar, dof_logpar = reduced_chi_square(y_filtered, y_fit_logpar, y_err_eff, len(popt_logpar))

    # Extract parameter uncertainties
    perr_logpar = np.sqrt(np.diag(pcov_logpar))

    # Define bounds for axion_func parameters [Norm, alpha_, beta_]
    def LogPar_axion_func(E, Norm, alpha_, beta_, w):
        return LogPar(E, Norm, alpha_, beta_) * (1 - p0 / (1 + (E_c / E) ** k) * (1+0.2*np.tanh(w)))
    bounds_alp = ([1e-13, -1.0, -2.0, 0.0], [1e-9, 5.0, 2.0, 2.0*np.pi])  # Lower and upper bounds
    p0_alp= [1e-11, 2.0, 0.1, np.pi]  # Initial guesses
    popt_axion, pcov_axion = curve_fit(
        LogPar_axion_func, x_filtered, y_filtered, sigma=y_err_eff, p0=p0_alp, bounds=bounds_alp, absolute_sigma=True
    )
    y_fit_axion = LogPar_axion_func(x_filtered, *popt_axion)
    chi2_axion, dof_axion = reduced_chi_square(y_filtered, y_fit_axion, y_err_eff, len(popt_axion))

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
            "chi2": chi2_logpar
        },
        "LogPar_Axion": {
            "params": popt_axion,
            "errors": perr_axion,
            "cov": pcov_axion,  # Return covariance matrix
            "chi2": chi2_axion
        },
        "DeltaChi2": delta_chi2,
        "y_fit_LogPar": y_fit_logpar,
        "y_fit_Axion": y_fit_axion,
    }
'''

def nested_fits(datasets, source_name, useEBL=True):
    results = {}
    # Here we assume p0_masked and ec_masked are defined globally (or accessible in this scope)
    # and have the same shape (n_mass_masked, n_g_masked) corresponding to your m, g grid.
    for dataset_label, (x, y, y_err) in datasets.items():
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
                    p0=p0_val,
                    E_c=ec_val,
                    k=k,  # Ensure k is defined in your scope
                    source_name=source_name,
                    useEBL=useEBL
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

def plot_mean_delta_chi2_heatmap2(all_results, dataset_labels, png_naming):
    """
    Generates mean Δχ² heatmaps for each filtering category in two spaces:
      (E_c, p₀) and (mₐ, gₐ).
    
    The function assumes that the data were generated in a nested loop over masses
    and g-values so that:
      - ec_all_full and p0_all_full (reshaped from axion_data) have shape (n_mass, n_g).
      - g_unique comes from the first n_g rows and mass_unique from every n_g-th row.
    
    It uses tolerance-based matching (via np.isclose) to place each result (which stores
    a pair [p₀, E_c]) into its proper grid cell.
    """
    global axion_data, ec_all_full, p0_all_full, mass_unique, g_unique, n_mass, n_g
    
    # Get filtering methods from the first source in all_results.
    first_source = next(iter(all_results.values()))
    filtering_methods = list(first_source.keys())
    
    # Helper function to compute cell edges from centers.
    def edges_from_centers(centers):
        d = np.diff(centers)
        left = centers[0] - d[0] / 2 if len(d) > 0 else centers[0] - 0.5
        right = centers[-1] + d[-1] / 2 if len(d) > 0 else centers[-1] + 0.5
        internal = centers[:-1] + d / 2
        return np.concatenate(([left], internal, [right]))
    
    for filter_label in filtering_methods:
        # Initialize grids for summing Δχ² and counting entries.
        sum_grid = np.zeros((n_mass, n_g))
        count_grid = np.zeros((n_mass, n_g))
        
        # Loop over all sources for this filter.
        for source_name in dataset_labels:
            if source_name not in all_results or filter_label not in all_results[source_name]:
                print(f"Warning: Missing data for {filter_label} in {source_name}. Skipping.")
                continue
            dataset_results = all_results[source_name][filter_label]
            for result in dataset_results:
                p0_val = result["p0"]
                ec_val = result["E_c"]
                # Use np.isclose to locate the index in the grid:
                matches = np.where(np.isclose(p0_all_full, p0_val) & np.isclose(ec_all_full, ec_val))
                if matches[0].size == 1:
                    i = matches[0][0]
                    j = matches[1][0]
                    DeltaChi2 = result["fit_result"]["DeltaChi2"]
                    sum_grid[i, j] += DeltaChi2
                    count_grid[i, j] += 1
                else:
                    # No unique match found (could be due to floating point differences)
                    continue
        
        # Compute the mean Δχ² for each cell.
        with np.errstate(divide='ignore', invalid='ignore'):
            mean_delta_chi2_grid = np.where(count_grid != 0, sum_grid / count_grid, np.nan)
        
        # Set up colormap properties.
        vmin, vmax = -10, 25
        num_colors = 30
        boundaries = np.linspace(vmin, vmax, num_colors + 1)
        cmap = plt.get_cmap('gnuplot2_r', num_colors)
        norm = mcolors.BoundaryNorm(boundaries=boundaries, ncolors=num_colors, clip=True)
        
        # ===== Plot 1: (E_c, p₀) heatmap =====
        # For this plot, we compute cell edges using the centers from the grid.
        # Assume that along columns, the E_c values (from the first row) represent centers;
        # along rows, the p₀ values (from the first column) represent centers.
        ec_centers = ec_all_full[0, :]  # length n_g
        p0_centers = p0_all_full[:, 0]   # length n_mass
        ec_edges = edges_from_centers(ec_centers)
        p0_edges = edges_from_centers(p0_centers)
        E_c_mesh_edges, p0_mesh_edges = np.meshgrid(ec_edges, p0_edges)
        
        plt.figure(figsize=(10, 6))
        heatmap = plt.pcolormesh(E_c_mesh_edges, p0_mesh_edges, mean_delta_chi2_grid,
                                 cmap=cmap, norm=norm, shading='flat')
        # Overlay red contour if any cell has Δχ² <= -6.2.
        if np.nanmin(mean_delta_chi2_grid) <= -6.2:
            plt.contour(ec_all_full, p0_all_full, mean_delta_chi2_grid,
                        levels=[-6.2], colors='red', linewidths=2)
        cbar = plt.colorbar(heatmap, ticks=np.linspace(vmin, vmax, 11))
        cbar.set_label(r'$\langle \Delta \chi^2 \rangle$', fontsize=15)
        plt.xlabel(r'$E_c$ [MeV]', fontsize=15)
        plt.ylabel('p0', fontsize=15)
        #plt.ylim(0.0, 1/3)
        plt.title(f'Mean $\Delta \chi^2$ Heatmap for {filter_label} in (E_c, p0) Space', fontsize=15)
        plt.xscale('log')
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.tight_layout()
        plt.savefig(f'{path_to_save_heatmap_Ec_p0}{png_naming}_{filter_label}_ec_p0.png', dpi=300)
        plt.close()
        
        # =========================================================================
        # Plot 2: (mₐ, gₐ) heatmap.
        # Here we use the unique mass and g values (the grid is separable).
        # The mean Δχ² grid is the same as above (reshaped as n_mass x n_g).
        # Build a meshgrid from mass_unique and g_unique.
        ma_mesh, g_mesh = np.meshgrid(mass_unique, g_unique, indexing='ij')
        plt.figure(figsize=(10, 6))
        heatmap = plt.pcolormesh(ma_mesh/1e-9, g_mesh, mean_delta_chi2_grid,
                                 cmap=cmap, norm=norm, shading='auto')
        if np.any(mean_delta_chi2_grid <= -6.2):
            plt.contour(ma_mesh/1e-9, g_mesh, mean_delta_chi2_grid,
                        levels=[-6.2], colors='red', linewidths=2)

        cbar = plt.colorbar(heatmap, ticks=np.linspace(vmin, vmax, 11))
        cbar.set_label(r'$\langle \Delta \chi^2 \rangle$', fontsize=15)
        plt.xlabel(r'$m_a$ [neV]', fontsize=15)
        plt.ylabel(r'$g_{a\gamma}$ [GeV$^{-1}$]', fontsize=15)
        plt.title(f'Mean $\Delta \chi^2$ Heatmap for {filter_label} in ($m_a$, $g_{{a\gamma}}$) Space', fontsize=15)
        plt.xscale('log')
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.xscale('log')
        plt.yscale('log')
        plt.xlim(1e-1, 1e3)
        plt.tight_layout()
        plt.savefig(f'{path_to_save_heatmap_m_g}{png_naming}_{filter_label}_ma_ga.png', dpi=300)
        plt.close()
        
        print(f"Finished plotting for filter: {filter_label}")

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
        '''
        # Prepare scatter plot data by flattening the grids.
        # For (p0, Ec) scatter:
        p0_scatter = p0_masked.flatten()
        Ec_scatter = ec_masked.flatten()
        delta_scatter = mean_delta_chi2_grid.flatten()

        # For (m_a, g) scatter: create a full grid from the unique values.
        m_a_grid, g_grid = np.meshgrid(m_masked, g_masked, indexing='ij')
        m_a_scatter = m_a_grid.flatten()
        g_scatter = g_grid.flatten()

        # ============================================================================
        # Create Linked Scatter Plots with Color Coding by Δχ²
        # ============================================================================
        fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(14, 6))

        # Left panel: (m_a, g) space (all points, log-log scale).
        sc1 = ax_left.scatter(m_a_scatter, g_scatter, c=delta_scatter, cmap='viridis',
                            s=30, edgecolor='k')
        ax_left.set_xscale('log')
        ax_left.set_yscale('log')
        ax_left.set_xlabel('m_a (eV)')
        ax_left.set_ylabel('g_a (GeV^-1)')
        ax_left.set_title('(m_a, g_a) Space\nColor-coded by Δχ²')
        cbar1 = plt.colorbar(sc1, ax=ax_left)
        cbar1.set_label('Δχ²')

        # Right panel: (p0, Ec) space.
        # Filter to only include points with negative Δχ².
        sc2 = ax_right.scatter(Ec_scatter, p0_scatter, c=delta_scatter, cmap='viridis',
                            s=30, edgecolor='k')
        ax_right.set_xlabel('E_c (MeV)')
        ax_right.set_ylabel('p0')
        ax_right.set_title('(p0, E_c) Space')
        ax_right.set_ylim(0, 1)  # p0 goes from 0 to 1
        ax_right.set_xlim(0, 2000)
        cbar2 = plt.colorbar(sc2, ax=ax_right)
        cbar2.set_label('Δχ²')

        plt.tight_layout()
        plt.savefig(f"{path_to_save_heatmap_m_g}{png_naming}_{filter_label}_linked_scatter_color_coded.png", dpi=300)
        '''
        
        print(f"Finished plotting for filter: {filter_label}")


all_results_none = {}
all_results_snr = {}
all_results_lin = {}
with open(f'Top5_Source_ra_dec_specin.txt', 'r') as file:
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
                    datasets = {f"No_Filtering": (sorted_data_none['geometric_mean'], sorted_data_none['flux_tot_value']/bin_size, sorted_data_none['flux_tot_error']/bin_size)}
                    datasets_snr = {f"snr_3": (sorted_data_snr3['geometric_mean'], sorted_data_snr3['flux_tot_value']/bin_size, sorted_data_snr3['flux_tot_error']/bin_size),
                                    f"snr_5": (sorted_data_snr5['geometric_mean'], sorted_data_snr5['flux_tot_value']/bin_size, sorted_data_snr5['flux_tot_error']/bin_size),
                                    f"snr_10": (sorted_data_snr10['geometric_mean'], sorted_data_snr10['flux_tot_value']/bin_size, sorted_data_snr10['flux_tot_error']/bin_size)}
                    datasets_lin = {f"week": (sorted_data_lin_week['geometric_mean'], sorted_data_lin_week['flux_tot_value']/bin_size, sorted_data_lin_week['flux_tot_error']/bin_size),
                                    f"month": (sorted_data_lin_month['geometric_mean'], sorted_data_lin_month['flux_tot_value']/bin_size, sorted_data_lin_month['flux_tot_error']/bin_size)}
                    print(source_name)

                    results = nested_fits(datasets, source_name, useEBL=True)
                    results_snr = nested_fits(datasets_snr, source_name, useEBL=True)
                    results_lin = nested_fits(datasets_lin, source_name, useEBL=True)

                    all_results_none[source_name] = results
                    all_results_snr[source_name] = results_snr
                    all_results_lin[source_name] = results_lin

                    '''
                    plot_delta_chi2_heatmap(results, dataset_label="No_Filtering", png_naming =f"{source_name_cleaned}")
                    
                    plot_delta_chi2_heatmap(results_snr, dataset_label="snr_3", png_naming =f"{source_name_cleaned}")
                    plot_delta_chi2_heatmap(results_snr, dataset_label="snr_5", png_naming =f"{source_name_cleaned}")
                    plot_delta_chi2_heatmap(results_snr, dataset_label="snr_10", png_naming =f"{source_name_cleaned}")
                    
                    plot_delta_chi2_heatmap(results_lin, dataset_label="week", png_naming =f"{source_name_cleaned}")
                    plot_delta_chi2_heatmap(results_lin, dataset_label="month", png_naming =f"{source_name_cleaned}")
                    print("(p0, Ec) Heatmaps done!")
                    '''

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


'''
plot_delta_chi2_heatmap_m_g(results, dataset_label="No_Filtering")
plot_delta_chi2_heatmap_m_g(results_snr, dataset_label="snr_3")
plot_delta_chi2_heatmap_m_g(results_snr, dataset_label="snr_5")
plot_delta_chi2_heatmap_m_g(results_snr, dataset_label="snr_10")
plot_delta_chi2_heatmap_m_g(results_lin, dataset_label="week")
plot_delta_chi2_heatmap_m_g(results_lin, dataset_label="month")


def plot_fits_with_residuals(datasets, results, png_naming ="", residual_colors=None):
    """
    Plot fits with normalized residuals for each dataset.

    Parameters:
        datasets (dict): Dictionary of datasets with format:
                         {label: (x, y, y_err)}
        results (dict): Dictionary of fit results with format:
                        {label: {
                            "y_fit_LogPar": array,
                            "y_fit_Axion": array,
                            "LogPar": {
                                "params": [values],
                                "errors": [errors]
                            },
                            "Axion": {
                                "params": [values],
                                "errors": [errors]
                            },
                            "chi2_LogPar": value,
                            "chi2_Axion": value,
                            "dof": value  # Degrees of freedom
                        }}
        residual_colors (dict): Optional. Dictionary of colors for residuals:
                                {"LogPar": color1, "Axion": color2}
    """
    # Default colors for residuals
    if residual_colors is None:
        residual_colors = {"LogPar": "red", "Axion": "blue"}

    for dataset_label, (x, y, y_err) in datasets.items():
        # Extract fit results
        y_fit_logpar = results[dataset_label]["y_fit_LogPar"]
        y_fit_axion = results[dataset_label]["y_fit_Axion"]

        # Extract parameters and chi-squared values
        params_logpar = results[dataset_label]["LogPar"]["params"]
        errors_logpar = results[dataset_label]["LogPar"]["errors"]
        params_axion = results[dataset_label]["Axion"]["params"]
        errors_axion = results[dataset_label]["Axion"]["errors"]
        chi2_logpar = results[dataset_label]["LogPar"]['chi2']
        chi2_axion = results[dataset_label]['Axion']['chi2']
        red_chi_sq = results[dataset_label]['DeltaChi2']

        # Calculate normalized residuals
        residuals_logpar = (y - y_fit_logpar) / y_err
        residuals_axion = (y - y_fit_axion) / y_err

        # Create figure
        fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True, gridspec_kw={'height_ratios': [3, 1]})

        # Top plot: Data and fits
        axs[0].errorbar(x, y, xerr = [e_lowers, e_uppers],yerr=y_err, fmt='o', label="Data", color='black')
        axs[0].plot(x, y_fit_logpar, label="LogPar Fit", linestyle='-', color=residual_colors["LogPar"])
        axs[0].plot(x, y_fit_axion, label="Axion Fit", linestyle='--', color=residual_colors["Axion"])
        axs[0].set_ylabel("dN/dE [ photons/cm²/s/MeV ]")
        axs[0].set_title(f"Fits for {dataset_label}")
        axs[0].set_xscale('log')
        axs[0].set_yscale('log')
        axs[0].legend()
        axs[0].grid(True, which="both", linestyle="--", linewidth=0.5)

        # Add parameter box
        textstr = "LogPar:\n"
        for param, value, error in zip(["Norm", "alpha_", "beta_"], params_logpar, errors_logpar):
            textstr += f"  {param}: {value:.2e} ± {error:.2e}\n"
        textstr += f"  Reduced $\chi^2$: {chi2_logpar}\n\n"

        textstr += "Axion:\n"
        for param, value, error in zip(["Norm", "alpha_", "beta_", "p0", "E_c", "k"], params_axion, errors_axion):
            textstr += f"  {param}: {value:.2e} ± {error:.2e}\n"
        textstr += f"  Reduced $\chi^2$: {chi2_axion}\n\n"

        textstr += f"Δχ² (Axion - LogPar): {red_chi_sq:.2f}"

        axs[0].text(0.02, 0.56, textstr, transform=axs[0].transAxes, fontsize=10,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        # Bottom plot: Residuals
        axs[1].scatter(x, residuals_logpar, label="LogPar Residuals", color=residual_colors["LogPar"], marker='o')
        axs[1].scatter(x, residuals_axion, label="Axion Residuals", color=residual_colors["Axion"], marker='x')
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
        plt.savefig(f'snr/{dataset_label}{png_naming}.png', dpi=300)


def save_fit_results_to_latex_manual(results, filename="fit_results.tex"):
    """
    Save fit results into a LaTeX table manually, placing LogPar and Axion on separate rows.

    Parameters:
        results (dict): Fit results for datasets.
        filename (str): LaTeX output file.
    """
    # Define LaTeX table headers
    headers = [
        "Dataset",
        "Model",
        "$\\chi^2$",
        "Norm",
        "$\\alpha$",
        "$\\beta$",
        "$p_0$",
        "$E_c$",
        "$k$",
    ]

    # Start building the LaTeX table string
    table_lines = []
    table_lines.append("\\begin{table}[h!]")
    table_lines.append("\\centering")
    table_lines.append("\\resizebox{\\textwidth}{!}{%")
    table_lines.append("\\begin{tabular}{lcccccccc}")
    table_lines.append("\\toprule")

    # Add headers
    table_lines.append(" & ".join(headers) + " \\\\" + "\\midrule")

    # Add data rows
    for label, result in results.items():
        # LogPar results
        logpar_params = result["LogPar"]["params"]
        logpar_errors = result["LogPar"]["errors"]
        logpar_chi2 = result["LogPar"]["chi2"]

        # Axion results
        axion_params = result["Axion"]["params"]
        axion_errors = result["Axion"]["errors"]
        axion_chi2 = result["Axion"]["chi2"]

        # Delta Chi2
        delta_chi2 = result["DeltaChi2"]
            # Row for LogPar
        logpar_row = [
            label,  # Dataset name
            "LogPar",  # Model name
            f"{logpar_chi2:.2f}",
            f"{float(logpar_params[0]):.2e} $\\pm$ {float(logpar_errors[0]):.2e}",  # Norm
        ] + [
            f"{float(p):.3f} $\\pm$ {float(e):.3f}"
            for p, e in zip(logpar_params[1:], logpar_errors[1:])
        ] + ["", "", ""]  # Fill remaining cells

        # Row for Axion
        axion_row = [
            "",  # Empty dataset name for continuation row
            "Axion",  # Model name
            f"{axion_chi2:.2f}",
            f"{float(axion_params[0]):.2e} $\\pm$ {float(axion_errors[0]):.2e}",  # Norm
        ] + [
            f"{float(p):.3f} $\\pm$ {float(e):.3f}"
            for p, e in zip(axion_params[1:], axion_errors[1:])
        ]

        # Row for Axion
        deltaChi_row = [
            "$\\Delta\\chi^2$",  # Empty dataset name for continuation row
            "",  # empty Model name
            f"{delta_chi2:.2f}",
        ] + ["", "", "" ,"", "", ""]  # Fill remaining cells

        # Add rows to table
        table_lines.append(" & ".join(logpar_row) + " \\\\")
        table_lines.append(" & ".join(axion_row) + " \\\\")
        table_lines.append(" & ".join(deltaChi_row) + " \\\\"+ "\\midrule")

    # End the table
    table_lines.append("\\end{tabular}%")
    table_lines.append("}")
    table_lines.append("\\caption{Fit Results for Spectral Points}")
    table_lines.append("\\label{tab:fit_results}")
    table_lines.append("\\end{table}")

    # Write the LaTeX table to a file
    with open(filename, "w") as f:
        f.write("\n".join(table_lines))

    print(f"LaTeX table saved to {filename}")


save_fit_results_to_latex_manual(results, filename="no_filtering_fit_results.tex")
save_fit_results_to_latex_manual(results_snr, filename="snr_fit_results.tex")
save_fit_results_to_latex_manual(results_lin, filename="lin_fit_results.tex")

plot_fits_with_residuals(datasets, results)
plot_fits_with_residuals(datasets_snr, results_snr)
plot_fits_with_residuals(datasets_lin, results_lin)

axion_data = np.load('denys/Rikke/Data/scan12.npy')
prob = lambda E, p0, ec, k:  p0/(1 + (ec/E)**k ) # conversion probability fit-function
test_energies = np.logspace(-2,7,91) #MeV
        
ma_all = axion_data[:,0] #eV
g_all = axion_data[:,1] # GeV**-1
ec_all = axion_data[:,2]/1e6 #MeV
p0_all = axion_data[:,3]
k_all = axion_data[:,4]
print('first p0', p0_all[0], p0_all[-1])
print('first ec', ec_all[0], ec_all[-1])
print(len(ma_all))
def flip_heatmap_to_ma_g(ec_values, p0_values, delta_chi2_grid, ma_sim, g_sim, ec_sim, p0_sim):
    """
    Converts a heatmap from (Ec, p0) space to (ma, g) space using nearest-neighbor mapping.

    Parameters:
    - ec_values: 1D np.array of Ec grid points
    - p0_values: 1D np.array of p0 grid points
    - delta_chi2_grid: 2D np.array of mean Δχ² values in (Ec, p0) space
    - ma_sim: np.array of simulated ma values (eV)
    - g_sim: np.array of simulated g values (GeV^-1)
    - ec_sim: np.array of simulated Ec values (eV)
    - p0_sim: np.array of simulated p0 values

    Returns:
    - ma_grid: 2D np.array of mapped ma values
    - g_grid: 2D np.array of mapped g values
    - delta_chi2_ma_g: 2D np.array of Δχ² values in (ma, g) space
    """

    # Build KDTree for simulated (Ec, p0)
    tree = KDTree(np.column_stack((ec_sim, p0_sim)))

    # Map each (Ec, p0) grid point to the closest (ma, g)
    ma_grid = np.zeros_like(delta_chi2_grid)
    g_grid = np.zeros_like(delta_chi2_grid)

    for i in range(len(p0_values)):
        for j in range(len(ec_values)):
            # Query closest (ma, g) for each (Ec, p0) grid point
            _, idx = tree.query([ec_values[j], p0_values[i]])
            ma_grid[i, j] = ma_sim[idx]
            g_grid[i, j] = g_sim[idx]

    return ma_grid, g_grid, delta_chi2_grid

def plot_delta_chi2_heatmap_m_g(results, dataset_label):
    
    # Extract results for the specified dataset
    if dataset_label not in results:
        raise ValueError(f"Dataset label '{dataset_label}' not found in results.")

    dataset_results = results[dataset_label]

    # Initialize a 2D array to store DeltaChi2 values
    delta_chi2_grid_p0_Ec = np.zeros((len(p0_values), len(E_c_values)))
    delta_chi2_grid_m_g = np.zeros((len(g_all), len(ma_all)))

    for result in dataset_results:
        p0 = result["p0"]
        E_c = result["E_c"]
        DeltaChi2 = result["fit_result"]["DeltaChi2"]

        # Find indices
        p0_index = p0_values.tolist().index(p0)
        E_c_index = E_c_values.tolist().index(E_c)

        # Store DeltaChi2 value
        delta_chi2_grid_p0_Ec[p0_index, E_c_index] = DeltaChi2

    # Find the minimum Chi² and corresponding parameters
    min_idx = np.unravel_index(np.argmin(delta_chi2_grid_p0_Ec), delta_chi2_grid_p0_Ec.shape)
    best_p0 = p0_values[min_idx[0]]
    best_Ec = E_c_values[min_idx[1]]
    print(f"Best fit Δχ²={delta_chi2_grid_p0_Ec[min_idx[0],min_idx[1]]:.2f}: p0 = {best_p0:.2f}, Ec = {best_Ec:.2f}")
    #p0diff = np.abs( p0_all - best_p0 )
    Ecdiff = np.abs( ec_all/1e6 - best_Ec )
    
    #p00 = p0_all[ np.argmin(p0diff) ]
    Ec0 = ec_all[ np.argmin(Ecdiff) ]
    
    print(f'Closest p0, Ec = None ; {Ec0/1e6:.2f} MeV')
    #idx = (p0_all==p00)&(ec_all==Ec0)
    #idx0 = np.where(p0_all==p00)
    idx1 = np.where(ec_all==Ec0)
    #print(idx)
    #print(idx0)
    print(idx1)
    
    ma0 = ma_all[idx1]
    ga0 = g_all[idx1]

    print(f'Closest ma, g = {ma0} ; {ga0}')

    # Plot the transition probability
    plt.figure(figsize=(8, 6))
    plt.errorbar(
        test_energies,
        prob(test_energies, best_p0, best_Ec, k),
        fmt='-',
        label=f'Best fit Δχ²={delta_chi2_grid_p0_Ec[min_idx[0],min_idx[1]]:.2f}: p0={best_p0:.2f}, Ec={best_Ec:.2f}, k={k}'
    )

    # Set log scale for x-axis
    plt.xscale('log')
    plt.xlabel('E [MeV]')
    plt.ylabel('P(γ → a)')
    plt.title('Transition Probability')
    plt.legend(loc="upper left")
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.savefig(f'snr/heatmaps/transition_prob_{dataset_label}.png', dpi=300)
'''