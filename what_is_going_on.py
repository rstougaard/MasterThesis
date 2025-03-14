import pickle
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

path_to_save_heatmap_m_g = "./fit_results/heatmaps_m_g/"
path_to_save_heatmap_Ec_p0 = "./fit_results/heatmaps_Ec_p0/"

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

def plot_individual_delta_chi2_heatmap_with_pdf(all_results, dataset_labels, png_naming, filtering_methods="No_Filtering", pdf_filename="heatmaps.pdf", no_filtering_grid=None):
    """
    Generates individual Δχ² heatmaps for each source for the specified filtering method(s) in (mₐ, gₐ) space,
    saves each as a PNG file, and collects all plots into a single PDF.
    
    Parameters:
      - all_results: dict where keys are source names and values are dicts containing filtering method data.
      - dataset_labels: labels associated with the datasets.
      - png_naming: base string for saving PNG files.
      - filtering_methods: filtering method(s) to use. Can be a string (e.g., "No_Filtering") or a list of strings.
      - pdf_filename: filename for the combined PDF output.
      - no_filtering_grid: Optional grid to overlay as a contour if provided.
    
    Assumes that global variables (axion_data, ec_masked, p0_masked, m_masked, g_masked, n_mass, n_g, path_to_save_heatmap_m_g)
    are available and that compute_mean_delta_chi2_grid is defined elsewhere.
    """
    global axion_data, ec_masked, p0_masked, m_masked, g_masked, n_mass, n_g, path_to_save_heatmap_m_g
    
    # If filtering_methods is a string, convert it to a list.
    if isinstance(filtering_methods, str):
        filtering_methods = [filtering_methods]
    
    # Build the (mₐ, gₐ) mesh once.
    ma_mesh, g_mesh = np.meshgrid(m_masked, g_masked, indexing='ij')
    
    # Create a PdfPages object to collect plots.
    with PdfPages(f'./fit_results/{pdf_filename}') as pdf:
        # Loop over each source in the results.
        for source_name, source_data in all_results.items():
            for filter_label in filtering_methods:
                # Compute the mean Δχ² grid for the current source and filtering method.
                mean_delta_chi2_grid = compute_mean_delta_chi2_grid(
                    all_results={source_name: source_data},
                    dataset_labels=dataset_labels,
                    filter_label=filter_label,
                    p0_masked=p0_masked,
                    ec_masked=ec_masked
                )
                
                print(f"Source: {source_name} | Filter: {filter_label} | min: {np.min(mean_delta_chi2_grid)}, max: {np.max(mean_delta_chi2_grid)}")
                
                # Set up colormap parameters.
                vmin, vmax = -10, 25
                num_colors = 30
                boundaries = np.linspace(vmin, vmax, num_colors + 1)
                cmap = plt.get_cmap('gnuplot2', num_colors)
                norm = mcolors.BoundaryNorm(boundaries=boundaries, ncolors=num_colors, clip=True)
                
                # Create the figure.
                fig = plt.figure(figsize=(10, 6))
                heatmap = plt.pcolormesh(ma_mesh / 1e-9, g_mesh, mean_delta_chi2_grid,
                                         cmap=cmap, norm=norm, shading='auto')
                
                # Overlay no_filtering_grid contour if provided and if we aren't plotting "No_Filtering" itself.
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
                plt.title(f'{source_name} | {filter_label} Δχ² Heatmap in ($m_a$, $g_{{a\gamma}}$) Space', fontsize=15)
                plt.xscale('log')
                plt.yscale('log')
                plt.xticks(fontsize=15)
                plt.yticks(fontsize=15)
                plt.tight_layout()
                
                # Save the current figure as a PNG file.
                png_filepath = f'{path_to_save_heatmap_m_g}{png_naming}_{source_name}_{filter_label}_ma_ga.png'
                #plt.savefig(png_filepath, dpi=300)
                #print(f"Saved PNG: {png_filepath}")
                
                # Add the current figure as a page in the PDF.
                pdf.savefig(fig)
                plt.close(fig)
                
                print(f"Finished plotting for source: {source_name} with filter: {filter_label}")

    print(f"All plots have been saved to the PDF: {pdf_filename}")

with open("all_results_none_32_curve_fit_nodivbin.pkl", "rb") as file:
    all_results_none = pickle.load(file)

with open("all_results_lin_32_curve_fit_nodivbin.pkl", "rb") as file:
    all_results_lin = pickle.load(file)

with open("all_results_none_32_curve_fit_nodivbin.pkl", "rb") as file:
    all_results_snr = pickle.load(file)

no_filtering_sources = list(all_results_none.keys()) 
plot_individual_delta_chi2_heatmap_with_pdf(all_results_snr, no_filtering_sources, " ", filtering_methods="snr_10", pdf_filename="indv_heatmaps_snr10.pdf")
