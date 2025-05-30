import pickle
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D
import numpy as np
import os
from matplotlib.backends.backend_pdf import PdfPages

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
m_start_val = 1e-10
g_start_val = 5e-13
m_stop_val  = 9e-9
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
                                 p0_masked, ec_masked, remove_source_label):
    """
    all_results: e.g. all_results_none or all_results_snr
    dataset_labels: list of keys in that dictionary
    filter_label: which key to look for in all_results[source][filter_label]
    p0_masked, ec_masked: your global grids
    Returns: sum Δχ² grid
    """
    sum_grid = np.zeros((ec_masked.shape[0], ec_masked.shape[1]))
    count_grid = np.zeros((ec_masked.shape[0], ec_masked.shape[1]))

    if remove_source_label is not None:
        if isinstance(remove_source_label, (list, tuple, set)):
            labels_to_remove = remove_source_label
        else:
            labels_to_remove = [remove_source_label]

        for lbl in labels_to_remove:
            all_results.pop(lbl, None)
    
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
    
    print("Minimum sum Δχ²:", np.nanmin(sum_grid))
    print("Maximum sum Δχ²:", np.nanmax(sum_grid))
    return sum_grid

def plot_individual_delta_chi2_heatmap_with_pdf(all_results, dataset_labels, systematic_results ,png_naming="", filtering_methods="No_Filtering", pdf_filename="heatmaps.pdf", no_filtering_grid=None):
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

        for (source_name, source_data1), (source_name, source_data2) in zip(all_results.items(),systematic_results.items()):
            for filter_label in filtering_methods:
                # Compute the mean Δχ² grid for the current source and filtering method.
                mean_delta_chi2_grid = compute_mean_delta_chi2_grid(
                    all_results={source_name: source_data1},
                    dataset_labels=dataset_labels,
                    filter_label=filter_label,
                    p0_masked=p0_masked,
                    ec_masked=ec_masked,
                    remove_source_label=None
                )
                mean_systematic_results = compute_mean_delta_chi2_grid(
                                all_results={source_name: source_data2},  
                                dataset_labels=dataset_labels,
                                filter_label=filter_label,
                                p0_masked=p0_masked,
                                ec_masked=ec_masked,
                                remove_source_label=None
                            )
                                
                print(f"Source: {source_name} | Filter: {filter_label} | min: {np.min(mean_delta_chi2_grid)}, max: {np.max(mean_delta_chi2_grid)}")
                
                # Set up colormap parameters.
                grid_min = np.min(mean_delta_chi2_grid)
                grid_max = np.max(mean_delta_chi2_grid)

                # If both the min and max are within [-1, 1], use fixed values.
                if source_name == "4FGL J0319.8+4130":
                    vmin, vmax = -10 , 25
                elif grid_min > -1 and grid_max < 1:
                    vmin, vmax = -1, 1
                else:
                    vmin, vmax = grid_min, grid_max
                num_colors = 120
                boundaries = np.linspace(vmin, vmax, num_colors + 1)
                cmap = plt.get_cmap('gnuplot2', num_colors)
                norm = mcolors.BoundaryNorm(boundaries=boundaries, ncolors=num_colors, clip=True)
                
                # Create the figure.
                fig = plt.figure(figsize=(10, 6))
                heatmap = plt.pcolormesh(ma_mesh / 1e-9, g_mesh, mean_delta_chi2_grid,
                                         cmap=cmap, norm=norm, shading='auto')
                                    
                plot_specs = []
                # — Mean (filtered) —
                if mean_delta_chi2_grid is not None:
                    plot_specs.append({
                        'grid': mean_delta_chi2_grid,
                        'label': 'Without systematics',
                        'color_pos': 'cyan',   'linestyle_pos': 'solid',
                        'color_neg': 'lime',  'linestyle_neg': 'solid'
                    })

                # — Systematics (filtered) —
                if systematic_results is not None:
                    plot_specs.append({
                        'grid': mean_systematic_results,
                        'label': 'With systematics',
                        'color_pos': 'green',   'linestyle_pos': 'dashed',
                        'color_neg': 'red',  'linestyle_neg': 'dashed'
                    })


                # --- Plot contours (remove label= from contour calls) ---
                for spec in plot_specs:
                    grid = spec['grid']
                    x = ma_mesh / 1e-9
                    y = g_mesh

                    if spec.get('color_pos') and np.any(grid >= 6.2):
                        plt.contour(x, y, grid,
                                    levels=[6.2],
                                    colors=spec['color_pos'],
                                    linestyles=spec['linestyle_pos'],
                                    linewidths=2)

                    if spec.get('color_neg') and np.any(grid <= -6.2):
                        plt.contour(x, y, grid,
                                    levels=[-6.2],
                                    colors=spec['color_neg'],
                                    linestyles=spec['linestyle_neg'],
                                    linewidths=2)

                # --- Build proxy legend handles for both aspects ---
                
                color_handles = [
                    Line2D([0], [0], color='red', linestyle='-', linewidth=2, label=f'$> 6.2$'),
                    Line2D([0], [0], color='lime', linestyle='-', linewidth=2, label=f'$< -6.2$')
                ]


                # 2. Legend for systematics (line style) mapping:
                linestyle_handles = [
                    Line2D([0], [0], color='black', linestyle='solid', linewidth=2, label='Without systematics'),
                    Line2D([0], [0], color='black', linestyle='dashed', linewidth=2, label='With systematics')
                ]

                # Create the two legends. Add the first legend to the axes so that the second does not overwrite it.
                #legend1 = plt.legend(handles=color_handles, loc='upper left', title="Threshold")
                #plt.gca().add_artist(legend1)
                #plt.legend(handles=linestyle_handles, loc='lower left', title="Systematics")
                        
                cbar = plt.colorbar(heatmap, ticks=np.linspace(vmin, vmax, 11))
                cbar.set_label(r'$ \Delta \chi^2 $')
                plt.xlabel(r'$m_a$ [neV]')
                plt.ylabel(r'$g_{a\gamma}$ [GeV$^{-1}$]')
                '''if filter_label == "No_Filtering":
                    if source_name == "4FGL J0319.8+4130":
                        plt.title(f'No filtering')
                    else:
                        plt.title(f'{source_name} : No filtering $ \Delta \chi^2 $ Heatmap')
                elif filter_label == "week":
                    if source_name == "4FGL J0319.8+4130":
                        plt.title(f'Weekly filter')
                    else:
                        plt.title(f'{source_name} : Weekly filter $ \Delta \chi^2 $ Heatmap')
                elif filter_label == "month":
                    if source_name == "4FGL J0319.8+4130":
                        plt.title(f'Monthly filter')
                    else:
                        plt.title(f'{source_name} : Monthly filter $ \Delta \chi^2 $ Heatmap')
                elif filter_label == "snr_3":
                    if source_name == "4FGL J0319.8+4130":
                        plt.title(f'SNR=3 filter')
                    else:
                        plt.title(f'{source_name} : SNR=3 filter $ \Delta \chi^2 $ Heatmap')
                elif filter_label == "snr_5":
                    if source_name == "4FGL J0319.8+4130":
                        plt.title(f'SNR 5 filter ')
                    else:
                        plt.title(f'{source_name} : SNR=5 filter $ \Delta \chi^2 $ Heatmap')
                elif filter_label == "snr_10":
                    if source_name == "4FGL J0319.8+4130":
                        plt.title(f'SNR=10 filter')
                    else:
                        plt.title(f'{source_name} : SNR 10 filter $ \Delta \chi^2 $ Heatmap')'''
                plt.xscale('log')
                plt.yscale('log')
                ax = plt.gca()
                #ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%g"))
                ax.set_xlim(0.3, 9)
                plt.tick_params(
                axis='both',
                which='both',
                color='white',       # tick *lines*
                labelcolor='black',  # tic        # if you want to override tick‐label size here
                direction='in',
                top=True, right=True
                )
                ax.set_xticks([1, 9])                  # put ticks at 10^0, 10^1
                ax.set_xticklabels(['1', '9'])  
                plt.tight_layout()
                
                # Save the current figure as a PNG file.
                if source_name == "4FGL J0319.8+4130":
                    png_filepath = f'./fit_results/NGC1275_{filter_label}_ma_ga_{png_naming}.png'
                    plt.savefig(png_filepath, dpi=300)
                #print(f"Saved PNG: {png_filepath}")
                
                # Add the current figure as a page in the PDF.
                pdf.savefig(fig)
                plt.close(fig)
                
                print(f"Finished plotting for source: {source_name} with filter: {filter_label}")

    print(f"All plots have been saved to the PDF: {pdf_filename}")
    return

def plot_mean_delta_chi2_heatmap_nosys_base(all_results,
    all_results_sys,
    dataset_labels,
    png_naming,
    no_filtering_grid=None,
    no_filtering_grid_other=None,
    remove_source_label=None
):
    # Remove the specified source, if provided.
    if remove_source_label is not None:
        if isinstance(remove_source_label, (list, tuple, set)):
            labels_to_remove = remove_source_label
        else:
            labels_to_remove = [remove_source_label]

        for lbl in labels_to_remove:
            all_results.pop(lbl, None)
            all_results_sys.pop(lbl, None)

    # Extract all filter labels from the first remaining source.
    first_source = next(iter(all_results.values()))
    filtering_methods = list(first_source.keys())

    # Build the (m_a, g_a) mesh
    ma_mesh, g_mesh = np.meshgrid(m_masked, g_masked, indexing='ij')

    for filter_label in filtering_methods:
        # -- 1) Compute mean Δχ² for this filter_label --
        mean_delta_chi2_grid = compute_mean_delta_chi2_grid(
            all_results=all_results,
            dataset_labels=dataset_labels,
            filter_label=filter_label,
            p0_masked=p0_masked,
            ec_masked=ec_masked,
            remove_source_label=None
        )
        systematic_grid = (compute_mean_delta_chi2_grid(
            all_results=all_results_sys,
            dataset_labels=dataset_labels,
            filter_label=filter_label,
            p0_masked=p0_masked,
            ec_masked=ec_masked,
            remove_source_label=None
        ) if all_results_sys else None)

        # Set up colormap.
        vmin, vmax = -10, 25#-20, 50 #int(np.min(mean_delta_chi2_grid))-1, int(np.max(mean_delta_chi2_grid))+1
        num_colors = 120
        boundaries = np.linspace(vmin, vmax, num_colors + 1)
        cmap = plt.get_cmap('gnuplot2', num_colors)
        norm = mcolors.BoundaryNorm(boundaries=boundaries, ncolors=num_colors, clip=True)

        ma_mesh, g_mesh = np.meshgrid(m_masked, g_masked, indexing='ij')

        plt.figure(figsize=(10, 6))
        heatmap = plt.pcolormesh(ma_mesh / 1e-9, g_mesh, mean_delta_chi2_grid,
                                 cmap=cmap, norm=norm, shading='auto')

        plot_specs = []
        # — Mean (filtered) —
        if mean_delta_chi2_grid is not None:
            plot_specs.append({
                'grid': mean_delta_chi2_grid,
                'label': 'Without systematics',
                'color_pos': 'cyan','linestyle_pos': 'solid',
                'color_neg': 'lime', 'linestyle_neg': 'solid'
            })

        # — Mean (no filtering) —
        if no_filtering_grid is not None:
            plot_specs.append({
                'grid': no_filtering_grid,
                'label': 'No filtering',
                'color_neg': 'brown', 'linestyle_neg': 'solid',
                'color_pos': 'cyan', 'linestyle_pos': 'solid'
            })

        # — Systematics (filtered) —
        if systematic_grid is not None:
            plot_specs.append({
                'grid': systematic_grid,
                'label': 'With systematics',
                'color_pos': 'green',   'linestyle_pos': 'dashed',
                'color_neg': 'red',  'linestyle_neg': 'dashed'
            })

        # — Systematics (no filtering) —
        if no_filtering_grid_other is not None:
            plot_specs.append({
                'grid': no_filtering_grid_other,
                'label': 'No filtering',
                'color_neg': 'darkorange', 'linestyle_neg': 'dashed',
                'color_pos': 'cyan', 'linestyle_pos': 'dashed'
            })

        # --- Plot contours (remove label= from contour calls) ---
        for spec in plot_specs:
            grid = spec['grid']
            x = ma_mesh / 1e-9
            y = g_mesh

            if spec.get('color_pos') and np.any(grid >= 6.2):
                plt.contour(x, y, grid,
                            levels=[6.2],
                            colors=spec['color_pos'],
                            linestyles=spec['linestyle_pos'],
                            linewidths=2)

            if spec.get('color_neg') and np.any(grid <= -6.2):
                plt.contour(x, y, grid,
                            levels=[-6.2],
                            colors=spec['color_neg'],
                            linestyles=spec['linestyle_neg'],
                            linewidths=2)

        # --- Build proxy legend handles for both aspects ---
        if no_filtering_grid is not None:
            # 1. Legend for threshold (color) mapping:
            color_handles = [
                Line2D([0], [0], color='red', linestyle='-', linewidth=2, label=f'$> 6.2)$'),
                Line2D([0], [0], color='lime', linestyle='-', linewidth=2, label=f'$< -6.2$'),
                # For "No filtering" the color is white. Use a marker with a black edge to make it visible.
                Line2D([0], [0], marker='s', markersize=8, markerfacecolor='white', 
                    markeredgecolor='black', linestyle='-', linewidth=2, label='No filtering (white)')
            ]
        else:
            color_handles = [
                Line2D([0], [0], color='red', linestyle='-', linewidth=2, label=f'$> 6.2$'),
                Line2D([0], [0], color='lime', linestyle='-', linewidth=2, label=f'$< -6.2$')
            ]


        # 2. Legend for systematics (line style) mapping:
        linestyle_handles = [
            Line2D([0], [0], color='black', linestyle='solid', linewidth=2, label='Without systematics'),
            Line2D([0], [0], color='black', linestyle='dashed', linewidth=2, label='With systematics')
        ]

        # Create the two legends. Add the first legend to the axes so that the second does not overwrite it.
        #legend1 = plt.legend(handles=color_handles, loc='upper left', title="Threshold")
        #plt.gca().add_artist(legend1)
        #plt.legend(handles=linestyle_handles, loc='lower left', title="Systematics")

        #data = np.load('fermi3_10_contours_b5.2_eta0.67_rescale.npz')
        #plt.errorbar(10**data['x1'][data['y1'] < 1e-11]/ 1e-9, 10**data['y1'][data['y1'] < 1e-11], color='b', ls=(0,(4,2,1,1,1,2)), alpha=1)
        '''
        ##### plot contours for 0 level of syst uncertainty ##################
        data = np.load('fermi0_0_contours_ebl_6.2_scan12_jointfit.npz')
        plt.errorbar(10**data['x'][data['y'] < 1e-11]/ 1e-9, 1.*10**data['y'][data['y'] < 1e-11], fmt='w:')
        plt.errorbar(10**data['x1'][data['y1'] < 1e-11]/ 1e-9, 1.*10**data['y1'][data['y1'] < 1e-11], fmt='w:')
        
        data = np.load('fermi3_10_contours_06032025.npz')
        print(10**data['x1']/ 1e-9, 10**data['y1'])
        plt.errorbar(10**data['x1']/ 1e-9, 10**data['y1'], fmt='w-')#, zorder=-10)
        '''
        cbar = plt.colorbar(heatmap, ticks=np.linspace(vmin, vmax, 11))
        cbar.set_label(r'$\sum \Delta \chi^2$')
        plt.xlabel(r'$m_a$ [neV]')
        plt.ylabel(r'$g_{a\gamma}$ [GeV$^{-1}$]')
        '''#plt.title(f'Summed $\Delta \chi^2$ Heatmap for {filter_label}', fontsize=15)
        if filter_label == "No_Filtering":
            plt.title(f'No filtering')
        elif filter_label == "week":
            plt.title(f'Weekly filter')
        elif filter_label == "month":
            plt.title(f'Monthly filter')
        elif filter_label == "snr_3":
            plt.title(f'SNR=3 filter')
        elif filter_label == "snr_5":
            plt.title(f'SNR=5 filter')
        elif filter_label == "snr_10":
            plt.title(f'SNR=10 filter')'''
        plt.xscale('log')
        plt.yscale('log')
        #plt.xticks(fontsize=15)
        #plt.yticks(fontsize=15)
        #plt.gca().xaxis.set_major_formatter(ticker.FormatStrFormatter("%g"))
        ax = plt.gca()
        plt.tick_params(
        axis='both',
        which='both',
        color='white',       # tick *lines*
        labelcolor='black',        # if you want to override tick‐label size here
        direction='in',
        top=True, right=True
        )
        plt.xlim(0.3, 9)
        ax.set_xticks([1, 9])                  # put ticks at 10^0, 10^1
        ax.set_xticklabels(['1', '9'])  
        plt.tight_layout()
        plt.savefig(f'{path_to_save_heatmap_m_g}{png_naming}_{filter_label}_ma_ga.png', dpi=300)
        plt.close()

        print(f"Finished plotting for filter: {filter_label}")

def plot_mean_delta_chi2_heatmap_sys_base(
    all_results,
    all_results_sys,
    dataset_labels,
    png_naming,
    no_filtering_grid=None,
    no_filtering_grid_other=None,
    remove_source_label=None
):
    # Remove the specified source, if provided.
    if remove_source_label is not None:
        if isinstance(remove_source_label, (list, tuple, set)):
            labels_to_remove = remove_source_label
        else:
            labels_to_remove = [remove_source_label]

        for lbl in labels_to_remove:
            all_results.pop(lbl, None)
            all_results_sys.pop(lbl, None)

    # Extract all filter labels from the first remaining source.
    first_source = next(iter(all_results_sys.values()))
    filtering_methods = list(first_source.keys())

    # Build the (m_a, g_a) mesh
    ma_mesh, g_mesh = np.meshgrid(m_masked, g_masked, indexing='ij')

    for filter_label in filtering_methods:
        # -- 1) Compute mean Δχ² for this filter_label --
        mean_delta_chi2_grid = compute_mean_delta_chi2_grid(
            all_results=all_results_sys,
            dataset_labels=dataset_labels,
            filter_label=filter_label,
            p0_masked=p0_masked,
            ec_masked=ec_masked,
            remove_source_label=None
        )
        systematic_grid = (compute_mean_delta_chi2_grid(
            all_results=all_results,
            dataset_labels=dataset_labels,
            filter_label=filter_label,
            p0_masked=p0_masked,
            ec_masked=ec_masked,
            remove_source_label=None
        ) if all_results else None)

        # Set up colormap.
        vmin, vmax = -10, 25#int(np.min(mean_delta_chi2_grid))-1, int(np.max(mean_delta_chi2_grid))+1
        num_colors = 120
        boundaries = np.linspace(vmin, vmax, num_colors + 1)
        cmap = plt.get_cmap('gnuplot2', num_colors)
        norm = mcolors.BoundaryNorm(boundaries=boundaries, ncolors=num_colors, clip=True)

        ma_mesh, g_mesh = np.meshgrid(m_masked, g_masked, indexing='ij')

        plt.figure(figsize=(10, 6))
        heatmap = plt.pcolormesh(ma_mesh / 1e-9, g_mesh, mean_delta_chi2_grid,
                                 cmap=cmap, norm=norm, shading='auto')

        plot_specs = []
        # — Mean (filtered) —
        if mean_delta_chi2_grid is not None:
            plot_specs.append({
                'grid': mean_delta_chi2_grid,
                'label': 'With systematics',
                'color_pos': 'red',   'linestyle_pos': 'solid',
                'color_neg': 'lime',  'linestyle_neg': 'solid'
            })

        # — Mean (no filtering) —
        if no_filtering_grid is not None:
            plot_specs.append({
                'grid': no_filtering_grid,
                'label': 'No filtering',
                'color_neg': 'white', 'linestyle_neg': 'solid'
            })

        # — Systematics (filtered) —
        if systematic_grid is not None:
            plot_specs.append({
                'grid': systematic_grid,
                'label': 'No systematics',
                'color_pos': 'red',   'linestyle_pos': 'dashed',
                'color_neg': 'lime',  'linestyle_neg': 'dashed'
            })

        # — Systematics (no filtering) —
        if no_filtering_grid_other is not None:
            plot_specs.append({
                'grid': no_filtering_grid_other,
                'label': 'No filtering',
                'color_neg': 'white', 'linestyle_neg': 'dashed'
            })

        # --- Plot contours (remove label= from contour calls) ---
        for spec in plot_specs:
            grid = spec['grid']
            x = ma_mesh / 1e-9
            y = g_mesh

            if spec.get('color_pos') and np.any(grid >= 6.2):
                plt.contour(x, y, grid,
                            levels=[6.2],
                            colors=spec['color_pos'],
                            linestyles=spec['linestyle_pos'],
                            linewidths=2)

            if spec.get('color_neg') and np.any(grid <= -6.2):
                plt.contour(x, y, grid,
                            levels=[-6.2],
                            colors=spec['color_neg'],
                            linestyles=spec['linestyle_neg'],
                            linewidths=2)

        # --- Build proxy legend handles for both aspects ---
        
        if no_filtering_grid_other is not None: # 1. Legend for threshold (color) mapping:
            color_handles = [
                Line2D([0], [0], color='red', linestyle='-', linewidth=2, label='> 6.2'),
                Line2D([0], [0], color='lime', linestyle='-', linewidth=2, label='< -6.2'),
                # For "No filtering" the color is white. Use a marker with a black edge to make it visible.
                Line2D([0], [0], marker='s', markersize=8, markerfacecolor='white', 
                    markeredgecolor='black', linestyle='-', linewidth=2, label='No filtering (white)')
            ]
        else:
            color_handles = [
                Line2D([0], [0], color='red', linestyle='-', linewidth=2, label='> 6.2'),
                Line2D([0], [0], color='lime', linestyle='-', linewidth=2, label='< -6.2')
            ]


        # 2. Legend for systematics (line style) mapping:
        linestyle_handles = [
            Line2D([0], [0], color='black', linestyle='solid', linewidth=2, label='With systematics'),
            Line2D([0], [0], color='black', linestyle='dashed', linewidth=2, label='Without systematics')
        ]

        # Create the two legends. Add the first legend to the axes so that the second does not overwrite it.
        #legend1 = plt.legend(handles=color_handles, loc='upper left', title="Threshold")
        #plt.gca().add_artist(legend1)
        #plt.legend(handles=linestyle_handles, loc='lower left', title="Systematics")

        #data = np.load('fermi3_10_contours_b5.2_eta0.67_rescale.npz')
        #plt.errorbar(10**data['x1'][data['y1'] < 1e-11]/ 1e-9, 10**data['y1'][data['y1'] < 1e-11], color='b', ls=(0,(4,2,1,1,1,2)), alpha=1)

        ##### plot contours for 0 level of syst uncertainty ##################
        '''
        data = np.load('fermi0_0_contours_ebl_6.2_scan12_jointfit.npz')
        plt.errorbar(10**data['x'][data['y'] < 1e-11]/ 1e-9, 1.*10**data['y'][data['y'] < 1e-11], fmt='w:')
        plt.errorbar(10**data['x1'][data['y1'] < 1e-11]/ 1e-9, 1.*10**data['y1'][data['y1'] < 1e-11], fmt='w:')
        
        data = np.load('fermi3_10_contours_06032025.npz')
        print(10**data['x1']/ 1e-9, 10**data['y1'])
        plt.errorbar(10**data['x1']/ 1e-9, 10**data['y1'], fmt='w-')#, zorder=-10)
        '''

        cbar = plt.colorbar(heatmap, ticks=np.linspace(vmin, vmax, 11))
        cbar.set_label(r'$\sum \Delta \chi^2$', fontsize=15)
        plt.xlabel(r'$m_a$ [neV]', fontsize=15)
        plt.ylabel(r'$g_{a\gamma}$ [GeV$^{-1}$]', fontsize=15)
        plt.title(f'$\Delta \chi^2$ Heatmap for {filter_label}', fontsize=15)
        plt.xscale('log')
        plt.yscale('log')
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        #plt.gca().xaxis.set_major_formatter(ticker.FormatStrFormatter("%g"))
        ax = plt.gca()
        plt.xlim(0.3, 9)
        ax.set_xticks([1, 9])                  # put ticks at 10^0, 10^1
        ax.set_xticklabels(['1', '9']) 
        plt.tight_layout()
        plt.savefig(f'{path_to_save_heatmap_m_g}{png_naming}_{filter_label}_ma_ga.png', dpi=300)
        plt.close()

        print(f"Finished plotting for filter: {filter_label}")

################################################################################################# WITH systematic errors #######################################################################################

with open("none_new0_sys_error.pkl", "rb") as file:
    all_results_none_sys = pickle.load(file)

with open("lin_new0_sys_error.pkl", "rb") as file:
    all_results_lin_sys = pickle.load(file)

with open("snr_new0_sys_error.pkl", "rb") as file:
    all_results_snr_sys = pickle.load(file)

no_filtering_sources_sys = list(all_results_none_sys.keys()) 
with open("none_new0_no_sys_error.pkl", "rb") as file:
    all_results_none = pickle.load(file)

with open("lin_new0_no_sys_error.pkl", "rb") as file:
    all_results_lin = pickle.load(file)

with open("snr_new0_no_sys_error.pkl", "rb") as file:
    all_results_snr = pickle.load(file)

no_filtering_sources = list(all_results_none.keys())

#plot_individual_delta_chi2_heatmap_with_pdf(all_results_none_sys, no_filtering_sources_sys, all_results_none, "sys", filtering_methods="No_Filtering", pdf_filename="NEW_indv_heatmaps_no_filter_logpar_sys_error.pdf")
##plot_individual_delta_chi2_heatmap_with_pdf(all_results_lin_sys, no_filtering_sources_sys, all_results_lin, "sys", filtering_methods="week", pdf_filename="NEW_indv_heatmaps_week_logpar_sys_error.pdf")
#plot_individual_delta_chi2_heatmap_with_pdf(all_results_lin_sys, no_filtering_sources_sys, all_results_lin, "sys", filtering_methods="month", pdf_filename="NEW_indv_heatmaps_month_logpar_sys_error.pdf")

#plot_individual_delta_chi2_heatmap_with_pdf(all_results_snr_sys, no_filtering_sources_sys, all_results_snr, "sys", filtering_methods="snr_3", pdf_filename="NEW_indv_heatmaps_snr3_logpar_sys_error.pdf")
#plot_individual_delta_chi2_heatmap_with_pdf(all_results_snr_sys, no_filtering_sources_sys, all_results_snr, "sys", filtering_methods="snr_5", pdf_filename="NEW_indv_heatmaps_snr5_logpar_sys_error.pdf")
#plot_individual_delta_chi2_heatmap_with_pdf(all_results_snr_sys, no_filtering_sources_sys, all_results_snr, "sys", filtering_methods="snr_10", pdf_filename="NEW_indv_heatmaps_snr10_logpar_sys_error.pdf")

################################################################################################# NO systematic errors #######################################################################################



#plot_individual_delta_chi2_heatmap_with_pdf(all_results_none, no_filtering_sources, all_results_none_sys, "nosys", filtering_methods="No_Filtering", pdf_filename="NEW_indv_heatmaps_no_filter_logpar_no_sys_error.pdf")
#plot_individual_delta_chi2_heatmap_with_pdf(all_results_lin, no_filtering_sources, all_results_lin_sys, "nosys", filtering_methods="week", pdf_filename="NEW_indv_heatmaps_week_logpar_no_sys_error.pdf")
#plot_individual_delta_chi2_heatmap_with_pdf(all_results_lin, no_filtering_sources, all_results_lin_sys, "nosys", filtering_methods="month", pdf_filename="NEW_indv_heatmaps_month_logpar_no_sys_error.pdf")
#plot_individual_delta_chi2_heatmap_with_pdf(all_results_snr, no_filtering_sources, all_results_snr_sys, "nosys", filtering_methods="snr_3", pdf_filename="NEW_indv_heatmaps_snr3_logpar_no_sys_error.pdf")
#plot_individual_delta_chi2_heatmap_with_pdf(all_results_snr, no_filtering_sources, all_results_snr_sys, "nosys", filtering_methods="snr_5", pdf_filename="NEW_indv_heatmaps_snr5_logpar_no_sys_error.pdf")
#plot_individual_delta_chi2_heatmap_with_pdf(all_results_snr, no_filtering_sources, all_results_snr_sys, "nosys", filtering_methods="snr_10", pdf_filename="NEW_indv_heatmaps_snr10_logpar_no_sys_error.pdf")

print('Plotting summed chi-squared heatmap!') # e.g. ["No_Filtering"] or sometimes multiple sources

def plot_delta_chi2_heatmap_nosys_base(
    all_results,
    all_results_sys,
    dataset_labels,
    png_naming,
    no_filtering_grid=None,
    no_filtering_grid_other=None,
    remove_source_label=None
):
    # Assumes these are defined in the calling scope:
    # m_masked, g_masked, p0_masked, ec_masked, path_to_save_heatmap_m_g

    # Extract filter labels
    first_source = next(iter(all_results.values()))
    filtering_methods = list(first_source.keys())

    # Define filter-based source removals
    filter_source_removals = {
        "No_Filtering": ["4FGL J0317.8-4414"],
        "week": ["4FGL J0317.8-4414"],
        "month": ["4FGL J0132.7-0804","4FGL J0317.8-4414", "4FGL J1242.9+7315"],
        "snr_3": ["4FGL J0317.8-4414", "4FGL J1516.8+2918"],
        "snr_5": ["4FGL J0132.7-0804", "4FGL J0317.8-4414", "4FGL J0912.5+1556", "4FGL J1516.8+2918"],
        "snr_10": ["4FGL J0132.7-0804", "4FGL J0317.8-4414" ,"4FGL J1213.0+5129"]
    }

    # Remove specified sources
    for filter_label in filtering_methods:
        labels_to_remove = filter_source_removals.get(filter_label, [])[:]
        if remove_source_label is not None:
            if isinstance(remove_source_label, (list, tuple, set)):
                labels_to_remove.extend(remove_source_label)
            else:
                labels_to_remove.append(remove_source_label)
        for lbl in labels_to_remove:
            all_results.pop(lbl, None)
            all_results_sys.pop(lbl, None)

    # Refresh filter labels after removals
    first_source = next(iter(all_results.values()))
    filtering_methods = list(first_source.keys())

    # Precompute mesh
    ma_mesh, g_mesh = np.meshgrid(m_masked, g_masked, indexing='ij')

    # Ensure output directory exists
    outdir = path_to_save_heatmap_m_g
    os.makedirs(outdir, exist_ok=True)

    for filter_label in filtering_methods:
        # Compute grids
        mean_delta_chi2_grid = compute_mean_delta_chi2_grid(
            all_results=all_results,
            dataset_labels=dataset_labels,
            filter_label=filter_label,
            p0_masked=p0_masked,
            ec_masked=ec_masked,
            remove_source_label=None
        )
        systematic_grid = (compute_mean_delta_chi2_grid(
            all_results=all_results_sys,
            dataset_labels=dataset_labels,
            filter_label=filter_label,
            p0_masked=p0_masked,
            ec_masked=ec_masked,
            remove_source_label=None
        ) if all_results_sys else None)

        # Save contour points as text files
        x = ma_mesh / 1e-9
        y = g_mesh

        # Without systematics
        if mean_delta_chi2_grid is not None:
            cs_nosys = plt.contour(
                x, y, mean_delta_chi2_grid,
                levels=[-6.2, 6.2],
                colors=['lime', 'cyan'],
                linestyles=['solid', 'solid'],
                linewidths=2
            )
            cs_nosys_fortxt = plt.contour(
                x, y, mean_delta_chi2_grid,
                levels=[6.2],  # <- only this level
                colors=['cyan'],
                linestyles=['solid'],
                linewidths=2
            )
            verts = []
            if cs_nosys_fortxt.levels[0] == 6.2:
                for path in cs_nosys_fortxt.collections[0].get_paths():
                    verts.append(path.vertices)
            all_verts_nosys = np.vstack(verts) if verts else np.empty((0, 2))
            if all_verts_nosys.size > 0:
                np.savetxt(
                    os.path.join(outdir, f"{filter_label}_nosys.txt"),
                    all_verts_nosys,
                    header="x [neV]    y [GeV^-1]"
                )

        # With systematics
        if systematic_grid is not None:
            cs_withsys = plt.contour(
                x, y, systematic_grid,
                levels=[-6.2, 6.2],
                colors=['red', 'green'],
                linestyles=['dashed', 'dashed'],
                linewidths=2
            )
            cs_withsys_fortxt = plt.contour(
                x, y, systematic_grid,
                levels=[6.2],  # <- only this level
                colors=['green'],
                linestyles=['dashed'],
                linewidths=2
            )
            verts = []
            if cs_withsys_fortxt.levels[0] == 6.2:
                for path in cs_withsys_fortxt.collections[0].get_paths():
                    verts.append(path.vertices)
            all_verts_withsys = np.vstack(verts) if verts else np.empty((0, 2))
            if all_verts_withsys.size > 0:
                np.savetxt(
                    os.path.join(outdir, f"{filter_label}_withsys.txt"),
                    all_verts_withsys,
                    header="x [neV]    y [GeV^-1]"
                )

        # Plot heatmap
        plt.figure(figsize=(10, 6))
        cmap = plt.get_cmap('gnuplot2', 120)
        norm = mcolors.BoundaryNorm(np.linspace(-10, 25, 121), ncolors=120, clip=True)
        heatmap = plt.pcolormesh(
            x, y, mean_delta_chi2_grid,
            cmap=cmap, norm=norm, shading='auto'
        )

        # Plot contours again for visualization
        # Build specs
        plot_specs = []
        if mean_delta_chi2_grid is not None:
            plot_specs.append({'grid': mean_delta_chi2_grid, 'color_pos': 'cyan', 'linestyle_pos': 'solid', 'color_neg': 'lime', 'linestyle_neg': 'solid'})
        if no_filtering_grid is not None:
            plot_specs.append({'grid': no_filtering_grid, 'color_neg': 'white', 'linestyle_neg': 'dotted', 'color_pos': 'black', 'linestyle_pos': 'dotted'})
        if systematic_grid is not None:
            plot_specs.append({'grid': systematic_grid, 'color_pos': 'green', 'linestyle_pos': 'dashed', 'color_neg': 'red', 'linestyle_neg': 'dashed'})
        if no_filtering_grid_other is not None:
            plot_specs.append({'grid': no_filtering_grid_other, 'color_neg': 'white', 'linestyle_neg': 'dashed', 'color_pos': 'black', 'linestyle_pos': 'dashed'})

        for spec in plot_specs:
            grid = spec['grid']
            if spec.get('color_pos') and np.any(grid >= 6.2):
                plt.contour(x, y, grid, levels=[6.2], colors=spec['color_pos'], linestyles=spec['linestyle_pos'], linewidths=2)
            if spec.get('color_neg') and np.any(grid <= -6.2):
                plt.contour(x, y, grid, levels=[-6.2], colors=spec['color_neg'], linestyles=spec['linestyle_neg'], linewidths=2)

        # Colorbar and labels
        cbar = plt.colorbar(heatmap, ticks=np.linspace(-10, 25, 11))
        cbar.set_label(r'$\sum \Delta \chi^2$')
        plt.xlabel(r'$m_a$ [neV]')
        plt.ylabel(r'$g_{a\gamma}$ [GeV$^{-1}$]')

        # Title
        title_map = {
            "No_Filtering": "No filtering",
            "week": "Weekly filter",
            "month": "Monthly filter",
            "snr_3": "SNR=3 filter",
            "snr_5": "SNR=5 filter",
            "snr_10": "SNR=10 filter"
        }
        #plt.title(title_map.get(filter_label, filter_label))
        '''if filter_label == "No_Filtering":
            data = np.load('fermi0_0_contours_ebl_6.2_scan12_jointfit.npz')
            plt.errorbar(10**data['x'][data['y'] < 1e-11]/ 1e-9, 1.*10**data['y'][data['y'] < 1e-11], fmt='k:')
            plt.errorbar(10**data['x1'][data['y1'] < 1e-11]/ 1e-9, 1.*10**data['y1'][data['y1'] < 1e-11], fmt='k:')
            
            data = np.load('fermi3_10_contours_06032025.npz')
            print(10**data['x1']/ 1e-9, 10**data['y1'])
            plt.errorbar(10**data['x1']/ 1e-9, 10**data['y1'], fmt='k-')#, zorder=-10)'''
        # Log scales and ticks
        
        plt.xscale('log')
        plt.yscale('log')
        ax = plt.gca()
        plt.axhspan(3.5e-12, ax.get_ylim()[1], color='grey', alpha=0.6, zorder=10)
        plt.tick_params(axis='both', which='both', color='white', labelcolor='black', direction='in', top=True, right=True)
        plt.xlim(0.3, 9)
        ax.set_xticks([1, 9])
        ax.set_xticklabels(['1', '9'])

        # Save figure
        plt.tight_layout()
        plt.savefig(f'{outdir}/{png_naming}_{filter_label}_ma_ga.png', dpi=300)
        plt.close()

        print(f"Finished plotting for filter: {filter_label}")

'''
no_filtering_grid_sys = compute_mean_delta_chi2_grid(
    all_results=all_results_none_sys,
    dataset_labels=no_filtering_sources_sys,
    filter_label="No_Filtering",
    p0_masked=p0_masked,
    ec_masked=ec_masked,
    remove_source_label = "4FGL J0317.8-4414"
) 

no_filtering_grid = compute_mean_delta_chi2_grid(
    all_results=all_results_none,
    dataset_labels=no_filtering_sources,
    filter_label="No_Filtering",
    p0_masked=p0_masked,
    ec_masked=ec_masked,
    remove_source_label = "4FGL J0317.8-4414" #7

)
'''
plot_delta_chi2_heatmap_nosys_base(all_results_none, all_results_none_sys, list(all_results_none.keys()), "base_nosys_", remove_source_label=["4FGL J0317.8-4414"])


# Summed heatmaps for no filter
#plot_mean_delta_chi2_heatmap_sys_base(all_results_none, all_results_none_sys, list(all_results_none.keys()), "base_sys_", no_filtering_grid_other=None, remove_source_label="4FGL J0317.8-4414")
#plot_mean_delta_chi2_heatmap_nosys_base(all_results_none, all_results_none_sys, list(all_results_none.keys()), "base_nosys_",no_filtering_grid_other=None,  remove_source_label="4FGL J0317.8-4414")

# Summed heatmaps for month and week
'''no_filtering_grid_sys_week = compute_mean_delta_chi2_grid(
    all_results=all_results_none_sys,
    dataset_labels=no_filtering_sources_sys,
    filter_label="No_Filtering",
    p0_masked=p0_masked,
    ec_masked=ec_masked,
    remove_source_label = ["4FGL J0317.8-4414"]
) 

no_filtering_grid_week = compute_mean_delta_chi2_grid(
    all_results=all_results_none,
    dataset_labels=no_filtering_sources,
    filter_label="No_Filtering",
    p0_masked=p0_masked,
    ec_masked=ec_masked,
    remove_source_label = ["4FGL J0317.8-4414"] #7

) 
no_filtering_grid_month = compute_mean_delta_chi2_grid(
    all_results=all_results_none,
    dataset_labels=no_filtering_sources,
    filter_label="No_Filtering",
    p0_masked=p0_masked,
    ec_masked=ec_masked,
    remove_source_label = ["4FGL J0132.7-0804","4FGL J0317.8-4414", "4FGL J1242.9+7315"]) #7 and 12
no_filtering_grid_sys_month = compute_mean_delta_chi2_grid(
    all_results=all_results_none_sys,
    dataset_labels=no_filtering_sources_sys,
    filter_label="No_Filtering",
    p0_masked=p0_masked,
    ec_masked=ec_masked,
    remove_source_label = ["4FGL J0132.7-0804", "4FGL J0317.8-4414", "4FGL J1242.9+7315"]) #4, 7 and 12'''


##plot_delta_chi2_heatmap_nosys_base(all_results_lin, all_results_lin_sys, list(all_results_lin.keys()), "base_nosys_", no_filtering_grid=None , no_filtering_grid_other=None, remove_source_label=["4FGL J0317.8-4414"])
#plot_delta_chi2_heatmap_nosys_base(all_results_lin, all_results_lin_sys, list(all_results_lin.keys()), "base_nosys_", no_filtering_grid=None , no_filtering_grid_other=None, remove_source_label=["4FGL J0132.7-0804","4FGL J0317.8-4414", "4FGL J1242.9+7315"])

##plot_mean_delta_chi2_heatmap_sys_base(all_results_lin, all_results_lin_sys, list(all_results_lin.keys()), "base_sys_", no_filtering_grid=no_filtering_grid_sys,no_filtering_grid_other=None, remove_source_label=["4FGL J1242.9+7315", "4FGL J0912.5+1556", "4FGL J1516.8+2918"])
#plot_mean_delta_chi2_heatmap_nosys_base(all_results_lin, all_results_lin_sys, list(all_results_lin.keys()), "base_nosys_", no_filtering_grid=[no_filtering_grid_week, no_filtering_grid_month] , no_filtering_grid_other=[no_filtering_grid_sys_week, no_filtering_grid_sys_month], remove_source_label=["4FGL J0317.8-4414", ["4FGL J0317.8-4414", "4FGL J1242.9+7315"]])

#plot_delta_chi2_heatmap_nosys_base(all_results_snr, all_results_snr_sys, list(all_results_snr.keys()), "base_nosys_", no_filtering_grid=None, no_filtering_grid_other=None,remove_source_label=True)
# summed heatmaps for snr
'''no_filtering_grid_sys = compute_mean_delta_chi2_grid(
    all_results=all_results_none_sys,
    dataset_labels=no_filtering_sources_sys,
    filter_label="No_Filtering",
    p0_masked=p0_masked,
    ec_masked=ec_masked,
    remove_source_label = ["4FGL J1242.9+7315", "4FGL J1516.8+2918"]
) 

no_filtering_grid = compute_mean_delta_chi2_grid(
    all_results=all_results_none,
    dataset_labels=no_filtering_sources,
    filter_label="No_Filtering",
    p0_masked=p0_masked,
    ec_masked=ec_masked,
    remove_source_label = ["4FGL J1242.9+7315", "4FGL J1516.8+2918"]

) '''
#plot_mean_delta_chi2_heatmap_nosys_base(all_results_snr, all_results_snr_sys, list(all_results_snr.keys()), "base_nosys_", no_filtering_grid=no_filtering_grid, no_filtering_grid_other=None,remove_source_label=["4FGL J1242.9+7315", "4FGL J1516.8+2918"])
##plot_mean_delta_chi2_heatmap_sys_base(all_results_snr, all_results_snr_sys, list(all_results_snr.keys()), "base_sys_", no_filtering_grid=no_filtering_grid_sys, no_filtering_grid_other=None, remove_source_label=["4FGL J1242.9+7315", "4FGL J1516.8+2918"])


'''
plot_mean_delta_chi2_heatmap_nosys_base(all_results_none, all_results_none_sys, list(all_results_none.keys()), "base_nosys_",  remove_source_label=None)
plot_mean_delta_chi2_heatmap_nosys_base(all_results_lin, all_results_lin_sys, list(all_results_lin.keys()), "base_nosys_", no_filtering_grid=no_filtering_grid, no_filtering_grid_other=no_filtering_grid_sys, remove_source_label=None)
plot_mean_delta_chi2_heatmap_nosys_base(all_results_snr, all_results_snr_sys, list(all_results_snr.keys()), "base_nosys_", no_filtering_grid=no_filtering_grid, no_filtering_grid_other=no_filtering_grid_sys,remove_source_label=None)

plot_mean_delta_chi2_heatmap_sys_base(all_results_none, all_results_none_sys, list(all_results_none.keys()), "base_sys_",  remove_source_label=None)
plot_mean_delta_chi2_heatmap_sys_base(all_results_lin, all_results_lin_sys, list(all_results_lin.keys()), "base_sys_", no_filtering_grid=no_filtering_grid_sys,no_filtering_grid_other=no_filtering_grid, remove_source_label=None)
plot_mean_delta_chi2_heatmap_sys_base(all_results_snr, all_results_snr_sys, list(all_results_snr.keys()), "base_sys_", no_filtering_grid=no_filtering_grid_sys, no_filtering_grid_other=no_filtering_grid, remove_source_label=None)

'''
def split_clusters(xs, ys, threshold=1.0):
    """
    Split a sequence of points into clusters whenever the consecutive distance exceeds threshold.
    Returns list of (x_cluster, y_cluster).
    """
    pts = np.column_stack((xs, ys))
    if pts.shape[0] < 2:
        return []
    deltas = np.diff(pts, axis=0)
    dists = np.hypot(deltas[:,0], deltas[:,1])
    breaks = np.where(dists > threshold)[0]
    clusters = []
    start = 0
    for b in breaks:
        end = b + 1
        cluster = pts[start:end]
        if cluster.shape[0] > 1:
            clusters.append((cluster[:,0], cluster[:,1]))
        start = end
    last = pts[start:]
    if last.shape[0] > 1:
        clusters.append((last[:,0], last[:,1]))
    return clusters


def compute_and_plot_contours(
    all_results_none,
    all_results_none_sys,
    all_results_lin,
    all_results_lin_sys,
    dataset_labels_none,
    dataset_labels_lin,
    m_masked,
    g_masked,
    p0_masked,
    ec_masked,
    remove_sources,
    output_prefix,
    threshold=1.0
):
    """
    Remove sources uniformly, compute and plot ±6.2 Δχ² contours for three filters:
      - No_Filtering uses none datasets
      - week/month use linear datasets
    Produces two plots: one without systematics (_nosys) and one with systematics (_withsys).
    """
    import matplotlib.pyplot as plt
    
    # Prune unwanted sources from all datasets
    def prune(results):
        return {k: v for k, v in results.items() if k not in remove_sources}

    none = prune(all_results_none)
    none_sys = prune(all_results_none_sys)
    lin = prune(all_results_lin)
    lin_sys = prune(all_results_lin_sys)

    # Create meshgrid
    ma_mesh, g_mesh = np.meshgrid(m_masked, g_masked, indexing='ij')
    x = ma_mesh / 1e-9
    y = g_mesh

    # Plot settings
    if output_prefix == "LIN":
        filters = ['No_Filtering', 'week', 'month']
        colors = {'No_Filtering':'black', 'week':'olivedrab', 'month':'darkred'}
        alphas = {'No_Filtering':0.9, 'week':0.75, 'month':0.75}
    elif output_prefix == "SNR":
        filters = ['No_Filtering', 'snr_3','snr_5', 'snr_10']
        colors = {'No_Filtering':'black', "snr_3":"tab:green", 'snr_5':'b', 'snr_10':'tab:orange'}
        alphas = {'No_Filtering':0.9, 'snr_3':0.75, 'snr_5':0.75, 'snr_10':0.75}

    # Loop over mode: nosys and withsys
    for mode_label, (res_none_dict, res_lin_dict) in {
        'nosys': (none, lin),
        'withsys': (none_sys, lin_sys)
    }.items():
        fig, ax = plt.subplots(figsize=(8,6))
        for fl in filters:
            # Select appropriate result dict and dataset_labels
            if fl == 'No_Filtering':
                res_dict = res_none_dict
                ds_labels = dataset_labels_none
            else:
                res_dict = res_lin_dict
                ds_labels = dataset_labels_lin

            # Compute Δχ² grid for this filter and mode
            grid = compute_mean_delta_chi2_grid(
                res_dict, ds_labels, fl,
                p0_masked, ec_masked, remove_source_label=None
            )

            # Determine linestyle mapping per filter
            if fl == 'No_Filtering':
                linestyles = ['dashdot', 'solid']  # [-6.2, +6.2]
            elif fl == 'week':
                linestyles = [(0, (1, 1)), 'dashed']
            elif fl == 'month':
                linestyles = ['dotted', 'dashdot']
            elif fl == 'snr_3':
                linestyles = [(0, (1, 1)), 'dashed']
            elif fl == 'snr_5':
                linestyles = ['dotted',(0, (5,5))]
            elif fl == 'snr_10':
                linestyles = [(0, (1, 5)), (0, (5,10))]
            else:
                linestyles = ['dashed', 'solid']

            # Extract and plot contours at ±6.2
            ax.contour(
                x, y, grid,
                levels=[-6.2, 6.2],
                colors=[colors[fl]] * 2,
                linestyles=linestyles,
                linewidths=2,
                alpha=alphas[fl]
            )
        # Axis formatting
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlim(0.3, 9)
        ax.set_xticks([1, 9])
        ax.set_xticklabels(['1', '9'])
        ax.set_xlabel(r'$m_a$ [neV]')
        ax.set_ylabel(r'$g_{a\gamma}$ [GeV$^{-1}$]')
        plt.tight_layout()
        fig.savefig(f"{output_prefix}_{mode_label}.png", dpi=300)
        plt.close(fig)

outdir = path_to_save_heatmap_m_g
'''compute_and_plot_contours(all_results_none,
                        all_results_none_sys,
                        all_results_lin,
                        all_results_lin_sys,
                        list(all_results_none.keys()),
                        list(all_results_lin.keys()),
                        m_masked,
                        g_masked,
                        p0_masked,
                        ec_masked,
                        remove_sources=["4FGL J0132.7-0804", "4FGL J0317.8-4414", "4FGL J1242.9+7315"],
                        output_prefix="LIN",
                        threshold=1.0
                    )
compute_and_plot_contours(all_results_none,
                        all_results_none_sys,
                        all_results_snr,
                        all_results_snr_sys,
                        list(all_results_none.keys()),
                        list(all_results_snr.keys()),
                        m_masked,
                        g_masked,
                        p0_masked,
                        ec_masked,
                        remove_sources=["4FGL J0132.7-0804", "4FGL J0317.8-4414", "4FGL J0912.5+1556", "4FGL J1213.0+5129", "4FGL J1516.8+2918"],
                        output_prefix="SNR",
                        threshold=1.0
                    )'''