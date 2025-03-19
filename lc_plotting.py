import matplotlib.pyplot as plt
from astropy.io import fits
import numpy as np

def lc_plotting(vars, snrratios=None, time_intervals=None):
    # Extract variables
    source_name, ra, dec, method, specin, _, _, minimal_energy, maximal_energy = vars
    source_name_cleaned = source_name.replace(" ", "").replace(".", "dot").replace("+", "plus").replace("-", "minus")
      # Determine the loop items based on the method
    loop_items = snrratios if method == "SNR" else time_intervals
    
    initial_means = []
    final_means = []
    final_thresholds = []
    valid_intervals_list = []
    invalid_intervals_list = []
    
    if method == 'SNR':
        colors = ['blue', 'orange', 'green']
    elif method == 'LIN':
        colors = ['purple', 'brown']


    for loop_item, color in zip(loop_items, colors):
        if method == 'SNR':
            lc = f'./data/{source_name_cleaned}/{method}/lc_snr{loop_item}.fits'
            plot_file = f'./data/{source_name_cleaned}/{method}/lc_snr{loop_item}.png'
            
        elif method == 'LIN':
            lc = f'./data/{source_name_cleaned}/{method}/lc_{loop_item}.fits'
            plot_file = f'./data/{source_name_cleaned}/{method}/lc_{loop_item}.png'
                
        f_bin = fits.open(lc)
        bin_data = f_bin[1].data

        # Extract data
        X_bin = bin_data['TIME']
        Y_bin = bin_data['COUNTS'] / bin_data['EXPOSURE']  # Flux in photons/cm²/s
        time_intervals = bin_data['TIMEDEL']  # Duration of each time interval
        x_error_bin = bin_data['TIMEDEL'] / 2
        y_error_bin = bin_data['ERROR'] / bin_data['EXPOSURE']

        # Initialize filtering
        filtered_Y = Y_bin.copy()
        filtered_mask = np.full(Y_bin.shape, True)
        thresholds = []
        round_means = []
        removed_points = []

        while True:
            mean = np.mean(filtered_Y[filtered_mask])
            threshold = mean + 2 * np.std(filtered_Y[filtered_mask])
            thresholds.append(threshold)
            round_means.append(mean)

            round_removed_mask = filtered_Y > threshold
            new_filtered_mask = filtered_mask & ~round_removed_mask
            removed_points.append((X_bin[round_removed_mask], filtered_Y[round_removed_mask]))

            if np.array_equal(new_filtered_mask, filtered_mask):
                break

            filtered_mask = new_filtered_mask

        # Save final results for this file
        initial_means.append(round_means[0])
        final_means.append(round_means[-1])
        final_thresholds.append(thresholds[-1])
        valid_intervals_list.append(np.sum(filtered_mask))
        invalid_intervals_list.append(len(filtered_mask) - np.sum(filtered_mask))

        # Save flare intervals to a text file
        flare_intervals_start = X_bin[~filtered_mask] - x_error_bin[~filtered_mask]
        flare_intervals_stop = X_bin[~filtered_mask] + x_error_bin[~filtered_mask]
        flare_intervals = np.column_stack((flare_intervals_start, flare_intervals_stop))


        #np.savetxt(output_file_flares, flare_intervals, delimiter=' ')
        #print(f"File saved as: {output_file_flares}")

            # Create a new figure for this file
        plt.figure(figsize=(10, 6))
        plt.rcParams["font.family"] = "serif"
        plt.rcParams["mathtext.fontset"] = "cm"

        # Plot original data
        plt.errorbar(X_bin, Y_bin, xerr=x_error_bin, yerr=y_error_bin, fmt='o', capsize=5, color=color, alpha=0.3, label=f'Data')

        # Highlight removed points
        for i, (removed_x, removed_y) in enumerate(removed_points):
            plt.scatter(removed_x, removed_y, color='black', edgecolors='black')

        # Plot means and thresholds for each round
        plt.axhline(round_means[0], color='grey', linestyle='--', linewidth=3, alpha=1, label=f'Mean unfiltered')
        plt.axhline(thresholds[0], color='grey', linestyle='-', linewidth=3, alpha=1, label=f'Threshold unfiltered')
        plt.axhline(round_means[-1], color='black', linestyle='--', linewidth=3, alpha=1, label=f'Mean round {len(round_means)}')
        plt.axhline(thresholds[-1], color='black', linestyle='-', linewidth=3, alpha=1, label=f'Threshold round {len(round_means)}')

        # Customize plot
        plt.ylabel('Flux [photons/cm²/s]',fontsize=18)
        plt.xlabel('Time [s]', fontsize=18)
        plt.title(f'Lightcurve for {method}: {loop_item}', fontsize=20)
        plt.xscale('log')
        plt.yscale('log')
        plt.grid(True, which="both", linestyle="--", linewidth=0.5)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)

        # Move the legend outside the plot
        plt.legend(
            fontsize=16,
            ncol=1,  # Number of columns in the legend
            loc='upper left',  # Position the legend to the left center of the bounding box
            frameon=True,  # Add a box around the legend
        )

        # Save the plot or display it

        plt.tight_layout()  # Adjust layout to prevent clipping
        plt.savefig(plot_file, bbox_inches='tight', dpi=300)  # Save with adjusted bounding box
        print(f"Plot saved as: {plot_file}")

    return
vars_snr = ("4FGL J0319.8+413", None, None, "SNR", None, None, None, 100, 1000000)
vars_lin = ("4FGL J0319.8+413", None, None, "LIN", None, None, None, 100, 1000000)
snrratios = [10, 5, 3]
time_intervals = ["week","month"]
lc_plotting(vars_snr, snrratios=snrratios)
lc_plotting(vars_lin, snrratios=snrratios)