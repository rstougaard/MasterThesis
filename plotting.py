import json
import matplotlib.pyplot as plt
import numpy as np

time_interval_name = 'month'
num_time_intervals = 24  # Assume you have 24 months of data (2 years)
# Post-processing and plotting results
fluxes = []
flux_errors = []

month_name = ['Aug', 'Sep', 'Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'March', 'April', 'May', 'June', 'July']

# Define the starting year
start_year = 2008

# Output the flux and error for each month
if time_interval_name == 'month':
    print("Plotting Monthly Light Curve")
    months = np.arange(1, num_time_intervals + 1)

    def get_month_and_year(index):
        """ Return month and corresponding year based on index. """
        month = month_name[index % len(month_name)]  # Get month cyclically
        year = start_year + (index // len(month_name))  # Calculate year
        return f'{month} {year}'

    # Load flux and flux error from each month's JSON file
    month_names = []
    for i in range(num_time_intervals):
        with open(f'./data/LC_{time_interval_name}/likeresults/flux_{time_interval_name}_{i}.json', 'r') as f:
            data = json.load(f)
            fluxes.append(data['flux'])
            flux_errors.append(data['flux_error'])
            month_names.append(get_month_and_year(i))

    fluxes = np.array(fluxes)
    flux_errors = np.array(flux_errors)

    # Calculate the mean flux
    mean_flux = np.mean(fluxes)

    for month, flux, flux_error in zip(month_names, fluxes, flux_errors):
        print(f'Month: {month}, Flux: {flux}, Flux Error: {flux_error}')

    # Plotting the light curve
    plt.figure(figsize=(12, 6))
    plt.errorbar(month_names, fluxes, yerr=flux_errors, fmt='o', capsize=5, label='Flux with Error')

    # Plot the mean flux line
    plt.axhline(mean_flux, color='red', linestyle='--', label=f'Mean Flux: {mean_flux:.2e}')

    # Highlight the months where the flux is higher than the mean
    higher_flux_months = np.where(fluxes > mean_flux)[0]
    for idx in higher_flux_months:
        plt.annotate(month_names[idx], (month_names[idx], fluxes[idx]), xytext=(0, 5), textcoords='offset points', color='red')

    plt.xlabel('Time (months)', fontsize=14)
    plt.ylabel('Flux (ph/cm²/s)', fontsize=14)
    plt.title(f'Flux of 4FGL J0319.8+4130 as a function of time (per {time_interval_name})', fontsize=16)
    plt.xticks(rotation=45)  # Rotate month labels to avoid overlap
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'Light_Curve_{time_interval_name}.png')
    plt.show()

    # Print out months with flux higher than mean
    print("\nMonths with flux higher than mean:")
    for idx in higher_flux_months:
        print(f'{month_names[idx]}: Flux = {fluxes[idx]:.2e}')
