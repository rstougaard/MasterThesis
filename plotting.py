import json
import matplotlib.pyplot as plt
import numpy as np

time_interval_name = 'month'
num_time_intervals = 2
# Post-processing and plotting results
fluxes = []
flux_errors = []

month_name = ['Aug', 'Sep', 'Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'March', 'April', 'May', 'June', 'July']
# Output the flux and error for each month
if time_interval_name == 'month':
    print("Plotting Monthly Light Curve")
    months = np.arange(1, num_time_intervals + 1)
    

    def get_month_cyclic(index):
        return month_name[index % len(month_name)]

    # Load flux and flux error from each month's JSON file
    month_names = []
    for i in range(num_time_intervals):
        with open(f'./data/LC_{time_interval_name}/likeresults/flux_{time_interval_name}_{i}.json', 'r') as f:
            data = json.load(f)
            fluxes.append(data['flux'])
            flux_errors.append(data['flux_error'])
            month_names.append(get_month_cyclic(i))

    fluxes = np.array(fluxes)
    flux_errors = np.array(flux_errors)

    for month, flux, flux_error in zip(month_names, fluxes, flux_errors):
        print(f'Month: {month}, Flux: {flux}, Flux Error: {flux_error}')

    # Plotting the light curve
    plt.figure(figsize=(10, 6))
    plt.errorbar(month_names, fluxes, yerr=flux_errors, fmt='o', capsize=5, label='Flux with Error')
    plt.xlabel('Time (months)', fontsize=14)
    plt.ylabel('Flux (ph/cm²/s)', fontsize=14)
    plt.title(f'Flux of 4FGL J0319.8+4130 as a function of time (per {time_interval_name})', fontsize=16)
    #plt.yscale('log')
    plt.grid(True)
    plt.legend()
    plt.savefig(f'Light_Curve_{time_interval_name}.png')

    
