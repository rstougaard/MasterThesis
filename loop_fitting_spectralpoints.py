from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import shlex
from iminuit.cost import LeastSquares
from iminuit import Minuit
from matplotlib.backends.backend_pdf import PdfPages
import pickle

plt.rcParams.update({
    'font.family': 'serif',
    'mathtext.fontset': 'cm',
    'font.size': 18,
    'axes.titlesize': 20,
    'axes.labelsize': 20,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'legend.fontsize': 16
})
def GetCatalogueSpectrum(nn):
    with fits.open('test/gll_psc_v35.fit') as f:
        data = f[1].data
        ebounds = f[5].data
        emin = np.unique( ebounds['LowerEnergy'] )
        emax = np.unique( ebounds['UpperEnergy'] )
            
    names4fgl = data['Source_Name']
        
    eav = (emin*emax)**0.5
    de1 = eav - emin
    de2 = emax - eav
        
    ok = np.where(names4fgl==nn)

    fl = data['nuFnu_Band'][ok][0] #erg/cm2/s
    ratio0 = data['Unc_Flux_Band'][ok][0][:,0] / data['Flux_Band'][ok][0]
    ratio1 = data['Unc_Flux_Band'][ok][0][:,1] / data['Flux_Band'][ok][0]

    dfl1 = -fl*ratio0
    dfl2 = fl*ratio1

    dfl = np.maximum(dfl1, dfl2) #+ systematics*fl #add systematics
    
    ok = fl>0#1e-13
    
    return eav[ok], fl[ok], dfl[ok], [de1[ok],de2[ok]] # flux to erg/cm2/s


def simple_plot(dataset_none, dataset_snr, colors_snr, dataset_lin, colors_lin, source, with_cat=False):
    # Create a new figure
    fig = plt.figure(figsize=(10, 12))
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["mathtext.fontset"] = "cm"
    
    eav0, f0, df0, de0 = GetCatalogueSpectrum(source)
    # Top subplot: Spectrum - SNR Ratios
    ax1 = fig.add_subplot(2, 1, 1)
    
    # Plot the "none" dataset (using black)
    for dataset_label, (x, y, y_err, emin, emax) in dataset_none.items():
        x, y, y_err, emin, emax = np.array(x), np.array(y), np.array(y_err), np.array(emin), np.array(emax)
        e_lowers = x - emin
        e_uppers = emax - x
        #bin_size = emax - emin
        ax1.errorbar(x, y, xerr=[e_lowers, e_uppers], yerr=y_err,
                     fmt='s', capsize=5, color='black', label=f'{dataset_label}')
        
    
    # Plot the SNR datasets with their corresponding colors
    for i, (dataset_label, (x, y, y_err, emin, emax)) in enumerate(dataset_snr.items()):
        x, y, y_err, emin, emax = np.array(x), np.array(y), np.array(y_err), np.array(emin), np.array(emax)
        e_lowers = x - emin
        e_uppers = emax - x
        #bin_size = emax - emin
        # Get the color based on the index (defaulting to black if index is out of range)
        color = colors_snr[i] if i < len(colors_snr) else 'black'
        ax1.errorbar(x, y, xerr=[e_lowers, e_uppers], yerr=y_err,
                     fmt='o', capsize=5, color=color, label=f'{dataset_label}')
    if with_cat == True:    
        ax1.errorbar(eav0, f0, yerr=df0, xerr=de0, fmt='o', color="pink", label='Catalogue Spectrum')
        
    ax1.legend(ncol=1, loc='lower left')
    ax1.set_ylabel(r'E$^2$dN/dE [ erg/cm²/s ]')
    ax1.set_title(f'{source} - SNR Time Intervals')
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    #ax1.set_ylim(2e-14, 1e-10)
    ax1.grid(True, which="both", linestyle="--", linewidth=0.5)

    # Bottom subplot: Spectrum - Time Intervals
    ax2 = fig.add_subplot(2, 1, 2)
    
    # Plot the "none" dataset (again in black)
    for dataset_label, (x, y, y_err, emin, emax) in dataset_none.items():
        x, y, y_err, emin, emax = np.array(x), np.array(y), np.array(y_err), np.array(emin), np.array(emax)
        e_lowers = x - emin
        e_uppers = emax - x
        #bin_size = emax - emin
        ax2.errorbar(x, y, xerr=[e_lowers, e_uppers], yerr=y_err,
                     fmt='s', capsize=5, color='black', label=f'{dataset_label}')
    if with_cat == True:     
        ax2.errorbar(eav0, f0, yerr=df0, xerr=de0, fmt='o', color="pink", label='Catalogue Spectrum')
    
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
    
    ax2.legend(ncol=1, loc='lower left')
    ax2.set_ylabel(r'E$^2$dN/dE [ erg/cm²/s ]')
    ax2.set_xlabel('Energy [ MeV ]')
    ax2.set_title(f'{source} - Linear Time Intervals')
    ax2.set_xscale('log')
    #ax2.set_ylim(2e-14, 1e-10)
    ax2.set_yscale('log')
    ax2.grid(True, which="both", linestyle="--", linewidth=0.5)

    # Adjust layout to prevent overlap
    fig.tight_layout()
    
    # Optionally save the figure as a PNG
    if source == "4FGL J0319.8+4130" and with_cat == False :
        plt.savefig(f"./fit_results/NGC1275_spectral_points.png", dpi=600)
    

    return fig

with PdfPages('./fit_results/NEW_spectral_points.pdf') as pdf:
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
                        
                        '''snr3 = bin_data_snr[bin_data_snr['loop_item'] == '3']
                        sorted_indices_snr3 = np.argsort(snr3['emin'])  # Get sorted indices
                        sorted_data_snr3 = snr3[sorted_indices_snr3]'''
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
                        
                                               
                        datasets = {f"No_Filtering": (sorted_data_none['geometric_mean'], sorted_data_none['flux_tot_value'], sorted_data_none['flux_tot_error'], sorted_data_none['emin'], sorted_data_none['emax'] )}
                        '''f"snr_3": (sorted_data_snr3['geometric_mean'], sorted_data_snr3['flux_tot_value'], sorted_data_snr3['flux_tot_error'], sorted_data_snr3['emin'], sorted_data_snr3['emax']),'''
                        datasets_snr = {f"snr_5": (sorted_data_snr5['geometric_mean'], sorted_data_snr5['flux_tot_value'], sorted_data_snr5['flux_tot_error'], sorted_data_snr5['emin'], sorted_data_snr5['emax']),
                                        f"snr_10": (sorted_data_snr10['geometric_mean'], sorted_data_snr10['flux_tot_value'], sorted_data_snr10['flux_tot_error'], sorted_data_snr10['emin'], sorted_data_snr10['emax'])}
                        datasets_lin = {f"week": (sorted_data_lin_week['geometric_mean'], sorted_data_lin_week['flux_tot_value'], sorted_data_lin_week['flux_tot_error'], sorted_data_lin_week['emin'], sorted_data_lin_week['emax']),
                                        f"month": (sorted_data_lin_month['geometric_mean'], sorted_data_lin_month['flux_tot_value'], sorted_data_lin_month['flux_tot_error'], sorted_data_lin_month['emin'], sorted_data_lin_month['emax'])}
                        #print(source_name)
                       
                        fig = simple_plot(datasets, datasets_snr, colors_snr, datasets_lin, colors_lin, source_name, with_cat=False)
                        #fig = simple_plot(datasets, None, None, None, None, source_name, with_cat=False)
                        pdf.savefig(fig)
                        plt.close(fig)

with PdfPages('./fit_results/NEW_spectral_points_wCat.pdf') as pdf:
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

                        '''snr3 = bin_data_snr[bin_data_snr['loop_item'] == '3']
                        sorted_indices_snr3 = np.argsort(snr3['emin'])  # Get sorted indices
                        sorted_data_snr3 = snr3[sorted_indices_snr3]'''
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
                        
                                                  
                        datasets = {f"No_Filtering": (sorted_data_none['geometric_mean'], sorted_data_none['flux_tot_value'], sorted_data_none['flux_tot_error'], sorted_data_none['emin'], sorted_data_none['emax'] )}
                        '''f"snr_3": (sorted_data_snr3['geometric_mean'], sorted_data_snr3['flux_tot_value'], sorted_data_snr3['flux_tot_error'], sorted_data_snr3['emin'], sorted_data_snr3['emax']),'''
                        datasets_snr = {f"snr_5": (sorted_data_snr5['geometric_mean'], sorted_data_snr5['flux_tot_value'], sorted_data_snr5['flux_tot_error'], sorted_data_snr5['emin'], sorted_data_snr5['emax']),
                                        f"snr_10": (sorted_data_snr10['geometric_mean'], sorted_data_snr10['flux_tot_value'], sorted_data_snr10['flux_tot_error'], sorted_data_snr10['emin'], sorted_data_snr10['emax'])}
                        datasets_lin = {f"week": (sorted_data_lin_week['geometric_mean'], sorted_data_lin_week['flux_tot_value'], sorted_data_lin_week['flux_tot_error'], sorted_data_lin_week['emin'], sorted_data_lin_week['emax']),
                                        f"month": (sorted_data_lin_month['geometric_mean'], sorted_data_lin_month['flux_tot_value'], sorted_data_lin_month['flux_tot_error'], sorted_data_lin_month['emin'], sorted_data_lin_month['emax'])}
                        #print(source_name)
                       
                        fig = simple_plot(datasets, datasets_snr, colors_snr, datasets_lin, colors_lin, source_name, with_cat=True)
                        pdf.savefig(fig)
                        plt.close(fig)

