import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from naima.models import EblAbsorptionModel
from scipy.optimize import curve_fit
import astropy.units as u
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from naima.models import EblAbsorptionModel
from scipy.optimize import curve_fit
import astropy.units as u
plt.rcParams.update({
    'font.family': 'serif',
    'mathtext.fontset': 'cm',
    'font.size': 20,
    'axes.titlesize': 20,
    'axes.labelsize': 20,
    'xtick.labelsize': 18,
    'ytick.labelsize': 18,
    'legend.fontsize': 20
})
def logpar_base(x, Norm, alpha_, beta_, z):
    E_b = 1000  # MeV
    ebl = EblAbsorptionModel(z).transmission(x * u.MeV)
    return Norm * (x/E_b)**(-(alpha_ + beta_*np.log(x/E_b))) * ebl

def plot_all_logpar(datasets, datasets_lin, dataset_snr, source,
                    png='logpar_all_filters.png'):
    """
    Single figure with two panels:
      Top:    fitted log-parabola curves for No filtering, Week, Month
      Bottom: fitted log-parabola curves for No filtering, SNR=3,5,10

    Legend shows each curve's photon index α.
    """
    # --- 1) load redshift ---
    with fits.open('table-4LAC-DR3-h.fits') as f:
        tbl = f[1].data
        z = tbl['Redshift'][tbl['Source_Name'] == source][0]

    # --- 2) curve_fit wrapper ---
    def model_lp(x, K, alpha, beta):
        return logpar_base(x, K, alpha, beta, z)

    # --- 3) prepare groups ---
    groups = {
        'top': [
            ('No filter',    datasets['No_Filtering']),
            ('Week',         datasets_lin['week']),
            ('Month',        datasets_lin['month']),
        ],
        'bot': [
            ('No filter',    datasets['No_Filtering']),
            ('SNR=3',        datasets_snr['snr_3']),
            ('SNR=5',        datasets_snr['snr_5']),
            ('SNR=10',       datasets_snr['snr_10']),
        ]
    }

    # --- 4) make figure ---
    fig, (ax_top, ax_bot) = plt.subplots(1,2, figsize=(20,8), sharey=True)
    for ax, key in [(ax_top, 'top'), (ax_bot, 'bot')]:
        for label, data in groups[key]:
            x, y, yerr, emin, emax = map(np.array, data)
            mask = (y!=0) & (np.abs(y)>=1e-13)
            xm, ym, em = x[mask], y[mask], yerr[mask]

            # fit
            p0 = [np.median(ym), 2.0, 0.1]
            popt, _ = curve_fit(model_lp, xm, ym, p0=p0, sigma=em)
            K_fit, alpha_fit, beta_fit = popt

            # compute curve
            xgrid = np.logspace(np.log10(xm.min()), np.log10(xm.max()), 300)
            ygrid = model_lp(xgrid, *popt)

            # plot
            ax.plot(xgrid, ygrid,
                    lw=2,
                    label=f"{label} (α={alpha_fit:.3f})")

        ax.set_yscale('log')
        ax.set_xscale('log')
        ax.set_xlabel('Energy [MeV]')
        ax.grid(True, which='both', ls='--')
        ax.legend(frameon=True, fontsize='small', loc = "lower left")

    ax_top.set_ylabel(r'E$^2$dN/dE [ erg/cm²/s ]')
    

    plt.tight_layout()
    fig.savefig(f'./fit_results/{png}', dpi=300)
    plt.close(fig)
    return fig

def print_fitted_params(datasets, datasets_lin, dataset_snr, source):
    # load redshift
    with fits.open('table-4LAC-DR3-h.fits') as f:
        tbl = f[1].data
        z = tbl['Redshift'][tbl['Source_Name'] == source][0]

    def model_lp(x, K, alpha, beta):
        return logpar_base(x, K, alpha, beta, z)

    labels = [
        ('No filtering', datasets['No_Filtering']),
        ('Week',          datasets_lin['week']),
        ('Month',         datasets_lin['month']),
        ('SNR=3',         dataset_snr['snr_3']),
        ('SNR=5',         dataset_snr['snr_5']),
        ('SNR=10',        dataset_snr['snr_10']),
    ]

    print("Fitted parameters (K, α, β):")
    for lab, data in labels:
        x, y, yerr, emin, emax = map(np.array, data)
        mask = (y != 0) & (np.abs(y) >= 1e-13)
        xm, ym, em = x[mask], y[mask], yerr[mask]

        # initial guesses: K ~ median(y), alpha ~ 2.0, beta ~ 0.1
        p0 = [np.median(ym), 2.0, 0.1]
        popt, pcov = curve_fit(model_lp, xm, ym, p0=p0, sigma=em)
        K_fit, alpha_fit, beta_fit = popt
        # optional: parameter uncertainties
        perr = np.sqrt(np.diag(pcov))
        K_err, alpha_err, beta_err = perr

        print(f"  {lab:12s}: K = {K_fit:.4e} ± {K_err:.4e}, "
              f"α = {alpha_fit:.4f} ± {alpha_err:.4f}, "
              f"β = {beta_fit:.4f} ± {beta_err:.4f}")

# ————— Prepare data points —————
source_name = "4FGL J0319.8+4130" #"4FGL J2321.9+2734"##"4FGL J0038.2-2459"#
cleaned = (source_name.replace(" ", "").replace(".", "dot")
                        .replace("+", "plus").replace("-", "minus").replace('"',''))
f_bin = fits.open(f'./fit_results/{cleaned}_fit_data_NONE.fits')
f_bin_snr = fits.open(f'./fit_results/{cleaned}_fit_data_SNR.fits')
f_bin_lin = fits.open(f'./fit_results/{cleaned}_fit_data_LIN.fits')

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

                    
datasets = {f"No_Filtering": (sorted_data_none['geometric_mean'], sorted_data_none['flux_tot_value'], sorted_data_none['flux_tot_error'], sorted_data_none['emin'], sorted_data_none['emax'] )}

datasets_snr = {f"snr_3": (sorted_data_snr3['geometric_mean'], sorted_data_snr3['flux_tot_value'], sorted_data_snr3['flux_tot_error'], sorted_data_snr3['emin'], sorted_data_snr3['emax']),
            f"snr_5": (sorted_data_snr5['geometric_mean'], sorted_data_snr5['flux_tot_value'], sorted_data_snr5['flux_tot_error'], sorted_data_snr5['emin'], sorted_data_snr5['emax']),
            f"snr_10": (sorted_data_snr10['geometric_mean'], sorted_data_snr10['flux_tot_value'], sorted_data_snr10['flux_tot_error'], sorted_data_snr10['emin'], sorted_data_snr10['emax'])}

datasets_lin = {f"week": (sorted_data_lin_week['geometric_mean'], sorted_data_lin_week['flux_tot_value'], sorted_data_lin_week['flux_tot_error'], sorted_data_lin_week['emin'], sorted_data_lin_week['emax']),
            f"month": (sorted_data_lin_month['geometric_mean'], sorted_data_lin_month['flux_tot_value'], sorted_data_lin_month['flux_tot_error'], sorted_data_lin_month['emin'], sorted_data_lin_month['emax'])}

fig = plot_all_logpar(
datasets,
datasets_lin,
datasets_snr,
source=source_name,
png='NGC1275_logpar_compare.png'
)
'''
print(source_name)
print_fitted_params(datasets, datasets_lin, datasets_snr, source=source_name)
'''