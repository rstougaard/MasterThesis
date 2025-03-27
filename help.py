from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import shlex
from iminuit.cost import LeastSquares
from iminuit import Minuit
from matplotlib.backends.backend_pdf import PdfPages
import pickle

import pandas as pd

# Load your results
with open("all_results_none_31_logpar_no_sys_error.pkl", "rb") as f:
    all_results_none = pickle.load(f)

focus = all_results_none["4FGL J0309.4-4000"]["No_Filtering"]

# Flatten into rows
rows = []
for row in focus:
    for res in row:
        fr = res["fit_result"]
        base, ax = fr["Base"], fr["Axion"]
        rows.append({
            "p0": res["p0"],
            "E_c": res["E_c"],
            "Base_params": tuple(base["params"]),
            "Base_chi2/dof": f"{base['chi2']:.2f}/{base['dof']}",
            "Axion_params": tuple(ax["params"]),
            "Axion_chi2/dof": f"{ax['chi2']:.2f}/{ax['dof']}",
            "Δχ²": fr["DeltaChi2"]
        })

df = pd.DataFrame(rows)

# Export to LaTeX
df.to_latex(
    "fit_summary_focus.tex",
    index=False,
    float_format="%.3g",
    caption="Fit parameters and χ² for 4FGL J0309.4-4000 (No Filtering)",
    label="tab:fit_summary"
)

print("LaTeX table written to fit_summary_focus.tex")