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

# Flatten into rows (same as before)
rows = []
for row in focus:
    for res in row:
        fr = res["fit_result"]
        base, ax = fr["Base"], fr["Axion"]
        rows.append({
            "p0": res["p0"],
            "E_c": res["E_c"],
            "Base params": tuple(base["params"]),
            "Base χ²/dof": f"{base['chi2']:.2f}/{base['dof']}",
            "Axion params": tuple(ax["params"]),
            "Axion χ²/dof": f"{ax['chi2']:.2f}/{ax['dof']}",
            "Δχ²": fr["DeltaChi2"]
        })

df = pd.DataFrame(rows)

# Write as plain‑text table
with open("fit_summary_focus.txt", "w") as fout:
    fout.write(df.to_string(index=False))

print("LaTeX table written to fit_summary_focus.txt")