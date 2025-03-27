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

# Flatten into list of dicts
rows = []
for row in focus:
    for res in row:
        fr = res["fit_result"]
        base, ax = fr["Base"], fr["Axion"]
        rows.append({
            "p0": res["p0"],
            "E_c": res["E_c"],
            "Base params": tuple(base["params"]),
            "Base χ²/dof": f"{base['chi2']:.3g}/{base['dof']}",
            "Axion params": tuple(ax["params"]),
            "Axion χ²/dof": f"{ax['chi2']:.3g}/{ax['dof']}",
            "Δχ²": fr["DeltaChi2"]
        })

# Build DataFrame
df = pd.DataFrame(rows)

# Sort by Δχ² ascending
df_sorted = df.sort_values(by="Δχ²")

# Select top 5 best (lowest Δχ²) and bottom 4 worst (highest Δχ²)
best = df_sorted.head(5)
worst = df_sorted.tail(5)

# Concatenate summary and reset index
df_summary = pd.concat([best, worst]).reset_index(drop=True)

# Format tuple columns
for col in ["Base params", "Axion params"]:
    df_summary[col] = df_summary[col].apply(lambda t: "(" + ", ".join(f"{x:.3g}" for x in t) + ")")

# Write to plain-text file
output_path = "fit_summary_focus_betap0_0.1.txt"
with open(output_path, "w") as fout:
    fout.write(df_summary.to_string(index=False, float_format=lambda x: f"{x:.3g}"))

print(f"Saved summary (5 best, 5 worst) to {output_path}")