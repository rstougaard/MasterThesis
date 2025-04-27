#!/usr/bin/env python3
import shlex
import numpy as np
import pandas as pd
from astropy.io import fits
import pickle
tex_path = "./tex_files/"
# ──────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────
SOURCE_LIST    = "sources_for_heatmaps.txt"
FERMI_CAT_FiTS = './test/gll_psc_v35.fit'

#choose mass and coupling
ma = 1e-9
ga = 2.2e-12
def maketable_best_fit_indv(AGN_list, ma, ga, OUTPUT_TEX= f"{tex_path}source_redshifts.tex"):
    #if no filter:
    #source | average significance | chi_base | chi_ALP | delta_chi |
    # 
    # Total 
    with open(AGN_list, 'r') as f:
        for line in f:
            parts = shlex.split(line.strip())
            if not parts:
                continue
            source = parts[0]
            f = fits.open(FERMI_CAT_FiTS)
            names4fgl = data['Source_Name']
            data = f[1].data
            data["Source_Name"]
            ok = np.where(names4fgl == source)

            # Extract flux data and uncertainties
            av_sig = data['Signif_Avg'][ok][0]

    #if any filter:
    #source | chi_base | chi_ALP | delta_chi |
    # 
    # Total 

    return

def _find_match_delta(results, target_m=None, target_g=None, target_d=-1.072):
    """
    Given `results` which may be:
      - a list of dicts,
      - a list of lists of dicts,
      - a dict whose values are lists of dicts,
    flatten it to a single list of dicts, then pick the entry
    closest to (target_m,target_g) if provided, otherwise exact/nearest on Δχ²=target_d.
    Returns that entry's DeltaChi2 or np.nan if none found.
    """
    # 1) flatten
    flat = []
    if isinstance(results, dict):
        iterable = results.values()
    else:
        iterable = results

    for item in iterable:
        if isinstance(item, dict):
            flat.append(item)
        elif isinstance(item, (list, tuple)):
            for sub in item:
                if isinstance(sub, dict):
                    flat.append(sub)
                # you can nest more levels here if needed
        # else: skip non‐dicts/strings

    if not flat:
        return np.nan

    # 2) pick by (m,g) if given
    if target_m is not None and target_g is not None:
        dist = lambda s: abs(s["m"] - target_m) + abs(s["g"] - target_g)
        best = min(flat, key=dist)
    else:
        # try exact Δχ² match
        for s in flat:
            if np.isclose(s["fit_result"]["DeltaChi2"], target_d, atol=1e-3):
                best = s
                break
        else:
            best = min(flat, key=lambda s: abs(s["fit_result"]["DeltaChi2"] - target_d))

    return best["fit_result"]["DeltaChi2"]


def maketable_best_fit_all_deltaChi(
    AGN_list,
    none_pickle,
    lin_pickle,
    fermi_fits,
    target_m=None,
    target_g=None,
    target_d=-1.072,
    output_tex="all_fits.tex"
):
    # --- load your two pickles ---
    with open(none_pickle, "rb") as f:
        all_none = pickle.load(f)
    with open(lin_pickle, "rb")  as f:
        all_lin  = pickle.load(f)

    # --- load Fermi catalog & make Source→Signif_Avg lookup ---
    with fits.open(fermi_fits) as hdul:
        data = hdul[1].data
        sig_lookup = dict(zip(data["Source_Name"], data["Signif_Avg"]))

    # --- build rows for DataFrame ---
    rows = []
    with open(AGN_list, "r") as f:
        for line in f:
            parts = shlex.split(line.strip())
            if not parts:
                continue
            src = parts[0]
            avg_sig = sig_lookup.get(src, np.nan)

            # get Δχ² for each filter
            d_none  = _find_match_delta(all_none .get(src, {}).get("No_Filtering", []),
                                        target_m, target_g, target_d)
            week    = _find_match_delta(all_lin  .get(src, {}).get("week", []),
                                        target_m, target_g, target_d)
            month   = _find_match_delta(all_lin  .get(src, {}).get("month", []),
                                        target_m, target_g, target_d)

            rows.append({
                "Source":           src,
                "Signif_Avg":       avg_sig,
                "Δχ² (no filter)":  d_none,
                "Δχ² (week)":       week,
                "Δχ² (month)":      month
            })

    df = pd.DataFrame(rows)

    # --- add Total row summing numeric columns ---
    total = {
        "Source": "Total",
        **{col: df[col].sum() for col in df.columns if col != "Source"}
    }
    df = df.append(total, ignore_index=True)

    # --- emit LaTeX longtable ---
    with open(output_tex, "w") as out:
        out.write(r"""\begin{longtable}{l r r r r}
\caption{Best‐fit $\Delta\chi^2$ for each filter\label{tab:best_fit_deltaChi}}\\
\toprule
Source & Signif.\ Avg.\ & \multicolumn{3}{c}{$\Delta\chi^2$} \\
\cmidrule(lr){3-5}
       &                & No filter & Week & Month \\
\midrule
\endfirsthead

\multicolumn{5}{c}% 
{{\tablename\ \thetable{} -- continued}} \\
\toprule
Source & Signif.\ Avg.\ & \multicolumn{3}{c}{$\Delta\chi^2$} \\
\cmidrule(lr){3-5}
       &                & No filter & Week & Month \\
\midrule
\endhead

\midrule \multicolumn{5}{r}{{Continued on next page}} \\
\endfoot

\bottomrule
\endlastfoot
""")
        out.write(
            df.to_latex(
                index=False,
                float_format="%.2f",
                column_format="lrrrr",
                header=False
            )
        )
        out.write("\n\\end{longtable}\n")

    print(f"Wrote LaTeX table to {output_tex}")
    return df

# Example usage:
if __name__ == "__main__":
    df = maketable_best_fit_all_deltaChi(
        AGN_list=SOURCE_LIST,
        none_pickle="none_new0_sys_error.pkl",
        lin_pickle="lin_new0_sys_error.pkl",
        fermi_fits=FERMI_CAT_FiTS,
        target_m=ma,      # your desired m
        target_g=ga,      # your desired g
        # if you omit target_m/target_g it will use target_d below:
        # target_d=-1.072,
        output_tex="all_fits.tex"
    )
    print(df)

