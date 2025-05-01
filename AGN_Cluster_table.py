#!/usr/bin/env python3
import shlex
import numpy as np
import pandas as pd
from astropy.io import fits
import os
import pickle
tex_path = "./tex_files/"
# ──────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────
SOURCE_LIST    = "sources_for_heatmaps.txt"
FERMI_CAT_FiTS = './test/gll_psc_v35.fit'

#choose mass and coupling
ma = 0.9e-9
ga =  1.1e-12 #2.2
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
    snr_pickle,
    fermi_fits,
    target_m=None,
    target_g=None,
    target_d=-1.072,
    output_tex="all_fits.tex"
):
    import shlex
    import numpy as np
    import pandas as pd
    import pickle
    from astropy.io import fits

    # --- load your three pickles ---
    with open(none_pickle, "rb") as f:
        all_none = pickle.load(f)
    with open(lin_pickle, "rb")  as f:
        all_lin  = pickle.load(f)
    with open(snr_pickle, "rb")  as f:
        all_snr  = pickle.load(f)

    # --- load Fermi catalog & make Source→Signif_Avg lookup ---
    with fits.open(fermi_fits) as hdul:
        data = hdul[1].data
        sig_lookup = dict(zip(data["Source_Name"], data["Signif_Avg"]))

    # --- define which AGNs to blank per column ---
    flags = {
        "Δχ² (no filter)": ["4FGL J0317.8-4414"],   # fill your list
        "Δχ² (week)":       ["4FGL J0317.8-4414"],
        "Δχ² (month)":      ["4FGL J0132.7-0804","4FGL J0317.8-4414", "4FGL J1242.9+7315"],                           # empty = nobody flagged
        "Δχ² (snr3)":       ["4FGL J0317.8-4414", "4FGL J1516.8+2918"],
        "Δχ² (snr5)":       ["4FGL J0132.7-0804", "4FGL J0317.8-4414", "4FGL J0912.5+1556", "4FGL J1516.8+2918"],
        "Δχ² (snr10)":      ["4FGL J0132.7-0804", "4FGL J0317.8-4414","4FGL J1213.0+5129"]
    }
    
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
            d_none  = _find_match_delta(all_none.get(src, {}).get("No_Filtering", []),
                                        target_m, target_g, target_d)
            if src in flags["Δχ² (no filter)"]:
                d_none = np.nan

            week    = _find_match_delta(all_lin.get(src, {}).get("week", []),
                                        target_m, target_g, target_d)
            if src in flags["Δχ² (week)"]:
                week = np.nan

            month   = _find_match_delta(all_lin.get(src, {}).get("month", []),
                                        target_m, target_g, target_d)
            if src in flags["Δχ² (month)"]:
                month = np.nan

            snr3    = _find_match_delta(all_snr.get(src, {}).get("snr_3", []),
                                        target_m, target_g, target_d)
            if src in flags["Δχ² (snr3)"]:
                snr3 = np.nan

            snr5    = _find_match_delta(all_snr.get(src, {}).get("snr_5", []),
                                        target_m, target_g, target_d)
            if src in flags["Δχ² (snr5)"]:
                snr5 = np.nan

            snr10   = _find_match_delta(all_snr.get(src, {}).get("snr_10", []),
                                        target_m, target_g, target_d)
            if src in flags["Δχ² (snr10)"]:
                snr10 = np.nan

            rows.append({
                "Source":           src,
                "Signif_Avg":       avg_sig,
                "Δχ² (no filter)":  d_none,
                "Δχ² (week)":       week,
                "Δχ² (month)":      month,
                "Δχ² (snr3)":       snr3,
                "Δχ² (snr5)":       snr5,
                "Δχ² (snr10)":      snr10
            })

    df = pd.DataFrame(rows)

    # --- add Total row summing numeric columns ---
    total = {
        "Source": "Total",
        **{col: df[col].sum() for col in df.columns if col != "Source"}
    }
    df = pd.concat([df, pd.DataFrame([total])], ignore_index=True)

    # --- emit LaTeX longtable (NaNs will be rendered as blanks) ---
    with open(output_tex, "w") as out:
        out.write(r"""\begin{longtable}{l r r r r r r r}
\caption{Best‐fit $\Delta\chi^2$ for each filter\label{tab:best_fit_deltaChi}}\\
\toprule
Source & Signif.\ Avg.\ & \multicolumn{6}{c}{$\Delta\chi^2$} \\
\cmidrule(lr){4-8}
       &                & No filter & Week & Month & SNR 3 & SNR 5 & SNR 10\\
\midrule
\endfirsthead

\multicolumn{8}{c}%
{{\tablename\ \thetable{} -- continued}} \\
\toprule
Source & Signif.\ Avg.\ & \multicolumn{6}{c}{$\Delta\chi^2$} \\
\cmidrule(lr){4-8}
       &                & No filter & Week & Month & SNR 3 & SNR 5 & SNR 10\\
\midrule
\endhead

\midrule \multicolumn{8}{r}{{Continued on next page}} \\
\endfoot

\bottomrule
\endlastfoot
""")
        out.write(
            df.to_latex(
                index=False,
                float_format="%.2f",
                na_rep="",
                column_format="lrrrrrrr",
                header=False
            )
        )
        out.write("\n\\end{longtable}\n")

    print(f"Wrote LaTeX table to {output_tex}")
    return df

def _extract_chi(results, target_m=None, target_g=None, target_d=-1.072):
    """
    Flatten results and pick the best match, returning:
      (chi_base, dof_base, chi_alp, dof_alp, delta_chi)
    """
    # flatten
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

    if not flat:
        return (np.nan, np.nan, np.nan, np.nan, np.nan)

    # pick best entry
    if target_m is not None and target_g is not None:
        dist = lambda s: abs(s.get("m", np.nan) - target_m) + abs(s.get("g", np.nan) - target_g)
        best = min(flat, key=dist)
    else:
        # match Δχ²
        fits = [s.get("fit_result", {}) for s in flat]
        # exact
        for s in flat:
            if np.isclose(s["fit_result"]["DeltaChi2"], target_d, atol=1e-3):
                best = s
                break
        else:
            best = min(flat, key=lambda s: abs(s["fit_result"]["DeltaChi2"] - target_d))

    base = best["fit_result"]["Base"]
    alp  = best["fit_result"]["Axion"]
    chi_base = base.get("chi2", np.nan)
    dof_base = base.get("dof", np.nan)
    chi_alp  = alp.get("chi2", np.nan)
    dof_alp  = alp.get("dof", np.nan)
    delta_chi = best["fit_result"]["DeltaChi2"]
    return (chi_base, dof_base, chi_alp, dof_alp, delta_chi)

def maketable_best_fit_all_Chi(
    AGN_list,
    none_pickle,
    lin_pickle,
    snr_pickle,
    fermi_fits,
    target_m=None,
    target_g=None,
    target_d=-1.072,
    output_prefix="all_fits"
):
    """
    Generate separate LaTeX tables for each filter:
      No_Filtering, week, month, snr_3, snr_5, snr_10
    Each table has columns: Source, chi_base (dof), chi_alp (dof), DeltaChi2, with Total row.
    """
    # load pickles
    with open(none_pickle, "rb") as f:
        all_none = pickle.load(f)
    with open(lin_pickle, "rb")  as f:
        all_lin  = pickle.load(f)
    with open(snr_pickle, "rb")  as f:
        all_snr  = pickle.load(f)

    # load fermi catalog
    with fits.open(fermi_fits) as hdul:
        data = hdul[1].data
        sig_lookup = dict(zip(data["Source_Name"], data["Signif_Avg"]))

    # define filters and flags to blank
    filters = [
        ("No_Filtering", all_none),
        ("week",       all_lin),
        ("month",      all_lin),
        ("snr_3",      all_snr),
        ("snr_5",      all_snr),
        ("snr_10",     all_snr)
    ]
    flags = {
        "No_Filtering": ["4FGL J0317.8-4414"],
        "week":         ["4FGL J0317.8-4414"],
        "month":        ["4FGL J0132.7-0804","4FGL J0317.8-4414","4FGL J1242.9+7315"],
        "snr_3":        ["4FGL J0317.8-4414", "4FGL J1516.8+2918"],
        "snr_5":        ["4FGL J0132.7-0804","4FGL J0317.8-4414","4FGL J0912.5+1556","4FGL J1516.8+2918"],
        "snr_10":       ["4FGL J0132.7-0804","4FGL J0317.8-4414","4FGL J1213.0+5129"]
    }

    # read AGN list
    sources = []
    with open(AGN_list, "r") as f:
        for line in f:
            parts = shlex.split(line.strip())
            if parts:
                sources.append(parts[0])

    # for each filter, build DataFrame
    for label, data_dict in filters:
        rows = []
        for src in sources:
            if src in flags.get(label, []):
                chi_base, dof_base, chi_alp, dof_alp, delta = (np.nan,)*5
            else:
                results = data_dict.get(src, {}).get(label, [])
                chi_base, dof_base, chi_alp, dof_alp, delta = _extract_chi(
                    results, target_m, target_g, target_d)
            rows.append({
                "Source": src,
                "chi_base": chi_base,
                "dof_base":  dof_base,
                "chi_alp":  chi_alp,
                "dof_alp":  dof_alp,
                "DeltaChi2": delta
            })
        df = pd.DataFrame(rows)
        # total row
        total = {"Source": "Total"}
        for col in ["chi_base","dof_base","chi_alp","dof_alp","DeltaChi2"]:
            total[col] = df[col].sum(skipna=True)
        df = pd.concat([df, pd.DataFrame([total])], ignore_index=True)

        # write LaTeX
        out_tex = f"{output_prefix}_{label}.tex"
        with open(out_tex, "w") as out:
            out.write(r"""\begin{longtable}{l r r r r r}
\caption{Best‐fit $\Delta\chi^2$ for filter: %s\label{tab:%s}}\\
\toprule
Source & Base $\chi^2$ (dof) & Axion $\chi^2$ (dof) & $\Delta\chi^2$ \\
\midrule
\endfirsthead

\multicolumn{6}{c}%%
{{\tablename\ \thetable{} -- continued}} \\
\toprule
Source & Base $\chi^2$ (dof) & Axion $\chi^2$ (dof) & $\Delta\chi^2$ \\
\midrule
\endhead
\midrule \multicolumn{6}{r}{{Continued on next page}} \\
\endfoot

\bottomrule
\endlastfoot
""" % (label, label))
            out.write(
                df.to_latex(
                    index=False,
                    columns=["Source","chi_base","dof_base","chi_alp","dof_alp","DeltaChi2"],
                    float_format="%.2f",
                    na_rep="",
                    column_format="lrrrrr"
                )
            )
            out.write("\n\\end{longtable}\n")
        print(f"Wrote table for {label} to {out_tex}")
    return True

def maketable_TS_total(
    agn_list_file,
    fits_dir,
    output_prefix="all_sources"
):
    """
    For each source in agn_list_file, read its NONE, LIN (week/month),
    and SNR (3/5/10) fit‐data FITS files, compute the total significance
    per source (sqrt of sum of TS over 7 bins), and write out LaTeX tables:
      - one for NONE (no filter) with columns Source, TotSignif
      - one for each filter with columns Source, TotSignif, ΔSignif vs NONE
    """
    import shlex
    import numpy as np
    import pandas as pd
    from astropy.io import fits
    import os

    # helper: read exactly 7 TS values from a FITS table
    def read_TS(fpath, loop_filter=None):
        with fits.open(fpath) as hdul:
            data = hdul[1].data
            if loop_filter is not None:
                data = data[data["loop_item"] == loop_filter]
            # sort by emin
            order = np.argsort(data["emin"])
            ts_vals = data["TS"][order]
            # ensure exactly 7 bins
            if len(ts_vals) >= 7:
                ts_vals = ts_vals[:7]
            else:
                ts_vals = np.pad(ts_vals, (0, 7 - len(ts_vals)), constant_values=np.nan)
        return ts_vals

    # read source list
    sources = []
    with open(agn_list_file, "r") as f:
        for line in f:
            parts = shlex.split(line.strip())
            if parts:
                sources.append(parts[0])

    # build a DataFrame for NONE
    rows_none = []
    for src in sources:
        clean = (src.replace(" ", "")
                    .replace(".", "dot")
                    .replace("+", "plus")
                    .replace("-", "minus")
                    .replace('"', ""))
        f_none = os.path.join(fits_dir, f"{clean}_fit_data_NONE.fits")
        ts = read_TS(f_none, loop_filter=None)
        tot_signif = np.sqrt(np.nansum(ts))
        rows_none.append({"Source": src, "Tot_Signif": tot_signif})
    df_none = pd.DataFrame(rows_none)

    # write LaTeX for NONE
    with open(f"{output_prefix}_NONE_TS_total.tex", "w") as out:
        out.write(r"""\begin{longtable}{l r}
\caption{Total significance per source (No filter)\label{tab:TS_NONE}}\\
\toprule
Source & TotSignif \\
\midrule
\endfirsthead

\multicolumn{2}{c}{{\tablename\ \thetable{} -- continued}} \\
\toprule
Source & TotSignif \\
\midrule
\endhead

\midrule \multicolumn{2}{r}{{Continued on next page}} \\
\endfoot

\bottomrule
\endlastfoot
""")
        out.write(
            df_none.to_latex(
                index=False,
                columns=["Source", "Tot_Signif"],
                float_format="%.2f",
                na_rep="",
                header=False,
                column_format="l r"
            )
        )
        out.write("\n\\end{longtable}\n")

    # define filters and their loop_item labels
    filters = {
        "LIN_week":   ("week",   "_LIN"),
        "LIN_month":  ("month",  "_LIN"),
        "SNR_3":      ("3",      "_SNR"),
        "SNR_5":      ("5",      "_SNR"),
        "SNR_10":     ("10",     "_SNR")
    }

    # process each filter
    for label, (loop_item, suffix) in filters.items():
        rows = []
        for src, row_none in zip(sources, rows_none):
            clean = (src.replace(" ", "")
                        .replace(".", "dot")
                        .replace("+", "plus")
                        .replace("-", "minus")
                        .replace('"', ""))
            fpath = os.path.join(fits_dir, f"{clean}_fit_data{suffix}.fits")
            ts = read_TS(fpath, loop_filter=loop_item)
            tot_signif = np.sqrt(np.nansum(ts))
            delta = tot_signif - row_none["Tot_Signif"]
            rows.append({
                "Source": src,
                "Tot_Signif": tot_signif,
                "Delta_Signif": delta
            })
        df = pd.DataFrame(rows)

        # write LaTeX for this filter
        out_tex = f"{output_prefix}_{label}_TS_total.tex"
        with open(out_tex, "w") as out:
            out.write(r"""\begin{longtable}{l r r}
\caption{Total significance per source (%s) and difference from No filter\label{tab:TS_%s}}\\
\toprule
Source & TotSignif & $\Delta$Signif \\
\midrule
\endfirsthead

\multicolumn{3}{c}{{\tablename\ \thetable{} -- continued}} \\
\toprule
Source & TotSignif & $\Delta$Signif \\
\midrule
\endhead

\midrule \multicolumn{3}{r}{{Continued on next page}} \\
\endfoot

\bottomrule
\endlastfoot
""" % (label, label))
            out.write(
                df.to_latex(
                    index=False,
                    columns=["Source", "Tot_Signif", "Delta_Signif"],
                    float_format="%.2f",
                    na_rep="",
                    header=False,
                    column_format="l r r"
                )
            )
            out.write("\n\\end{longtable}\n")

    return True

def maketable_TS_comparison(
    agn_list_file,
    fits_dir,
    output_tex="TS_comparison.tex"
):
    """
    Builds one LaTeX longtable with columns:
      Source | TotSignif_NONE | ΔWeek | ΔMonth | ΔSNR3 | ΔSNR5 | ΔSNR10
    """
    import shlex, os
    import numpy as np
    import pandas as pd
    from astropy.io import fits

    # helper: read exactly 7 TS values from a FITS file, optionally filtering by loop_item
    def read_TS(fpath, loop_filter=None):
        with fits.open(fpath) as hdul:
            data = hdul[1].data
        if loop_filter is not None:
            data = data[data["loop_item"] == loop_filter]
        order = np.argsort(data["emin"])
        ts_vals = data["TS"][order]
        # pad or truncate to exactly 7 bins
        if len(ts_vals) >= 7:
            return ts_vals[:7]
        return np.pad(ts_vals, (0, 7 - len(ts_vals)), constant_values=np.nan)

    # read our source list
    sources = []
    with open(agn_list_file, "r") as f:
        for line in f:
            parts = shlex.split(line.strip())
            if parts:
                sources.append(parts[0])

    # define our filters: name → (loop_item, filename‐suffix)
    filters = {
        "ΔWeek":  ("week",  "_LIN"),
        "ΔMonth": ("month", "_LIN"),
        "ΔSNR3":  ("3",     "_SNR"),
        "ΔSNR5":  ("5",     "_SNR"),
        "ΔSNR10": ("10",    "_SNR"),
    }

    # build a row per source
    rows = []
    for src in sources:
        # sanitize to match your file naming convention
        clean = (src.replace(" ", "")
                   .replace(".", "dot")
                   .replace("+", "plus")
                   .replace("-", "minus")
                   .replace('"', ""))

        # 1) compute no‐filter total significance
        f_none  = os.path.join(fits_dir, f"{clean}_fit_data_NONE.fits")
        ts_none = read_TS(f_none, loop_filter=None)
        tot_none = np.sqrt(np.nansum(ts_none))

        row = {
            "Source":             src,
            "TotSignif_NONE":     tot_none
        }

        # 2) for each filter, compute Δ = TotSignif_filter − TotSignif_NONE
        for col, (loop_item, suffix) in filters.items():
            f_filt   = os.path.join(fits_dir, f"{clean}_fit_data{suffix}.fits")
            ts_filt  = read_TS(f_filt, loop_filter=loop_item)
            tot_filt = np.sqrt(np.nansum(ts_filt))
            row[col] = tot_filt - tot_none

        rows.append(row)

    df = pd.DataFrame(rows)

    # prepare LaTeX longtable
    # 1 'l' + 6 'r' columns
    col_fmt = "l " + "r " * (1 + len(filters))
    headers = ["Source", "TotSignif\\_NONE"] + list(filters.keys())

    with open(output_tex, "w") as out:
        out.write(r"""\begin{longtable}{%s}
\caption{Total significance (no‐filter) and change under each filter\label{tab:TS_comparison}}\\
\toprule
%s \\
\midrule
\endfirsthead

\multicolumn{%d}{c}{{\tablename\ \thetable{} -- continued}} \\
\toprule
%s \\
\midrule
\endhead

\midrule \multicolumn{%d}{r}{{Continued on next page}} \\
\endfoot

\bottomrule
\endlastfoot
""" % (
            col_fmt.strip(),
            " & ".join(headers),
            len(headers),
            " & ".join(headers),
            len(headers),
        ))
        out.write(
            df.to_latex(
                index=False,
                columns=headers,
                float_format="%.2f",
                na_rep="",
                header=False,
                column_format=col_fmt.strip()
            )
        )
        out.write("\n\\end{longtable}\n")

    return df

def extract_TS_per_bin(
    agn_list_file: str,
    fits_dir: str,
    loop_filter: str = None,
    suffix: str = "_NONE"
) -> pd.DataFrame:
    """
    Reads each source in `agn_list_file`, opens the corresponding
    FITS file in `fits_dir` named
      <cleaned_source>_fit_data<suffix>.fits
    optionally filters by `loop_item == loop_filter`, sorts by `emin`,
    and returns a DataFrame with columns:
      Source, Bin1, Bin2, ..., Bin7, TotSignif
    where TotSignif = sqrt(sum_i TS_i).
    TS values are padded with NaN if fewer than 7 bins.
    """
    def clean_name(src: str) -> str:
        return (src.replace(" ", "")
                   .replace(".", "dot")
                   .replace("+", "plus")
                   .replace("-", "minus")
                   .replace('"', ""))

    def read_TS(fpath: str) -> np.ndarray:
        with fits.open(fpath) as hdul:
            data = hdul[1].data
        if loop_filter is not None:
            data = data[data["loop_item"] == loop_filter]
        order = np.argsort(data["emin"])
        ts_vals = data["TS"][order]
        if len(ts_vals) >= 7:
            return ts_vals[:7]
        return np.pad(ts_vals, (0, 7 - len(ts_vals)), constant_values=np.nan)

    # read source list
    sources = []
    with open(agn_list_file, "r") as f:
        for line in f:
            parts = shlex.split(line.strip())
            if parts:
                sources.append(parts[0])

    # build DataFrame rows
    rows = []
    for src in sources:
        cname = clean_name(src)
        fpath = os.path.join(fits_dir, f"{cname}_fit_data{suffix}.fits")
        ts = read_TS(fpath)
        row = {"Source": src}
        for i, val in enumerate(ts, start=1):
            row[f"Bin{i}"] = val
        # compute total significance = sqrt(sum TS_i)
        total_TS = np.nansum(ts)
        row["TotSignif"] = np.sqrt(total_TS) if total_TS >= 0 else np.nan
        rows.append(row)

    cols = ["Source"] + [f"Bin{i}" for i in range(1, 8)] + ["TotSignif"]
    df = pd.DataFrame(rows, columns=cols)
    return df

def make_TS_per_bin_tables(
    agn_list_file: str,
    fits_dir: str,
    output_dir: str = "."
):
    # mapping: (table_name, loop_filter, suffix)
    configs = [
        ("NONE",      None,     "_NONE"),
        ("LIN_week",  "week",   "_LIN"),
        ("LIN_month", "month",  "_LIN"),
        ("SNR_3",     "3",      "_SNR"),
        ("SNR_5",     "5",      "_SNR"),
        ("SNR_10",    "10",     "_SNR"),
    ]

    for name, loop_item, suffix in configs:
        # 1) extract
        df = extract_TS_per_bin(
            agn_list_file,
            fits_dir,
            loop_filter=loop_item,
            suffix=suffix
        )

        # 2) write LaTeX
        cols = ["Source"] + [f"Bin{i}" for i in range(1,8)] + ["TotSignif"]
        col_fmt = "l " + " ".join("r" for _ in cols[1:])
        header = " & ".join(cols).replace("_", "\\_") + r" \\"

        tex_path = f"TS_per_bin_{name}.tex"
        with open(tex_path, "w") as out:
            out.write(r"""\begin{longtable}{%s}
\caption{TS per bin and total significance for %s\label{tab:TS_%s}}\\
\toprule
%s
\midrule
\endfirsthead

\multicolumn{%d}{c}{{\tablename\ \thetable{} -- continued}} \\
\toprule
%s
\midrule
\endhead

\midrule \multicolumn{%d}{r}{{Continued on next page}} \\
\endfoot

\bottomrule
\endlastfoot
""" % (
                col_fmt,
                name.replace("_", " "),
                name,
                header,
                len(cols),
                header,
                len(cols),
            ))
            out.write(
                df.to_latex(
                    index=False,
                    columns=cols,
                    float_format="%.2f",
                    na_rep="",
                    header=False,
                    column_format=col_fmt
                )
            )
            out.write("\n\\end{longtable}\n")

        print(f"Wrote {tex_path}")
def maketable_TS_total_comparison(
    agn_list_file: str,
    fits_dir: str,
    output_tex: str = "TS_total_comparison.tex"
) -> pd.DataFrame:
    """
    For each source in `agn_list_file`, reads the NONE, LIN (week/month),
    and SNR (3/5/10) FITS under `fits_dir`, computes the total significance
    TotSignif = sqrt(sum_i TS_i) for NONE, and then ΔTotSignif for each filter.
    Emits a single LaTeX longtable with columns:
      Source | TotSignif_NONE | ΔWeek | ΔMonth | ΔSNR3 | ΔSNR5 | ΔSNR10
    Returns the DataFrame.
    """
    def clean_name(src: str) -> str:
        return (src.replace(" ", "")
                   .replace(".", "dot")
                   .replace("+", "plus")
                   .replace("-", "minus")
                   .replace('"', ""))

    def read_total_signif(clean: str, loop_filter: str = None, suffix: str = "_NONE") -> float:
        fpath = os.path.join(fits_dir, f"{clean}_fit_data{suffix}.fits")
        with fits.open(fpath) as hdul:
            data = hdul[1].data
        if loop_filter is not None:
            data = data[data["loop_item"] == loop_filter]
        order = np.argsort(data["emin"])
        ts_vals = data["TS"][order]
        ts7 = ts_vals[:7] if len(ts_vals) >= 7 else np.pad(ts_vals, (0,7-len(ts_vals)), constant_values=0.0)
        return np.sqrt(np.nansum(ts7))

    # read source list
    sources = []
    with open(agn_list_file, "r") as f:
        for line in f:
            parts = shlex.split(line.strip())
            if parts:
                sources.append(parts[0])

    # define datasets
    datasets = {
        "NONE":    (None,   "_NONE"),
        "ΔWeek":   ("week", "_LIN"),
        "ΔMonth":  ("month","_LIN"),
        "ΔSNR3":   ("3",    "_SNR"),
        "ΔSNR5":   ("5",    "_SNR"),
        "ΔSNR10":  ("10",   "_SNR"),
    }

    # build rows
    rows = []
    for src in sources:
        clean = clean_name(src)
        tot_none = read_total_signif(clean, *datasets["NONE"])
        row = {"Source": src, "TotSignif_NONE": tot_none}
        for label, (loop, suff) in datasets.items():
            if label == "NONE":
                continue
            tot_filt = read_total_signif(clean, loop, suff)
            row[label] = tot_filt - tot_none
        rows.append(row)
        print(row)

    df = pd.DataFrame(rows, columns=[
        "Source", "TotSignif_NONE",
        "ΔWeek", "ΔMonth", "ΔSNR3", "ΔSNR5", "ΔSNR10"
    ])

    # write LaTeX
    col_fmt = "l r " + " ".join("r" for _ in range(len(datasets)-1))
    headers = ["Source", "TotSignif\\_NONE", r"$\Delta$Week",
               r"$\Delta$Month", r"$\Delta$SNR3", r"$\Delta$SNR5", r"$\Delta$SNR10"]
    ncols = len(headers)                   # this is 7
    header_row = " & ".join(headers) + r" \\"

    with open(output_tex, "w") as out:
        out.write(f"""\\begin{{longtable}}{{{col_fmt}}}
\\caption{{Total significance (no‐filter) and change under each filter\\label{{tab:TS_total_comparison}}}}\\\\
\\toprule
{header_row}
\\midrule
\\endfirsthead

\\multicolumn{{{ncols}}}{{c}}{{{{\\tablename\\ \\thetable{{}} -- continued}}}} \\\\
\\toprule
{header_row}
\\midrule
\\endhead

\\midrule \\multicolumn{{{ncols}}}{{r}}{{Continued on next page}} \\\\
\\endfoot

\\bottomrule
\\endlastfoot
""")
        out.write(
            df.to_latex(
                index=False,
                columns=headers,
                float_format="%.2f",
                na_rep="",
                header=False,
                column_format=col_fmt
            )
        )
        out.write("\n\\end{longtable}\n")

    return df

def maketable_sum_signif_per_bin(
    agn_list_file: str,
    fits_dir: str,
    output_tex: str = "sum_signif_per_bin.tex"
) -> pd.DataFrame:
    """
    For each dataset suffix (NONE, LIN_week, LIN_month, SNR_3, SNR_5, SNR_10),
    reads the TS‐per‐bin for all sources, computes per‐bin significance = sqrt(TS),
    sums that significance across sources, and writes a LaTeX table whose rows
    are the suffixes and whose columns are Bin1…Bin7 summed significance.
    Returns the summary DataFrame.
    """
    # helper to clean source names
    def clean_name(src: str) -> str:
        return (src.replace(" ", "")
                   .replace(".", "dot")
                   .replace("+", "plus")
                   .replace("-", "minus")
                   .replace('"', ""))

    # helper to read exactly 7 TS values
    def read_TS(fpath: str, loop_filter: str = None) -> np.ndarray:
        with fits.open(fpath) as hdul:
            data = hdul[1].data
        if loop_filter is not None:
            data = data[data["loop_item"] == loop_filter]
        order = np.argsort(data["emin"])
        ts_vals = data["TS"][order]
        if len(ts_vals) >= 7:
            return ts_vals[:7]
        else:
            return np.pad(ts_vals, (0,7-len(ts_vals)), constant_values=np.nan)

    # read source list
    sources = []
    with open(agn_list_file, "r") as f:
        for line in f:
            parts = shlex.split(line.strip())
            if parts:
                sources.append(parts[0])

    # define suffixes and filters
    configs = [
        ("NONE",      None,   "_NONE"),
        ("LIN_week",  "week", "_LIN"),
        ("LIN_month", "month","_LIN"),
        ("SNR_3",     "3",    "_SNR"),
        ("SNR_5",     "5",    "_SNR"),
        ("SNR_10",    "10",   "_SNR"),
    ]

    # build summary rows
    summary = []
    for name, loop_item, suffix in configs:
        # collect TS for all sources, shape (n_sources, 7)
        all_ts = []
        for src in sources:
            cn = clean_name(src)
            fpath = os.path.join(fits_dir, f"{cn}_fit_data{suffix}.fits")
            ts7 = read_TS(fpath, loop_filter=loop_item)
            all_ts.append(ts7)
        all_ts = np.vstack(all_ts)  # shape (n_sources,7)

        # compute significance per bin and sum across sources
        # significance = sqrt(TS), but ignore nan or negative TS
        sig = np.sqrt(np.clip(all_ts, 0, None))
        sum_sig = np.nansum(sig, axis=0)  # length 7

        # build row
        row = {"Dataset": name}
        for i, v in enumerate(sum_sig, start=1):
            row[f"Bin{i}"] = v
        summary.append(row)

    df_sum = pd.DataFrame(summary, columns=["Dataset"] + [f"Bin{i}" for i in range(1,8)])

    # write LaTeX
    cols = ["Dataset"] + [f"Bin{i}" for i in range(1,8)]
    col_fmt = "l " + " ".join("r" for _ in cols[1:])
    header = " & ".join(cols) + r" \\"

    ncols = len(cols)
    with open(output_tex, "w") as out:
        out.write(f"""\\begin{{longtable}}{{{col_fmt}}}
\\caption{{Summed significance per bin across sources\\label{{tab:sum_signif_per_bin}}}}\\\\
\\toprule
{header}
\\midrule
\\endfirsthead

\\multicolumn{{{ncols}}}{{c}}{{{{\\tablename\\ \\thetable{{}} -- continued}}}} \\\\
\\toprule
{header}
\\midrule
\\endhead

\\midrule \\multicolumn{{{ncols}}}{{r}}{{Continued on next page}} \\\\
\\endfoot

\\bottomrule
\\endlastfoot
""")
        out.write(
            df_sum.to_latex(
                index=False,
                columns=cols,
                float_format="%.2f",
                na_rep="",
                header=False,
                column_format=col_fmt
            )
        )
        out.write("\n\\end{longtable}\n")

    return df_sum

def maketable_sum_signif_per_bin_with_deltas(
    agn_list_file: str,
    fits_dir: str,
    output_tex: str = "sum_signif_per_bin_with_deltas.tex"
) -> pd.DataFrame:
    """
    Computes per-bin summed significance across sources for each dataset suffix,
    then adds delta‐rows giving (suffix – NONE) per bin. Emits a LaTeX longtable
    with rows: NONE, LIN_week, LIN_month, SNR_3, SNR_5, SNR_10,
    followed by ΔLIN_week, ΔLIN_month, ΔSNR_3, ΔSNR_5, ΔSNR_10.
    Columns are Dataset, Bin1…Bin7.
    """
    
    # helper to clean names and read 7 TS bins
    def clean_name(src):
        return (src.replace(" ", "")
                   .replace(".", "dot")
                   .replace("+", "plus")
                   .replace("-", "minus")
                   .replace('"', ""))
    def read_TS(fpath, loop_filter=None):
        with fits.open(fpath) as hdul:
            data = hdul[1].data
        if loop_filter is not None:
            data = data[data["loop_item"] == loop_filter]
        order = np.argsort(data["emin"])
        ts = data["TS"][order]
        if len(ts) >= 7:
            return ts[:7]
        return np.pad(ts, (0,7-len(ts)), constant_values=np.nan)

    # read sources
    sources = []
    with open(agn_list_file) as f:
        for line in f:
            parts = shlex.split(line)
            if parts:
                sources.append(parts[0])

    # define configs
    configs = [
        ("NONE",      None,   "_NONE"),
        ("LIN_week",  "week", "_LIN"),
        ("LIN_month", "month","_LIN"),
        ("SNR_3",     "3",    "_SNR"),
        ("SNR_5",     "5",    "_SNR"),
        ("SNR_10",    "10",   "_SNR"),
    ]

    # compute summed significance per bin
    summary = []
    for name, loop_item, suffix in configs:
        all_sig = []
        for src in sources:
            cn = clean_name(src)
            fpath = os.path.join(fits_dir, f"{cn}_fit_data{suffix}.fits")
            ts = read_TS(fpath, loop_filter=loop_item)
            sig = np.sqrt(np.clip(ts, 0, None))
            all_sig.append(sig)
        all_sig = np.vstack(all_sig)  # shape (n_src, 7)
        sum_sig = np.nansum(all_sig, axis=0)
        row = {"Dataset": name}
        for i, v in enumerate(sum_sig, start=1):
            row[f"Bin{i}"] = v
        summary.append(row)

    # build DataFrame
    df = pd.DataFrame(summary)
    df.set_index("Dataset", inplace=True)

    # compute delta rows
    base = df.loc["NONE"]
    for name in df.index.drop("NONE"):
        delta = df.loc[name] - base
        df.loc[f"Δ{name}"] = delta

    # reset index for output
    df = df.reset_index()

    # LaTeX output
    cols = ["Dataset"] + [f"Bin{i}" for i in range(1,8)]
    col_fmt = "l " + " ".join("r" for _ in cols[1:])
    header = " & ".join(cols) + r" \\"
    ncols = len(cols)

    with open(output_tex, "w") as out:
        out.write(f"""\\begin{{longtable}}{{{col_fmt}}}
\\caption{{Summed significance per bin and deltas vs NONE\\label{{tab:sum_sig_deltas}}}}\\\\
\\toprule
{header}
\\midrule
\\endfirsthead

\\multicolumn{{{ncols}}}{{c}}{{{{\\tablename\\ \\thetable{{}} -- continued}}}} \\\\
\\toprule
{header}
\\midrule
\\endhead

\\midrule \\multicolumn{{{ncols}}}{{r}}{{Continued on next page}} \\\\
\\endfoot

\\bottomrule
\\endlastfoot
""")
        out.write(
            df.to_latex(
                index=False,
                columns=cols,
                float_format="%.2f",
                na_rep="",
                header=False,
                column_format=col_fmt
            )
        )
        out.write("\n\\end{longtable}\n")

    return df

# Example usage:
if __name__ == "__main__":
    '''df = maketable_best_fit_all_deltaChi(
        AGN_list=SOURCE_LIST,
        none_pickle="none_new0_no_sys_error.pkl",
        lin_pickle="lin_new0_no_sys_error.pkl",
        snr_pickle="snr_new0_no_sys_error.pkl",
        fermi_fits=FERMI_CAT_FiTS,
        target_m=ma,      # your desired m
        target_g=ga,      # your desired g
        # if you omit target_m/target_g it will use target_d below:
        # target_d=-1.072,
        #output_tex="all_fits.tex"
    )
    print(df)
    de = maketable_best_fit_all_Chi(
    AGN_list=SOURCE_LIST,
    none_pickle="none_new0_no_sys_error.pkl",
    lin_pickle="lin_new0_no_sys_error.pkl",
    snr_pickle="snr_new0_no_sys_error.pkl",
    fermi_fits=FERMI_CAT_FiTS,
    target_m=ma,
    target_g=ga,
    target_d=-1.072,
    output_prefix="all_fits"
    )
    print(de)

    maketable_TS_total( 
    "Source_ra_dec_specin.txt",
    "./fit_results",
    output_prefix="Table")

    df_comp = maketable_TS_comparison(
    agn_list_file="Source_ra_dec_specin.txt",
    fits_dir="./fit_results",
    output_tex="all_sources_TS_comparison.tex")
    print(df_comp.head())

    make_TS_per_bin_tables(
    agn_list_file="Source_ra_dec_specin.txt",
    fits_dir="./fit_results",
    output_dir=None
    ) 
    df = maketable_TS_total_comparison(
     agn_list_file="Source_ra_dec_specin.txt",
     fits_dir="./fit_results",
     output_tex="all_sources_TS_total_comparison.tex"
     ) 
     
    df_summary = maketable_sum_signif_per_bin(
    agn_list_file="Source_ra_dec_specin.txt",
    fits_dir="./fit_results",
    output_tex="sum_signif_per_bin.tex"
)     '''

    maketable_sum_signif_per_bin_with_deltas(agn_list_file="Source_ra_dec_specin.txt",
    fits_dir="./fit_results",
    output_tex = "sum_signif_per_bin_with_deltas.tex")