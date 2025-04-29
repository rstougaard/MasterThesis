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

def maketable_TS_per_bin(
    agn_list_file,
    fits_dir,
    output_prefix=None
):
    """
    Builds LaTeX tables of TS per bin for datasets NONE, LIN_week, LIN_month, SNR_3, SNR_5, SNR_10,
    and adds an average-TS column.
    Returns dict of DataFrames.
    """
    import shlex, numpy as np, pandas as pd
    from astropy.io import fits
    datasets={
        'NONE':'_fit_data_NONE.fits',
        'LIN_week':'_fit_data_LIN.fits',
        'LIN_month':'_fit_data_LIN.fits',
        'SNR_3':'_fit_data_SNR.fits',
        'SNR_5':'_fit_data_SNR.fits',
        'SNR_10':'_fit_data_SNR.fits'
    }
    filters={'NONE':None,'LIN_week':lambda r:r['loop_item']=='week','LIN_month':lambda r:r['loop_item']=='month',
             'SNR_3':lambda r:r['loop_item']=='3','SNR_5':lambda r:r['loop_item']=='5','SNR_10':lambda r:r['loop_item']=='10'}
    tables={ds:[] for ds in datasets}
    with open(agn_list_file) as f:
        for line in f:
            parts=shlex.split(line.strip())
            if not parts: continue
            src=parts[0]
            fname=src.replace(' ','').replace('+','plus').replace('-','minus').replace('.','dot').replace('"','')
            for ds,suf in datasets.items():
                path=f"{fits_dir}/{fname}{suf}"
                try: hdul=fits.open(path)
                except FileNotFoundError: continue
                data=hdul[1].data
                sel=data if filters[ds] is None else data[np.array([filters[ds](row) for row in data])]
                sd=sel[np.argsort(sel['emin'])]
                ts=list(sd['TS'])[:7]
                ts+=[np.nan]*(7-len(ts))
                avg_ts=np.nanmean(ts)
                row={'Source':src,**{f'bin{i+1}':ts[i] for i in range(7)},'sqrt_avg_TS':np.sqrt(avg_ts)}
                tables[ds].append(row)
                hdul.close()
    dfs={}
    for ds,rows in tables.items():
        df=pd.DataFrame(rows)
        if output_prefix:
            # Save CSV
            csv=f"{output_prefix}_{ds}_TS_per_bin.csv"; df.to_csv(csv,index=False)
            # Write LaTeX
            tex=f"{output_prefix}_{ds}_TS_per_bin.tex"
            with open(tex,'w') as out:
                out.write(r"""\begin{longtable}{l r r r r r r r r}
\caption{TS per bin for dataset: %s\label{tab:ts_%s}}\\
\toprule
Source & Bin1 & Bin2 & Bin3 & Bin4 & Bin5 & Bin6 & Bin7 & Av\_TS\\
\midrule
\endfirsthead
\multicolumn{9}{c}%%
{{\tablename\ \thetable{} -- continued}} \\
\toprule
Source & Bin1 & Bin2 & Bin3 & Bin4 & Bin5 & Bin6 & Bin7 & Av\_TS\\
\midrule
\endhead
\midrule \multicolumn{9}{r}{{Continued on next page}} \\
\endfoot
\bottomrule
\endlastfoot
""" % (ds,ds))
                out.write(df.to_latex(index=False,float_format="%.2f",na_rep="",column_format='lrrrrrrr r'))
                out.write("\n\\end{longtable}\n")
        dfs[ds]=df
    return dfs


def compare_avg_TS(dfs):
    """
    Given the dict of TS-per-bin DataFrames (with 'avg_TS'),
    computes for each filtering dataset the difference vs NONE.
    Returns a DataFrame with columns: Source, avg_TS_NONE, avg_TS_DS, delta_avg_TS.
    """
    none = dfs.get('NONE').set_index('Source')
    comparisons = {}
    for ds, df in dfs.items():
        if ds=='NONE': continue
        tmp = df.set_index('Source')[['avg_TS']].rename(columns={'avg_TS':f'avg_TS_{ds}'})
        comp = none[['avg_TS']].rename(columns={'avg_TS':'avg_TS_NONE'}).join(tmp)
        comp[f'delta_avg_TS_{ds}'] = comp[f'avg_TS_{ds}'] - comp['avg_TS_NONE']
        comparisons[ds] = comp.reset_index()
    return comparisons

# Example usage:
if __name__ == "__main__":
    df = maketable_best_fit_all_deltaChi(
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
    target_m=None,
    target_g=None,
    target_d=-1.072,
    output_prefix="all_fits"
    )
    print(de)

    maketable_TS_per_bin( 
    "Source_ra_dec_specin.txt",
    "./fit_results",
    output_prefix="Table")

#here is change

