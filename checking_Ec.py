import pickle

# Load the data
with open("none_new0_sys_error.pkl", "rb") as file:
    all_results_none_sys = pickle.load(file)

# Choose the source you're interested in
source_name = '4FGL J1242.9+7315' #16

# Extract results
results_dict = all_results_none_sys[source_name]
results_list = results_dict['No_Filtering']  # shape: [mass_rows][g_cols]

# Flatten and collect values
flat_results = []
for row in results_list:
    for entry in row:
        p0 = entry["p0"]
        Ec = entry["E_c"]
        chi2_base = entry["fit_result"]["Base"]["chi2"]
        dof_base = entry["fit_result"]["Base"]["dof"]
        chi2_axion = entry["fit_result"]["Axion"]["chi2"]
        dof_axion = entry["fit_result"]["Axion"]["dof"]
        delta_chi2 = chi2_axion - chi2_base
        flat_results.append({
            "p0": p0,
            "Ec": Ec,
            "chi2_base": chi2_base,
            "chi2_axion": chi2_axion,
            "delta_chi2": delta_chi2,
        })

# Sort by Ec
flat_results_sorted = sorted(flat_results, key=lambda x: x["Ec"])

# Print results
print("E_c [MeV]      p₀         χ²_base      χ²_axion      Δχ²")
print("------------------------------------------------------------")
for r in flat_results_sorted:
    print(f"{r['Ec']:.3e}   {r['p0']:.3e}   {r['chi2_base']:.3f}      {r['chi2_axion']:.3f}      {r['delta_chi2']:.3f}")
