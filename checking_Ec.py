import pickle

# Load the data
with open("all_results_none_new0_sys_error.pkl", "rb") as file:
    all_results_none_sys = pickle.load(file)

# Choose the source you're interested in
source_name = '4FGL J0038.2-2459'

# Extract results (this will be a dictionary with keys like 'No_Filtering')
results_dict = all_results_none_sys[source_name]

# We're assuming you want the "No_Filtering" dataset
results_list = results_dict['No_Filtering']  # shape: [mass_rows][g_cols]

# Flatten the nested result list into one long list of dicts with metadata
flat_results = []
for row in results_list:
    for entry in row:
        p0 = entry["p0"]
        Ec = entry["E_c"]
        chi2_base = entry["fit_result"]["Base"]["chi2"]
        dof_base = entry["fit_result"]["Base"]["dof"]
        chi2_axion = entry["fit_result"]["Axion"]["chi2"]
        dof_axion = entry["fit_result"]["Axion"]["dof"]
        flat_results.append({
            "p0": p0,
            "Ec": Ec,
            "chi2_base": chi2_base,
            "dof_base": dof_base,
            "chi2_axion": chi2_axion,
            "dof_axion": dof_axion
        })

# Sort by Ec
flat_results_sorted = sorted(flat_results, key=lambda x: x["Ec"])

# Print nicely
print("E_c [MeV]      p₀         χ²_base/dof      χ²_axion/dof")
print("------------------------------------------------------------")
for r in flat_results_sorted:
    chi2_base_red = r["chi2_base"] / r["dof_base"]
    chi2_axion_red = r["chi2_axion"] / r["dof_axion"]
    print(f"{r['Ec']:.3e}   {r['p0']:.3e}   {chi2_base_red:.3f}          {chi2_axion_red:.3f}")
