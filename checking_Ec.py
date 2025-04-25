import pickle

# Load the data
with open("none_new0_no_sys_error.pkl", "rb") as file:
    all_results_none_sys = pickle.load(file)

# Choose the source you're interested in
source_name = "4FGL J0319.8+4130" #16

# Extract results
results_dict = all_results_none_sys[source_name]
results_list = results_dict['No_Filtering']  # shape: [mass_rows][g_cols]

# Flatten and collect values
flat_results = []
for row in results_list:
    for entry in row:
        m = entry["m"]
        g = entry["g"]
        p0 = entry["p0"]
        Ec = entry["E_c"]
        chi2_base = entry["fit_result"]["Base"]["chi2"]
        dof_base = entry["fit_result"]["Base"]["dof"]
        chi2_axion = entry["fit_result"]["Axion"]["chi2"]
        dof_axion = entry["fit_result"]["Axion"]["dof"]
        delta_chi2 = chi2_axion - chi2_base
        flat_results.append({
            "m": m/1e-9,
            "g":g,
            "p0": p0,
            "Ec": Ec,
            "chi2_base": chi2_base,
            "chi2_axion": chi2_axion,
            "delta_chi2": delta_chi2,
        })

# Sort by Ec
flat_results_sorted = sorted(flat_results, key=lambda x: x["delta_chi2"])

# Print results
print("m      g         E_c [MeV]      p₀         χ²_base      χ²_axion      Δχ²")
print("------------------------------------------------------------")
for r in flat_results_sorted:
    print(f"{r['m']:.3e}   {r['g']:.3e}  {r['Ec']:.3e}   {r['p0']:.3e}   {r['chi2_base']:.3f}      {r['chi2_axion']:.3f}      {r['delta_chi2']:.3f}")
import math
target_m = .9      # in neV
target_g = 2e-12    # in GeV⁻¹
best = min(flat_results,
           key=lambda r: math.hypot(r['m'] - target_m,
                                    (r['g'] - target_g)/1e-12)
          )

print("Closest match:")
print(f"  mₐ     = {best['m']:.3f} neV")
print(f"  gₐγ    = {best['g']:.2e} GeV⁻¹")
print(f"  E_c    = {best['Ec']:.3e} MeV")
print(f"  p₀     = {best['p0']:.3f}")
print(f"  Δχ²    = {best['delta_chi2']:.3f}")
