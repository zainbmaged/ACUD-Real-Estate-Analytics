"""Synthetic data for the sustainability items that were missing from the prototype dataset.

Run:  python generate_sustainability_data.py      (deterministic, seed 7; overwrites the CSVs below in ./data)

Tables written
  Fact_Tariffs_Monthly.csv        city-wide blended residential tariffs (EGP) per month
  Fact_Residents_Monthly.csv      residents per district-month (occupied units x household size)
  Fact_Solar_Monthly.csv          installed kWp, generation, renewable share per district-month
  Sustainability_Solutions.csv    solution catalogue: capex, expected saving, lifetime, O&M
  Solution_Deployments.csv        which solution was installed where / when / how big / capex
  Fact_Solution_Pilot_Monthly.csv baseline (counterfactual) vs after-solution consumption of each pilot group

Everything is a SYNTHETIC SCENARIO ASSUMPTION, built on top of the existing Fact_Utilities_Monthly (which is left unchanged)
so that totals stay consistent. Replace with metered data / real tariffs / vendor quotes before real decisions.
"""
import os

import numpy as np
import pandas as pd

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
rng = np.random.default_rng(7)
u = pd.read_csv(os.path.join(DATA, "Fact_Utilities_Monthly.csv"))
dist = pd.read_csv(os.path.join(DATA, "Dim_District.csv"))
months = sorted(u.Month.unique())
mdt = pd.to_datetime(pd.Series(months))
STAT = "Synthetic assumption"

# ------------------------------------------------------------------ 1. tariffs (blended residential, EGP)
def tariff(m):
    d = pd.Timestamp(m)
    e = 1.55 if d < pd.Timestamp("2024-08-01") else 1.80 if d < pd.Timestamp("2025-07-01") else 2.05
    w = 5.00 if d < pd.Timestamp("2024-07-01") else 5.75 if d < pd.Timestamp("2025-07-01") else 6.60
    return e, w

tar = pd.DataFrame([{"Month": m, "Electricity_Tariff_EGP_per_kWh": tariff(m)[0], "Water_Tariff_EGP_per_m3": tariff(m)[1],
                     "Tariff_Basis": "Blended residential tariff, stepped increases (scenario)", "Data_Status": STAT} for m in months])
tar.to_csv(os.path.join(DATA, "Fact_Tariffs_Monthly.csv"), index=False)

# ------------------------------------------------------------------ 2. residents
hh = {"High-income / established": 3.6, "Middle-income families": 4.1, "Mixed": 3.8, "Off-plan buyers / investors": 3.4,
      "Upper-middle (price above segment)": 3.9}
dd = dist.set_index("District_ID").Target_Resident_Segment.map(hh).to_dict()
rows = []
for r in u.itertuples():
    size = dd[r.District_ID] * (1 + 0.012 * months.index(r.Month) / 12) * (1 + rng.normal(0, 0.01))   # households grow slightly
    rows.append({"District_ID": r.District_ID, "Month": r.Month, "Occupied_Units": r.Occupied_Units,
                 "Avg_Household_Size": round(size, 2), "Residents": int(round(r.Occupied_Units * size)), "Data_Status": STAT})
res = pd.DataFrame(rows)
res.to_csv(os.path.join(DATA, "Fact_Residents_Monthly.csv"), index=False)

# ------------------------------------------------------------------ 3. solar (rooftop PV on common areas / rooftops)
YIELD = {1: 3.6, 2: 4.4, 3: 5.3, 4: 6.0, 5: 6.6, 6: 7.0, 7: 6.9, 8: 6.6, 9: 5.9, 10: 4.8, 11: 3.8, 12: 3.3}   # kWh per kWp per day, PR included
# final kWp per district and the month the roll-out starts (0 = no solar yet)
SOLAR = {"D01": (220, "2024-02-01"), "D02": (170, "2024-05-01"), "D03": (90, "2024-09-01"), "D04": (80, "2024-11-01"),
         "D05": (40, "2025-03-01"), "D06": (60, "2025-01-01"), "D09": (25, "2025-06-01"), "D10": (30, "2025-04-01")}
rows = []
for r in u.itertuples():
    final, start = SOLAR.get(r.District_ID, (0, None))
    kwp = 0.0
    if final and pd.Timestamp(r.Month) >= pd.Timestamp(start):
        n = (pd.Timestamp(r.Month).year - pd.Timestamp(start).year) * 12 + pd.Timestamp(r.Month).month - pd.Timestamp(start).month
        kwp = round(final * min(1.0, 0.25 + 0.75 * n / 14), 1)          # phased roll-out over ~14 months
    mo = pd.Timestamp(r.Month)
    gen = kwp * YIELD[mo.month] * mo.days_in_month * float(rng.normal(1.0, 0.04))
    rows.append({"District_ID": r.District_ID, "Month": r.Month, "Installed_Solar_kWp": kwp, "Solar_Generation_kWh": round(gen, 0),
                 "Renewable_Share_Pct": round(100 * gen / r.Electricity_kWh, 2) if r.Electricity_kWh else 0.0, "Data_Status": STAT})
sol = pd.DataFrame(rows)
sol.to_csv(os.path.join(DATA, "Fact_Solar_Monthly.csv"), index=False)

# ------------------------------------------------------------------ 4. solution catalogue
CAT = pd.DataFrame([
    ("S01", "Rooftop solar PV", "الطاقة الشمسية على الأسطح", "Electricity", "kWp", 18000, None, 25, 1.5),
    ("S02", "LED + smart lighting (common areas)", "إنارة LED ذكية للمناطق المشتركة", "Electricity", "occupied unit", 1200, 0.06, 12, 1.0),
    ("S03", "High-efficiency AC / smart thermostats", "تكييف عالي الكفاءة وترموستات ذكي", "Electricity", "occupied unit", 4500, 0.14, 12, 2.0),
    ("S04", "Smart water metering + leak detection", "عدادات مياه ذكية وكشف التسريب", "Water", "occupied unit", 600, 0.08, 10, 3.0),
    ("S05", "Low-flow fixtures", "تركيبات مياه منخفضة التدفق", "Water", "occupied unit", 700, 0.12, 10, 0.5),
    ("S06", "Greywater reuse for irrigation", "إعادة استخدام المياه الرمادية للري", "Water", "occupied unit", 1500, 0.18, 20, 3.0),
], columns=["Solution_ID", "Solution_Name_EN", "Solution_Name_AR", "Resource", "Cost_Basis", "Capex_EGP_per_basis_unit",
            "Expected_Saving_Pct", "Lifetime_Years", "OM_Pct_of_Capex_per_Year"])
CAT["Data_Status"] = STAT
CAT.to_csv(os.path.join(DATA, "Sustainability_Solutions.csv"), index=False)

# ------------------------------------------------------------------ 5. deployments + pilot baseline vs after
cap = CAT.set_index("Solution_ID")
PILOTS = [  # district, solution, install month, pilot units
    ("D01", "S02", "2024-04-01", 300), ("D01", "S05", "2024-07-01", 250), ("D02", "S03", "2024-06-01", 200), ("D02", "S04", "2024-10-01", 300),
    ("D03", "S02", "2024-09-01", 200), ("D03", "S05", "2025-01-01", 150), ("D04", "S03", "2025-01-01", 120), ("D04", "S04", "2024-11-01", 200),
    ("D05", "S02", "2025-03-01", 100), ("D06", "S05", "2025-02-01", 150), ("D06", "S06", "2025-04-01", 80), ("D09", "S04", "2025-06-01", 80),
    ("D10", "S03", "2025-05-01", 100)]
dep, pilot = [], []
for i, (d, s, inst, n) in enumerate(PILOTS, 1):
    dep.append({"Deployment_ID": f"DEP{i:02d}", "District_ID": d, "Solution_ID": s, "Install_Month": inst, "Pilot_Units": n,
                "Capex_EGP": round(n * cap.loc[s, "Capex_EGP_per_basis_unit"] * float(rng.normal(1.0, 0.06)), -2), "Status": "Implemented (pilot)", "Data_Status": STAT})
    elec = cap.loc[s, "Resource"] == "Electricity"
    sav = cap.loc[s, "Expected_Saving_Pct"] * float(rng.normal(1.0, 0.12))       # realised saving differs from the expected one
    for m in months:
        ur = u[(u.District_ID == d) & (u.Month == m)].iloc[0]
        per_unit = (ur.Electricity_kWh if elec else ur.Water_m3) / ur.Occupied_Units
        base = n * per_unit * float(rng.normal(1.0, 0.025))
        k = (pd.Timestamp(m).year - pd.Timestamp(inst).year) * 12 + pd.Timestamp(m).month - pd.Timestamp(inst).month
        eff = 0.0 if k < 0 else sav * min(1.0, (k + 1) / 3)                      # 3-month ramp-up
        if s == "S03":                                                            # AC saving is larger in hot months
            eff *= 0.6 + 0.8 * np.clip((ur.Avg_Temperature_C - 14) / 22, 0, 1)
        after = base * (1 - eff) * float(rng.normal(1.0, 0.012))
        pilot.append({"Deployment_ID": f"DEP{i:02d}", "District_ID": d, "Solution_ID": s, "Month": m, "Pilot_Units": n,
                      "Unit": "kWh" if elec else "m3", "Baseline_Consumption": round(base, 1), "After_Solution_Consumption": round(after, 1),
                      "Solution_Active": int(k >= 0), "Data_Status": STAT})
# one solar deployment per district that has PV (capex from final installed kWp)
j = len(dep)
for d, (final, start) in SOLAR.items():
    j += 1
    dep.append({"Deployment_ID": f"DEP{j:02d}", "District_ID": d, "Solution_ID": "S01", "Install_Month": start, "Pilot_Units": final,
                "Capex_EGP": round(final * cap.loc["S01", "Capex_EGP_per_basis_unit"] * float(rng.normal(1.0, 0.05)), -3),
                "Status": "Implemented (roll-out)", "Data_Status": STAT})
pd.DataFrame(dep).to_csv(os.path.join(DATA, "Solution_Deployments.csv"), index=False)
pd.DataFrame(pilot).to_csv(os.path.join(DATA, "Fact_Solution_Pilot_Monthly.csv"), index=False)

# ------------------------------------------------------------------ provenance / grains
prov = pd.read_csv(os.path.join(DATA, "Data_Provenance.csv"))
grains = pd.read_csv(os.path.join(DATA, "Table_Grains.csv"))
G = {"Fact_Tariffs_Monthly": ("City x month", "Month"), "Fact_Residents_Monthly": ("District x month", "District_ID + Month"),
     "Fact_Solar_Monthly": ("District x month", "District_ID + Month"), "Sustainability_Solutions": ("Solution", "Solution_ID"),
     "Solution_Deployments": ("Deployment", "Deployment_ID"), "Fact_Solution_Pilot_Monthly": ("Deployment x month", "Deployment_ID + Month")}
prov = prov[~prov.Dataset.str.replace(".csv", "", regex=False).isin(G)]
grains = grains[~grains.iloc[:, 0].isin(G)]
prov = pd.concat([prov, pd.DataFrame([{"Dataset": f"{k}.csv", "Provenance_Type": "Synthetic assumption", "Generation_or_Source": "generate_sustainability_data.py (seed 7)",
                                       "Data_Status": "Synthetic", "Notes": "Fills missing sustainability inputs; replace with metered data / real tariffs / vendor quotes"} for k in G])])
grains = pd.concat([grains, pd.DataFrame([{grains.columns[0]: k, grains.columns[1]: v[0], grains.columns[2]: v[1]} for k, v in G.items()])])
prov.to_csv(os.path.join(DATA, "Data_Provenance.csv"), index=False)
grains.to_csv(os.path.join(DATA, "Table_Grains.csv"), index=False)

if __name__ == "__main__":
    l12 = sol[sol.Month >= months[-12]].groupby("District_ID")[["Solar_Generation_kWh"]].sum()
    e12 = u[u.Month >= months[-12]].groupby("District_ID").Electricity_kWh.sum()
    print("renewable share last 12m by district (%):\n", (100 * l12.Solar_Generation_kWh / e12).round(1).to_dict())
    print("city renewable share %:", round(100 * l12.Solar_Generation_kWh.sum() / e12.sum(), 1))
    print("deployments:", len(dep), "| pilot rows:", len(pilot))
