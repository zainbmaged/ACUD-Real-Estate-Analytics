"""
ACUD dashboard backend: every function returns JSON-safe dicts built from the SAME agents._load() data,
so dashboard numbers and agent recommendations never disagree.
"""
import numpy as np
import pandas as pd
import agents
from agents import _load, _clean, _mm, DATA_STATUS

SERVICES = ["transport", "healthcare", "education", "security", "shopping", "cleanliness",
            "roads_lighting", "internet", "utilities", "recreation"]


def _months(D):
    return sorted(D["Fact_Inventory_Monthly"].Month.unique())


def meta():
    D = _load()
    p = D["projects"]
    return _clean({
        "developers": D["Dim_Developer"][["Developer_ID", "Developer_Name"]].to_dict("records"),
        "districts": D["Dim_District"][["District_ID", "District_Name", "Lifecycle_Archetype"]].to_dict("records"),
        "projects": p[["Project_ID", "Project_Name", "District_ID", "Developer_ID"]].to_dict("records"),
        "unit_types": sorted(D["Dim_Unit"].Unit_Type.unique()),
        "months": _months(D), "data_status": DATA_STATUS,
        "ml_predictions_loaded": D["ml"] is not None,
    })


def _monthly(D, district_id=None, project_ids=None):
    inv, col = D["Fact_Inventory_Monthly"], D["Fact_Collections_Monthly"]
    if district_id:
        inv, col = inv[inv.District_ID == district_id], col[col.District_ID == district_id]
    if project_ids is not None:
        inv, col = inv[inv.Project_ID.isin(project_ids)], col[col.Project_ID.isin(project_ids)]
    a = inv.groupby("Month").agg(units_sold=("Units_Sold", "sum"), sales_value_egp=("Sales_Value_EGP", "sum"),
                                 cancelled=("Units_Cancelled", "sum"), available_units=("Available_End", "sum")).reset_index()
    b = col.groupby("Month").agg(due=("Amount_Due_EGP", "sum"), collected=("Amount_Collected_EGP", "sum"),
                                 overdue=("Overdue_Balance_EGP", "sum")).reset_index()
    m = a.merge(b, on="Month", how="left")
    m["collection_rate_pct"] = m.collected / m.due.replace(0, np.nan) * 100
    return m


# ------------------------------------------------------------------ 1. Executive overview
def overview():
    D = _load()
    p = D["projects"]
    m = _monthly(D)
    rows = m.to_dict("records")
    recent, prev = m.tail(6), m.iloc[-12:-6]
    growth = (recent.sales_value_egp.sum() / max(prev.sales_value_egp.sum(), 1) - 1) * 100
    alerts = []
    for r in p.sort_values("risk_score", ascending=False).head(5).itertuples():
        alerts.append({"type": "project_risk", "severity": "high" if r.risk_score >= 60 else "medium", "id": r.Project_ID,
                       "text_ar": f"{r.Project_ID} ({r.District_ID}): مخاطرة {r.risk_score:.0f}/100، غير مباع {r.unsold_units} وحدة، عمر مخزون {r.inventory_age_days:.0f} يوم",
                       "text_en": f"{r.Project_ID} ({r.District_ID}): risk {r.risk_score:.0f}/100, {r.unsold_units} unsold units, inventory age {r.inventory_age_days:.0f} days"})
    dk = pd.DataFrame(agents.district_kpis())
    for r in dk.sort_values("service_gap_count", ascending=False).head(2).itertuples():
        alerts.append({"type": "service_gap", "severity": "medium", "id": r.District_ID,
                       "text_ar": f"{r.District_ID}: {r.service_gap_count:.0f} فجوات خدمات وتغطية {r.service_coverage_pct:.0f}%",
                       "text_en": f"{r.District_ID}: {r.service_gap_count:.0f} service gaps, coverage {r.service_coverage_pct:.0f}%"})
    for r in p.sort_values("delay_pts", ascending=False).head(2).itertuples():
        alerts.append({"type": "delay", "severity": "medium", "id": r.Project_ID,
                       "text_ar": f"{r.Project_ID}: تأخر إنشائي {r.delay_pts:.1f} نقطة",
                       "text_en": f"{r.Project_ID}: construction delay {r.delay_pts:.1f} pts"})
    k = agents.portfolio_kpis()
    k["sales_growth_6m_pct"] = growth
    k["total_projects"] = int(len(p)); k["total_units_planned"] = int(p.Units_Planned.sum())
    return _clean({"kpis": k, "monthly": rows, "alerts": alerts, "districts": dk.to_dict("records")})


# ------------------------------------------------------------------ 2. Sales & financial
def sales_financial(district_id=None):
    D = _load()
    p = D["projects"]
    m = _monthly(D, district_id)
    pp = p[p.District_ID == district_id] if district_id else p
    con = D["Fact_Construction_Monthly"]
    con = con[con.Project_ID.isin(pp.Project_ID)].groupby("Month")[["Planned_Cost_EGP", "Actual_Cost_EGP"]].sum().reset_index()
    s, u = D["Fact_Sales"], D["Dim_Unit"]
    s = s[s.Cancel_Date.isna() & (pd.to_datetime(s.Sale_Date) >= pd.Timestamp(_months(D)[-12]))]
    if district_id:
        s = s[s.District_ID == district_id]
    su = s.merge(u[["Unit_ID", "Unit_Type", "Area_m2"]], on="Unit_ID")
    bt = su.groupby("Unit_Type").agg(units=("Unit_ID", "count"), contract_value_egp=("Contract_Value_EGP", "sum"),
                                     area_m2=("Area_m2", "sum")).reset_index()
    bt["avg_selling_price_egp"] = bt.contract_value_egp / bt.units
    bt["revenue_per_m2_egp"] = bt.contract_value_egp / bt.area_m2
    by_d = (pd.DataFrame(agents.district_kpis()))[["District_ID", "District_Name", "sales_value_12m_egp", "units_sold",
                                                    "collection_rate", "avg_margin_pct", "sell_through_pct"]]
    tot = su.Contract_Value_EGP.sum()
    kp = {"total_contract_value_12m_egp": tot, "units_sold_12m": int(len(su)),
          "avg_selling_price_egp": tot / max(len(su), 1), "revenue_per_m2_egp": tot / max(su.Area_m2.sum(), 1),
          "collection_rate_pct": pp.collection_rate.mean() * 100, "avg_assumed_margin_pct": pp.margin_pct.mean(),
          "overdue_balance_egp": pp.overdue_latest_egp.sum(),
          "sales_growth_6m_pct": (m.tail(6).sales_value_egp.sum() / max(m.iloc[-12:-6].sales_value_egp.sum(), 1) - 1) * 100}
    return _clean({"kpis": kp, "monthly": m.to_dict("records"), "cost_monthly": con.to_dict("records"),
                   "by_unit_type": bt.to_dict("records"), "by_district": by_d.to_dict("records"),
                   "notes": ["Contract value is not recognised revenue; margin uses an assumed cost ratio (synthetic).",
                             "قيمة العقود ليست إيرادًا معترفًا به، والهامش مبني على تكلفة افتراضية"]})


# ------------------------------------------------------------------ 3. Inventory & sales risk
def inventory_risk(district_id=None):
    D = _load()
    p = D["projects"].copy()
    if district_id:
        p = p[p.District_ID == district_id]
    bins = [(0, 180, "0-180"), (180, 365, "181-365"), (365, 540, "366-540"), (540, 10 ** 6, "540+")]
    aging = [{"bucket": lbl, "projects": int(((p.inventory_age_days > lo) & (p.inventory_age_days <= hi)).sum()),
              "unsold_units": int(p[(p.inventory_age_days > lo) & (p.inventory_age_days <= hi)].unsold_units.sum())} for lo, hi, lbl in bins]
    m = _monthly(D, district_id)
    cols = ["Project_ID", "Project_Name", "District_ID", "Developer_Name", "Units_Planned", "unsold_units", "unsold_ratio",
            "inventory_age_days", "monthly_velocity", "months_of_inventory", "sales_momentum", "cancel_rate", "collection_rate",
            "delay_pts", "service_gap_count", "risk_score"]
    ml = D["ml"]
    t = p[cols].copy()
    if ml is not None and "Unsold_Risk_Probability" in ml.columns:
        t = t.merge(ml[["Project_ID", "Unsold_Risk_Probability"]], on="Project_ID", how="left")
    t = t.sort_values("risk_score", ascending=False)
    k = {"unsold_units": int(p.unsold_units.sum()), "avg_inventory_age_days": p.inventory_age_days.mean(),
         "sell_through_pct": p.sold_units_observed.sum() / p.Units_Planned.sum() * 100,
         "monthly_sales_velocity": p.monthly_velocity.sum(),
         "months_of_inventory": p.unsold_units.sum() / max(p.monthly_velocity.sum(), 0.1),
         "unsold_risk_rate_pct": (p.risk_score >= 60).mean() * 100, "high_risk_projects": int((p.risk_score >= 60).sum())}
    return _clean({"kpis": k, "aging": aging, "monthly": m[["Month", "available_units", "units_sold", "cancelled"]].to_dict("records"),
                   "projects": t.to_dict("records"), "risk_formula": agents.RISK_WEIGHTS_TXT})


# ------------------------------------------------------------------ 4. Developers
def developers(**filters):
    r = agents.developer_performance(**{k: v for k, v in filters.items() if v not in (None, "")})
    return r


# ------------------------------------------------------------------ 5. Smart city & resident needs
def smart_city():
    D = _load()
    ds, sv = D["Fact_District_Monthly_Status"], D["Fact_District_Service_Indicators"]
    last = ds.Month.max()
    a = ds[ds.Month == last].set_index("District_ID")
    b = sv[sv.Month == sv.Month.max()].set_index("District_ID")
    dd = D["Dim_District"].set_index("District_ID")
    x = a.join(b, rsuffix="_sv").join(dd[["District_Name", "Lifecycle_Archetype"]])
    comp = pd.DataFrame({
        "coverage": x.Essential_Service_Coverage_Pct,
        "infrastructure": x.Infrastructure_Readiness_Pct,
        "transport": x.Transport_Accessibility_Score * 10,
        "access": (x.Healthcare_Access_Score + x.School_Access_Score + x.Retail_Access_Score) / 3 * 10,
        "gaps": (100 - x.Service_Gap_Count * 10).clip(lower=0),
    })
    wts = {"coverage": .30, "infrastructure": .20, "transport": .20, "access": .15, "gaps": .15}
    x["readiness_score"] = sum(comp[k] * v for k, v in wts.items())
    x["capacity_stress"] = x.Service_Capacity_Utilization_Pct > 90
    # Expected demand per district (derived): every unit sold so far is assumed to be occupied eventually, at the district's household size.
    # Projected capacity use = today's utilisation scaled by expected / current residents, with service capacity held at today's level.
    rs = D["Fact_Residents_Monthly"]
    rs = rs[rs.Month == rs.Month.max()].set_index("District_ID")
    x["current_residents"] = rs.Residents
    x["expected_residents_buildout"] = x.Units_Sold_Active_Cum * rs.Avg_Household_Size
    populated = rs.Occupied_Units >= 50          # below this the utilisation figure is a floor value, not a usable base
    x["demand_growth_pct"] = ((x.expected_residents_buildout / x.current_residents - 1) * 100).where(populated)
    x["projected_utilization_pct"] = (x.Service_Capacity_Utilization_Pct * x.expected_residents_buildout / x.current_residents).where(populated)
    x["capacity_stress_buildout"] = x.projected_utilization_pct > 100
    x["capacity_gap_pct"] = (x.projected_utilization_pct - 100).clip(lower=0)      # extra capacity needed on top of today's
    rows = x.reset_index()[["District_ID", "District_Name", "Lifecycle_Archetype", "readiness_score", "Essential_Service_Coverage_Pct",
                            "Service_Capacity_Utilization_Pct", "Infrastructure_Readiness_Pct", "Transport_Accessibility_Score",
                            "Service_Gap_Count", "Occupancy_Pct", "Average_Travel_Time_Min", "Public_Transport_Coverage_Pct",
                            "Healthcare_Access_Score", "School_Access_Score", "Retail_Access_Score", "Utilities_Reliability_Pct",
                            "capacity_stress", "current_residents", "expected_residents_buildout", "demand_growth_pct",
                            "projected_utilization_pct", "capacity_stress_buildout", "capacity_gap_pct"]].to_dict("records")
    comp_rows = comp.reset_index().to_dict("records")
    s = D["survey"]
    surv = {"available": False}
    if not s.empty:
        e = s[s.is_eligible == True]  # noqa: E712
        imp = {k: e[f"q8_importance_{k}_1_5"].mean() for k in SERVICES}
        lvl = {k: e[f"q9_current_level_{k}_1_5"].mean() for k in SERVICES}
        gap = [{"service": k, "importance": imp[k], "current": lvl[k], "gap": imp[k] - lvl[k]} for k in SERVICES]
        gap.sort(key=lambda r: -r["gap"])
        heat = []
        for did, g in e.groupby("district_id"):
            heat.append({"District_ID": did, **{k: g[f"q8_importance_{k}_1_5"].mean() - g[f"q9_current_level_{k}_1_5"].mean() for k in SERVICES}})
        q11 = e.q11_problem_changed_decision.dropna()
        q11v = q11[~q11.str.startswith("لا ينطبق")]
        barrier = (q11v.value_counts() / len(q11v) * 100).reset_index()
        barrier.columns = ["answer", "pct"]
        surv = {"available": True, "responses": int(len(s)), "eligible": int(len(e)),
                "satisfaction_attractiveness_avg_1_5": e.q13_area_attractiveness_1_5.mean(),
                "top_problems": (e.q5_most_impactful_problem.value_counts().head(8)).reset_index().rename(columns={"index": "problem", "q5_most_impactful_problem": "problem", "count": "n"}).to_dict("records"),
                "top_improvement": e.q15_top_improvement_priority.value_counts().head(8).reset_index().set_axis(["priority", "n"], axis=1).to_dict("records"),
                "decision_factors": e.q10_top_3_decision_factors.dropna().str.split("; ").explode().value_counts().head(8).reset_index().set_axis(["factor", "n"], axis=1).to_dict("records"),
                "importance_vs_current": gap, "gap_heatmap": heat, "non_buyer_barrier_rate": barrier.to_dict("records")}
    return _clean({"districts": rows, "readiness_components": comp_rows, "readiness_weights": wts, "survey": surv,
                   "readiness_formula": "Urban Service Readiness = 30% coverage + 20% infrastructure + 20% transport + 15% access (health/school/retail) + 15% (100 - 10 x service gaps)",
                   "demand_note": ["الطلب المتوقع = الوحدات المباعة × متوسط حجم الأسرة في الحي (افتراض أن كل وحدة مباعة ستُسكن)، والاستخدام المتوقع = الاستخدام الحالي × (السكان المتوقعون ÷ الحاليون) بافتراض ثبات السعة؛ لا يُحسب للأحياء غير المأهولة بعد (أقل من 50 وحدة مشغولة)",
                                   "Expected demand = units sold × the district's household size (assumes every sold unit is eventually occupied); projected utilisation = current utilisation × (expected ÷ current residents) with capacity held constant; not computed for districts not yet populated (under 50 occupied units)"],
                   "missing": [["مواقع الخدمات وسعاتها ومسافاتها لكل خدمة على حدة (لا يوجد جدول خدمات؛ السعة متاحة كنسبة استخدام إجمالية فقط)",
                                "Per-service locations, capacities and distances (no per-service table; capacity is only available as an overall utilisation %)"]]})


# ------------------------------------------------------------------ 6. Sustainability
def sustainability():
    D = _load()
    tar = D["Fact_Tariffs_Monthly"].set_index("Month")
    te, tw = tar.Electricity_Tariff_EGP_per_kWh, tar.Water_Tariff_EGP_per_m3
    u = D["Fact_Utilities_Monthly"].merge(D["Fact_Residents_Monthly"][["District_ID", "Month", "Residents"]], on=["District_ID", "Month"], how="left") \
        .merge(D["Fact_Solar_Monthly"][["District_ID", "Month", "Installed_Solar_kWp", "Solar_Generation_kWh"]], on=["District_ID", "Month"], how="left")
    u["elec_cost_egp"] = u.Electricity_kWh * u.Month.map(te)
    u["water_cost_egp"] = u.Water_m3 * u.Month.map(tw)
    u["kwh_per_unit"] = u.Electricity_kWh / u.Occupied_Units.replace(0, np.nan)
    u["m3_per_unit"] = u.Water_m3 / u.Occupied_Units.replace(0, np.nan)
    months = sorted(u.Month.unique())
    l12 = u[u.Month.isin(months[-12:])]
    g = l12.groupby("District_ID").agg(kwh=("Electricity_kWh", "sum"), m3=("Water_m3", "sum"), units=("Occupied_Units", "mean"),
                                       residents=("Residents", "mean"), solar_kwh=("Solar_Generation_kWh", "sum"),
                                       elec_cost_egp=("elec_cost_egp", "sum"), water_cost_egp=("water_cost_egp", "sum"),
                                       kwh_per_unit_month=("kwh_per_unit", "mean"), m3_per_unit_month=("m3_per_unit", "mean")).reset_index()
    g = g.merge(D["Dim_District"][["District_ID", "District_Name"]], on="District_ID")
    g["solar_kwp"] = g.District_ID.map(u[u.Month == months[-1]].set_index("District_ID").Installed_Solar_kWp)
    g["renewable_share_pct"] = 100 * g.solar_kwh / g.kwh
    g["elec_cost_per_unit_month_egp"] = g.elec_cost_egp / 12 / g.units.replace(0, np.nan)
    g["m3_per_resident_month"] = g.m3 / 12 / g.residents.replace(0, np.nan)
    g["liters_per_resident_day"] = g.m3_per_resident_month * 1000 / 30.4
    valid = g[g.units > 50]
    bench_e, bench_w = valid.kwh_per_unit_month.quantile(.25), valid.m3_per_unit_month.quantile(.25)
    g["excess_kwh_vs_best_quartile"] = ((g.kwh_per_unit_month - bench_e).clip(lower=0) * g.units * 12).where(g.units > 50)
    g["excess_m3_vs_best_quartile"] = ((g.m3_per_unit_month - bench_w).clip(lower=0) * g.units * 12).where(g.units > 50)
    m = u.groupby("Month").agg(kwh=("Electricity_kWh", "sum"), m3=("Water_m3", "sum"), temp=("Avg_Temperature_C", "mean"), units=("Occupied_Units", "sum"),
                               residents=("Residents", "sum"), solar_kwh=("Solar_Generation_kWh", "sum"),
                               elec_cost_egp=("elec_cost_egp", "sum"), water_cost_egp=("water_cost_egp", "sum")).reset_index()
    m["kwh_per_unit"] = m.kwh / m.units.replace(0, np.nan)
    m["renewable_share_pct"] = 100 * m.solar_kwh / m.kwh
    m["liters_per_resident_day"] = m.m3 * 1000 / m.residents.replace(0, np.nan) / m.Month.map(lambda x: pd.Timestamp(x).days_in_month)
    m["electricity_tariff"] = m.Month.map(te); m["water_tariff"] = m.Month.map(tw)
    corr = float(np.corrcoef(m.temp, m.kwh_per_unit.fillna(m.kwh_per_unit.mean()))[0, 1])

    # ---- solutions: measured (pilot baseline vs after) savings, payback
    cat = D["Sustainability_Solutions"].set_index("Solution_ID")
    dep = D["Solution_Deployments"]; pil = D["Fact_Solution_Pilot_Monthly"]
    tar_e, tar_w = float(te.iloc[-1]), float(tw.iloc[-1])           # latest tariff used to value savings
    rows = []
    for r in dep.itertuples():
        c = cat.loc[r.Solution_ID]; elec = c.Resource == "Electricity"; tariff = tar_e if elec else tar_w
        if r.Solution_ID == "S01":
            sv = D["Fact_Solar_Monthly"]; sv = sv[(sv.District_ID == r.District_ID) & (sv.Month.isin(months[-12:]))]
            # annualise with the specific yield (kWh per installed kWp per year) so young roll-outs are not understated
            kwp_months = float(sv.Installed_Solar_kWp.sum())
            units_saved = float(sv.Solar_Generation_kWh.sum() / kwp_months * 12 * r.Pilot_Units) if kwp_months else 0.0
            pct = None; n_m = int((sv.Installed_Solar_kWp > 0).sum()); unit = "kWh"
        else:
            pp = pil[(pil.Deployment_ID == r.Deployment_ID) & (pil.Solution_Active == 1)]
            base, aft = pp.Baseline_Consumption.sum(), pp.After_Solution_Consumption.sum()
            pct = float(1 - aft / base) * 100 if base else None
            n_m = int(len(pp)); units_saved = float((base - aft) / max(n_m, 1) * 12); unit = "kWh" if elec else "m3"
        sav_egp = units_saved * tariff
        om = r.Capex_EGP * c.OM_Pct_of_Capex_per_Year / 100
        net = sav_egp - om
        rows.append({"Deployment_ID": r.Deployment_ID, "District_ID": r.District_ID, "Solution_ID": r.Solution_ID, "solution_en": c.Solution_Name_EN,
                     "solution_ar": c.Solution_Name_AR, "resource": c.Resource, "Install_Month": r.Install_Month, "scale": r.Pilot_Units,
                     "capex_egp": r.Capex_EGP, "months_measured": n_m, "expected_saving_pct": None if pd.isna(c.Expected_Saving_Pct) else c.Expected_Saving_Pct * 100,
                     "measured_saving_pct": pct, "annual_saving_units": units_saved, "unit": unit, "annual_saving_egp": sav_egp,
                     "annual_om_egp": om, "lifetime_years": float(c.Lifetime_Years), "payback_years": r.Capex_EGP / net if net > 0 else None})
    deployments = pd.DataFrame(rows)

    # ---- scale-up potential: apply each solution's measured saving % to the consumption of all occupied units
    scale = []
    for sid, grp in deployments[deployments.Solution_ID != "S01"].groupby("Solution_ID"):
        c = cat.loc[sid]; elec = c.Resource == "Electricity"
        pooled = float(np.average(grp.measured_saving_pct, weights=grp.months_measured * grp.scale)) / 100
        cons = float(valid.kwh.sum() if elec else valid.m3.sum()); tariff = tar_e if elec else tar_w
        capex = float(c.Capex_EGP_per_basis_unit * valid.units.sum()); sav = pooled * cons * tariff; om = capex * c.OM_Pct_of_Capex_per_Year / 100
        scale.append({"Solution_ID": sid, "solution_en": c.Solution_Name_EN, "solution_ar": c.Solution_Name_AR, "resource": c.Resource,
                      "measured_saving_pct": pooled * 100, "capex_egp": capex, "annual_saving_egp": sav,
                      "payback_years": capex / (sav - om) if sav > om else None, "lifetime_years": float(c.Lifetime_Years),
                      "pilots": int(len(grp))})
    scale = pd.DataFrame(scale).sort_values("payback_years", na_position="last")

    tot_capex, tot_net = deployments.capex_egp.sum(), (deployments.annual_saving_egp - deployments.annual_om_egp).sum()
    k = {"electricity_kwh_12m": g.kwh.sum(), "water_m3_12m": g.m3.sum(),
         "kwh_per_occupied_unit_month": valid.kwh_per_unit_month.mean(), "m3_per_occupied_unit_month": valid.m3_per_unit_month.mean(),
         "indicative_excess_kwh_12m": g.excess_kwh_vs_best_quartile.sum(), "indicative_excess_m3_12m": g.excess_m3_vs_best_quartile.sum(),
         "temp_vs_kwh_correlation": corr,
         "electricity_cost_12m_egp": g.elec_cost_egp.sum(), "water_cost_12m_egp": g.water_cost_egp.sum(),
         "elec_cost_per_unit_month_egp": float(g.elec_cost_egp.sum() / 12 / g.units.sum()),
         "renewable_share_pct_12m": float(100 * g.solar_kwh.sum() / g.kwh.sum()), "solar_generation_kwh_12m": g.solar_kwh.sum(),
         "installed_solar_kwp": g.solar_kwp.sum(), "residents_latest": float(g.residents.sum()),
         "liters_per_resident_day": float(g.m3.sum() * 1000 / 12 / g.residents.sum() / 30.4),
         "tariff_electricity_egp_per_kwh": tar_e, "tariff_water_egp_per_m3": tar_w,
         "solutions_capex_egp": tot_capex, "solutions_net_annual_saving_egp": tot_net,
         "portfolio_payback_years": tot_capex / tot_net if tot_net > 0 else None}
    return _clean({"kpis": k, "districts": g.to_dict("records"), "monthly": m.to_dict("records"),
                   "deployments": deployments.to_dict("records"), "scale_up": scale.to_dict("records"),
                   "benchmark": {"kwh_per_unit_month_best_quartile": bench_e, "m3_per_unit_month_best_quartile": bench_w},
                   "completeness": "complete (synthetic inputs)",
                   "missing": ["Real metered data, tariff slabs and vendor quotes: renewable share, tariffs, solution costs, before/after consumption and resident counts "
                               "are synthetic scenario assumptions (generate_sustainability_data.py) to be replaced with real data"]})


# ------------------------------------------------------------------ project drill-down (used by agents -> dashboard link)
def project_detail(project_id):
    D = _load()
    p = D["projects"]
    row = p[p.Project_ID == project_id]
    if row.empty:
        return None
    inv = D["Fact_Inventory_Monthly"]; col = D["Fact_Collections_Monthly"]; con = D["Fact_Construction_Monthly"]; fun = D["Fact_Project_Sales_Funnel"]
    f = lambda t: t[t.Project_ID == project_id].sort_values("Month")
    mon = f(inv)[["Month", "Units_Sold", "Units_Cancelled", "Available_End", "Avg_Inventory_Age_Days", "Sales_Value_EGP"]] \
        .merge(f(col)[["Month", "Amount_Due_EGP", "Amount_Collected_EGP", "Overdue_Balance_EGP"]], on="Month") \
        .merge(f(con)[["Month", "Planned_Pct", "Actual_Pct", "Handover_Pct"]], on="Month") \
        .merge(f(fun)[["Month", "Leads", "Viewings", "Bookings"]], on="Month")
    r = row.iloc[0].to_dict()
    return _clean({"project": r, "monthly": mon.to_dict("records")})
