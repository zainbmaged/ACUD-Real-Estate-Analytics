"""Build the hand-off package for teammates:   python export_handoff.py   ->  ./handoff/  (+ handoff.zip)
Everything is computed from the same code the agents and dashboard use, so numbers never disagree."""
import json
import os
import shutil
import numpy as np
import pandas as pd
import agents
import dashboard

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "handoff")
shutil.rmtree(OUT, ignore_errors=True)
for d in ("for_powerbi", "for_ml", "for_smart_city_sustainability", "agents_contract"):
    os.makedirs(os.path.join(OUT, d))
w = lambda d, name, df: df.to_csv(os.path.join(OUT, d, name), index=False, encoding="utf-8-sig")
D = agents._load()
p = D["projects"]

# ---------------------------------------------------------------- 1. Power BI / dashboard owner
pb = "for_powerbi"
w(pb, "kpi_projects.csv", pd.DataFrame(agents.project_kpis()))
w(pb, "kpi_districts.csv", pd.DataFrame(agents.district_kpis()))
w(pb, "kpi_portfolio.csv", pd.DataFrame([agents.portfolio_kpis()]))
ov = dashboard.overview()
w(pb, "monthly_portfolio.csv", pd.DataFrame(ov["monthly"]))
dv = agents.run_agent("developer_performance")["results"]
w(pb, "developer_scores.csv", pd.DataFrame(dv["ranking"]).drop(columns=["red_flags"]).assign(
    red_flags=[" | ".join(x) for x in [r["red_flags"] for r in dv["ranking"]]]))
w(pb, "unsold_inventory_risk_score.csv", p[["Project_ID", "District_ID", "Developer_ID", "unsold_units", "unsold_ratio", "inventory_age_days",
                                              "months_of_inventory", "monthly_velocity", "collection_rate", "delay_pts", "cancel_rate",
                                              "service_gap_count", "risk_score"]])
# Investment Attractiveness Score for 3 reference budgets (so BI can show it without calling the agent)
rows = []
for b in (6e6, 10e6, 20e6):
    for r in agents.run_agent("investment_advisor", budget_egp=b, top_n=40, horizon_years=5)["results"]["top_projects"]:
        rows.append({"budget_egp": b, "horizon_years": 5, "Project_ID": r["Project_ID"], "ticket_egp": r["ticket"], "units_in_budget": r["units_in_budget"],
                     "irr_conservative_pct": r["irr_cons_pct"], "irr_base_pct": r["irr_base_pct"], "irr_real_base_pct": r["irr_real_base_pct"],
                     "net_profit_base_egp": r["net_profit_base_egp"], "rent_income_base_egp": r["rent_income_base_egp"],
                     "gross_rent_yield_pct": r["gross_rent_yield_pct"], "net_rent_yield_pct": r["net_rent_yield_pct"],
                     "months_to_handover": r["months_to_handover"], "exit_depth_months": r["exit_depth_months"], "exit_depth_label": r["exit_depth_label"],
                     "attractiveness_score": r["attractiveness_score"], "score_complete": r["score_complete"],
                     **{"comp_" + k: v for k, v in r["components"].items()}})
w(pb, "investment_attractiveness_reference_budgets.csv", pd.DataFrame(rows))

KPI = [  # name, formula, tables/columns, grain, owner, status in this prototype
    ("Total Contract Value", "SUM(Contract_Value_EGP) of non-cancelled sales", "Fact_Sales", "Project x Month", "Dashboard", "Implemented (sales_value_12m)"),
    ("Units Sold", "COUNT non-cancelled sales", "Fact_Sales / Fact_Inventory_Monthly.Units_Sold", "Project x Month", "Dashboard", "Implemented"),
    ("Sales Growth %", "(last 6m sales value - previous 6m) / previous 6m x 100", "Fact_Inventory_Monthly.Sales_Value_EGP", "Portfolio", "Dashboard", "Implemented (sales_growth_6m_pct)"),
    ("Collection Rate %", "SUM(Amount_Collected) / SUM(Amount_Due) x 100, last 12m", "Fact_Collections_Monthly", "Project", "Dashboard", "Implemented"),
    ("Gross Profit Margin %", "(avg contract - assumed cost per unit) / avg contract", "Project_Investment_Inputs", "Project", "Dashboard", "PROXY - cost is an assumed ratio 68-78%"),
    ("Average Selling Price", "Contract value / units sold", "Fact_Sales", "Type / District", "Dashboard", "Implemented"),
    ("Revenue per m2", "Contract value / sold area", "Fact_Sales + Dim_Unit.Area_m2", "Type / District", "Dashboard", "Implemented"),
    ("Unsold Units", "count(Dim_Unit) - sold_units_observed", "Project_Investment_Inputs, Dim_Project", "Project", "Dashboard", "Implemented (reconciled: unit count taken from Dim_Unit = 2,201, matches Fact_Inventory_Monthly; Dim_Project.Units_Planned misses the unplanned P038 +300 release)"),
    ("Sell-Through Rate %", "sold_units_observed / Units_Planned", "Project_Investment_Inputs", "Project", "Dashboard", "Implemented (cumulative, not per-period)"),
    ("Inventory Aging", "Avg_Inventory_Age_Days at last month", "Fact_Inventory_Monthly", "Project", "Dashboard", "Implemented"),
    ("Monthly Sales Velocity", "Units sold last 6 months / 6", "Fact_Inventory_Monthly.Units_Sold", "Project", "Agents", "Implemented"),
    ("Months of Inventory", "Unsold units / monthly velocity (cap 120)", "derived", "Project", "Agents", "Implemented - assumes constant velocity"),
    ("Unsold Risk Rate %", "Projects with risk_score >= 60 / projects", "derived", "Portfolio", "Agents", "Implemented (project level, not unit level)"),
    ("Unsold Inventory Risk Score", "30% unsold ratio + 20% inv age + 15% collection shortfall + 15% delay + 10% cancel rate + 10% service gaps (min-max scaled)", "see unsold_inventory_risk_score.csv", "Project", "Agents", "Implemented - rule based; ML probability plugs in via ML_Predictions.csv"),
    ("Fair Developer Performance Score", "25% sales vs expected + 25% profitability + 20% collection + 15% delivery + 15% inventory", "developer_scores.csv", "Developer", "Agents", "Implemented - weights are initial, untested"),
    ("Construction Progress Variance", "Actual_Pct - Planned_Pct (last month)", "Fact_Construction_Monthly", "Project", "Dashboard", "Implemented (delay_pts = max(planned - actual, 0))"),
    ("On-Time Delivery Rate %", "Units delivered on time / units due", "-", "Developer", "Dashboard", "NOT POSSIBLE - no contractual delivery dates; proxy = share of projects with delay <= 2 pts. Planned / forecast completion months and forecast delay (months) are derived from the construction curves (kpi_projects.csv)"),
    ("Sales Target Achievement %", "Actual / target x 100", "-", "Developer", "Dashboard", "NOT POSSIBLE - no sales targets in data"),
    ("Investment Attractiveness Score", "25% return(IRR base) + 25% risk + 15% liquidity + 15% services + 10% construction + 10% price value", "investment_attractiveness_reference_budgets.csv", "Project x budget", "Agents", "Implemented - shows 'incomplete' if a component is missing"),
    ("Real IRR (after inflation)", "(1 + nominal IRR) / (1 + inflation) - 1; real net profit = sum of cash flows deflated by inflation", "investment_attractiveness_reference_budgets.csv", "Project x budget", "Agents", "Implemented - inflation 13.9% (CAPMAS Sep-2026) is an editable constant assumption"),
    ("Net Rental Yield", "gross yield by unit type x (1 - vacancy) - service charges, on current value; rent starts at handover", "investment_attractiveness_reference_budgets.csv", "Project x budget", "Agents", "SCENARIO ASSUMPTION - no rental data (apartments anchored on Global Property Guide Q4 2025); editable"),
    ("Exit Market Depth", "60% project months of inventory + 40% district months of inventory (unsold / 6-month sales pace)", "kpi_projects.csv (exit_depth_months)", "Project", "Agents", "Derived from sales pace; sets the exit discount (2% + 0.15% per month, cap 10%)"),
    ("Expected Service Demand / Projected Capacity Use", "sold units x household size; current utilisation x expected / current residents (capacity held constant)", "for_smart_city_sustainability/service_readiness.csv", "District", "Smart City", "Derived; not computed for districts with < 50 occupied units"),
    ("Urban Service Readiness Score", "30% coverage + 20% infrastructure + 20% transport + 15% access + 15% (100 - 10 x gaps)", "for_smart_city_sustainability/service_readiness.csv", "District", "Smart City", "Implemented (prototype formula - agree on it with the owner)"),
    ("Service Availability / Capacity Utilization", "Essential_Service_Coverage_Pct / Service_Capacity_Utilization_Pct", "Fact_District_Monthly_Status", "District x Month", "Smart City", "Implemented as given in data"),
    ("Non-Buyer Barrier Rate %", "Respondents choosing option / valid respondents", "synthetic_survey_responses_ar", "District", "Smart City", "Implemented for q11 (decision impact) and q5 (problems)"),
    ("Electricity Cost per Unit", "(kWh x monthly tariff) / occupied unit", "Fact_Utilities_Monthly + Fact_Tariffs_Monthly", "District x Month", "Sustainability", "Implemented (tariffs are synthetic assumptions)"),
    ("Water per Resident", "m3 / residents (litres per resident per day)", "Fact_Utilities_Monthly + Fact_Residents_Monthly", "District x Month", "Sustainability", "Implemented (resident counts are synthetic: occupied units x household size)"),
    ("Renewable Share / Cost Savings / Payback", "solar kWh / total kWh; (baseline - after) x tariff; capex / (saving - O&M)", "Fact_Solar_Monthly, Fact_Solution_Pilot_Monthly, Solution_Deployments, Sustainability_Solutions, Fact_Tariffs_Monthly", "District / Solution", "Sustainability", "Implemented on SYNTHETIC inputs - replace with metered data, real tariffs and vendor quotes"),
    ("Forecast MAE / WAPE", "see spec", "ML predictions on held-out time period", "Project / Portfolio", "ML", "NOT IN THIS CODE - owned by ML teammate"),
]
w(pb, "kpi_dictionary.csv", pd.DataFrame(KPI, columns=["KPI", "Formula", "Tables_or_Columns", "Grain", "Owner", "Status_in_prototype"]))

# ---------------------------------------------------------------- 2. ML owner
ml = "for_ml"
inv, fun, col, con = D["Fact_Inventory_Monthly"], D["Fact_Project_Sales_Funnel"], D["Fact_Collections_Monthly"], D["Fact_Construction_Monthly"]
panel = (inv.merge(fun.drop(columns=["District_ID", "Source_Status"]), on=["Project_ID", "Month"], how="left")
         .merge(col.drop(columns=["District_ID"]), on=["Project_ID", "Month"], how="left")
         .merge(con.drop(columns=["District_ID"]), on=["Project_ID", "Month"], how="left"))
w(ml, "panel_project_month.csv", panel)
w(ml, "features_project_latest.csv", p.drop(columns=["Launch_Date"]))
tmpl = pd.DataFrame([{"Project_ID": "P030", "Unsold_Risk_Probability": 0.0, "Forecast_Horizon_Months": 6, "Model_Name": "your_model_name",
                      "Evaluation_Metric": "WAPE", "Evaluation_WAPE": 0.0, "Evaluation_Period": "2025-07..2025-12 (held out)"}])
w(ml, "ML_Predictions_TEMPLATE.csv", tmpl)
open(os.path.join(OUT, ml, "README_ML.md"), "w", encoding="utf-8").write("""# For the ML teammate
- `panel_project_month.csv`: 40 projects x 24 months (inventory, sales, funnel, collections, construction) - the training table.
- `features_project_latest.csv`: one row per project incl. the current rule-based `risk_score` (use it only as a benchmark, not as a label).
- Deliver predictions as **data/ML_Predictions.csv** (see `ML_Predictions_TEMPLATE.csv`): required columns `Project_ID`, `Unsold_Risk_Probability` (0-1);
  optional `Model_Name`, `Evaluation_WAPE`, `Forecast_Horizon_Months`. Drop it in the `data/` folder - the Unsold Inventory Agent and the dashboard
  pick it up automatically (no code change). Evaluate on a time period NOT used for training and state it in `Evaluation_Period`.
- For forecasting (sales / inventory / collections) provide a long table: `Project_ID, Month, Metric, Actual, Forecast, Lower, Upper`.
- Data is synthetic: 24 months only, so keep the validation window short (e.g. last 6 months) and report WAPE.
""")

# ---------------------------------------------------------------- 3. Smart city + sustainability
sd = "for_smart_city_sustainability"
sc, su = dashboard.smart_city(), dashboard.sustainability()
w(sd, "service_readiness.csv", pd.DataFrame(sc["districts"]))
w(sd, "readiness_components.csv", pd.DataFrame(sc["readiness_components"]))
if sc["survey"]["available"]:
    s = sc["survey"]
    w(sd, "survey_importance_vs_current.csv", pd.DataFrame(s["importance_vs_current"]))
    w(sd, "survey_gap_heatmap_by_district.csv", pd.DataFrame(s["gap_heatmap"]))
    w(sd, "survey_top_problems.csv", pd.DataFrame(s["top_problems"]))
    w(sd, "survey_decision_impact.csv", pd.DataFrame(s["non_buyer_barrier_rate"]))
w(sd, "sustainability_districts.csv", pd.DataFrame(su["districts"]))
w(sd, "sustainability_monthly.csv", pd.DataFrame(su["monthly"]))
w(sd, "sustainability_solutions_measured.csv", pd.DataFrame(su["deployments"]))
w(sd, "sustainability_scale_up_potential.csv", pd.DataFrame(su["scale_up"]))
for t in ("Fact_Tariffs_Monthly", "Fact_Residents_Monthly", "Fact_Solar_Monthly", "Sustainability_Solutions", "Solution_Deployments", "Fact_Solution_Pilot_Monthly"):
    w(sd, t + ".csv", D[t])

# ---------------------------------------------------------------- 4. agents contract (+ real sample outputs)
ac = "agents_contract"
samples = {"investment_advisor": dict(budget_egp=10_000_000, horizon_years=5, risk_tolerance="medium"),
           "developer_performance": dict(developer_id="DEV5"), "unsold_inventory": dict(project_id="P030")}
for n, kw in samples.items():
    json.dump(agents.run_agent(n, **kw), open(os.path.join(OUT, ac, f"sample_{n}.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
open(os.path.join(OUT, ac, "README_AGENTS.md"), "w", encoding="utf-8").write("""# Agents API contract
POST /agents/{investment_advisor|developer_performance|unsold_inventory}  body: {"params": {...}}
POST /agents/ask  body: {"text": "free text"}      GET /kpis/portfolio | /kpis/districts | /kpis/projects
Every agent returns: agent, summary, results, evidence, assumptions, risks, missing_data, data_status.
Params
- investment_advisor: budget_egp*, horizon_years, risk_tolerance(low|medium|high), liquidity_need, unit_type, district_id, top_n, weights{return,risk,liquidity,services,construction,price_value},
  rental_yield_pct (one gross yield for all types; default by type), service_charge_pct (1.0), vacancy_pct (8), inflation_pct (13.9)  -> all optional scenario assumptions
- developer_performance: developer_id, stage(completed|under_construction), min_units, max_units, max_age_months, weights{sales,profitability,collection,delivery,inventory}
- unsold_inventory: project_id (omit for top risky), top_n, discount_rate_pct, response_assumptions{price_cut_pct, price_elasticity, extend_years, down_payment_cut_pts, uplift_per_dp_pt, limited_discount_pct}
See sample_*.json for real outputs.
""")

shutil.make_archive(OUT, "zip", OUT)
for root, _, files in os.walk(OUT):
    for f in files:
        print(os.path.relpath(os.path.join(root, f), OUT), os.path.getsize(os.path.join(root, f)))
