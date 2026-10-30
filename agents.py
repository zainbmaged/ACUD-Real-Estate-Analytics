"""
ACUD - Person 3: Investment & Developer AI Agents
Rule-based, explainable agents on top of the CSVs in ./data
(set ACUD_DATA_DIR to change the data folder).

Public API (same shape for every agent):
    run_agent("investment_advisor", budget_egp=10_000_000, horizon_years=5, risk_tolerance="medium",
              unit_type="Apartment", liquidity_need="medium", weights={...})
    run_agent("developer_performance", developer_id="DEV5", stage="completed", min_units=100)
    run_agent("unsold_inventory", project_id="P030")
    ask("قارن أفضل 3 مشروعات بميزانية 8 مليون لمدة 5 سنوات")      # free-text -> agent(s)
    portfolio_kpis(), district_kpis(), project_kpis()                # for the dashboard backend

Output shape:
    {agent, summary, results, evidence, assumptions, risks, missing_data, data_status}

Optional ML hook: put data/ML_Predictions.csv with columns
    Project_ID, Unsold_Risk_Probability  [, Model_Name, Evaluation_WAPE]
and the Unsold Inventory Agent will use it automatically.
"""
import contextvars
import os
import re
import numpy as np
import pandas as pd

DATA_DIR = os.environ.get("ACUD_DATA_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"))
_CACHE = {}

DATA_STATUS = "Synthetic demonstration data - not real ACUD market evidence"
def _missing_always():
    # Only items that genuinely cannot be derived or reasonably assumed stay here. Rent / inflation / exit depth moved to
    # the Investment Advisor's assumptions; planned & forecast completion dates are derived from the construction curves.
    return [T("تكاليف المطورين الحقيقية (نسبة التكلفة 68-78% افتراض سيناريو)", "Real developer costs (cost ratio 68-78% is a scenario assumption)"),
            T("أسعار الأراضي", "Land prices"),
            T("بيانات تشغيلية حقيقية (كل الجداول اصطناعية)", "Real operational data (all tables are synthetic)")]

# ----------------------------------------------------------------- scenario assumptions (all overridable)
SCENARIO_APPRECIATION_FACTOR = {"conservative": 0.35, "base": 0.6, "optimistic": 1.0}  # x trailing price-index growth
TRANSACTION_COST_PCT = 2.5            # exit cost (agent fee / registration) - assumption
# Return assumptions added to the Investment Advisor (all overridable from the UI / API). NOT measured - scenario assumptions.
# Gross rental yield (% of current value per year). Apartments anchored on Global Property Guide Q4 2025 (New Cairo 6.5-7.8%, Egypt avg 6.7%);
# other types are relative scenario assumptions (villas/townhouses yield less, commercial/office more).
RENTAL_YIELD_BY_TYPE_PCT = {"Apartment": 6.5, "Townhouse": 5.0, "Villa": 4.5, "Commercial": 8.5, "Office": 8.0}
DEFAULT_RENTAL_YIELD_PCT = 6.5        # used for a unit type not in the table
SERVICE_CHARGE_PCT = 1.0              # owner's annual service charges + maintenance, % of current value (paid after handover, even if vacant)
VACANCY_PCT = 8.0                     # share of rentable time the unit is empty
INFLATION_PCT = 13.9                  # annual urban inflation, CAPMAS Sep-2026 (latest reading) - held constant over the horizon
HANDOVER_LAG_MONTHS = 3               # fit-out / handover paperwork after construction completes, before rent starts
EXIT_WEIGHT_PROJECT = 0.6             # exit depth = 60% own-project overhang + 40% district overhang (months to absorb unsold units)
DISCOUNT_RATE_PCT = 18.0              # developer cost of capital - assumption
RESPONSE_ASSUMPTIONS = {              # how much sales velocity reacts to each lever - NOT measured, scenario assumptions
    "price_cut_pct": 5, "price_elasticity": 2.0,           # +2% velocity per 1% price cut
    "extend_years": 2, "uplift_per_extra_year": 0.06,      # +6% velocity per extra instalment year
    "down_payment_cut_pts": 5, "uplift_per_dp_pt": 0.016,  # +1.6% velocity per point of down payment removed
    "limited_discount_pct": 3, "limited_months": 3, "limited_uplift": 0.25,
}
DEFAULT_INV_WEIGHTS = {"return": 0.25, "risk": 0.25, "liquidity": 0.15, "services": 0.15,
                       "construction": 0.10, "price_value": 0.10}
DEV_WEIGHTS = {"sales": 0.25, "profitability": 0.25, "collection": 0.20, "delivery": 0.15, "inventory": 0.15}


# ----------------------------------------------------------------- language
_LANG = contextvars.ContextVar("agent_lang", default="ar")


def T(ar, en):
    """Return the Arabic or English text according to the language of the current agent call."""
    return en if _LANG.get() == "en" else ar


# ----------------------------------------------------------------- helpers
def _mm(s):
    """min-max to 0..1 (constant column -> 0.5 neutral... returns 0 for constant)."""
    s = pd.Series(s, dtype=float)
    rng = s.max() - s.min()
    return (s - s.min()) / rng if rng and not np.isnan(rng) else s * 0.0


def _clean(o):
    """Make output JSON-safe (numpy -> python, NaN -> None)."""
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple, set)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        return None if (np.isnan(o) or np.isinf(o)) else round(float(o), 3)
    if isinstance(o, pd.Timestamp):
        return o.strftime("%Y-%m-%d")
    return o


def _pack(agent, summary, results, evidence, assumptions, risks, extra_missing=None):
    return _clean({
        "agent": agent, "summary": summary, "results": results, "evidence": evidence,
        "assumptions": assumptions, "risks": risks,
        "missing_data": _missing_always() + (extra_missing or []), "data_status": DATA_STATUS,
    })


def _irr(flows):
    """Annual IRR by bisection (flows[0] at t=0). None if no sign change."""
    f = np.array(flows, dtype=float)
    if not (f.min() < 0 < f.max()):
        return None
    npv = lambda r: sum(cf / (1 + r) ** t for t, cf in enumerate(f))
    lo, hi = -0.9, 5.0
    if npv(lo) * npv(hi) > 0:
        return None
    for _ in range(80):
        mid = (lo + hi) / 2
        if npv(lo) * npv(mid) <= 0:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2


# ----------------------------------------------------------------- data
TABLES = ["Dim_Project", "Dim_Developer", "Dim_District", "Dim_Unit", "Dim_Price_Index",
          "Fact_Sales", "Fact_Inventory_Monthly", "Fact_Collections_Monthly",
          "Fact_Construction_Monthly", "Fact_District_Monthly_Status",
          "Fact_District_Service_Indicators", "Fact_Project_Sales_Funnel",
          "Project_Investment_Inputs", "Fact_Utilities_Monthly",
          "Fact_Tariffs_Monthly", "Fact_Residents_Monthly", "Fact_Solar_Monthly", "Sustainability_Solutions",
          "Solution_Deployments", "Fact_Solution_Pilot_Monthly"]


def _load():
    if _CACHE:
        return _CACHE
    d = {n: pd.read_csv(os.path.join(DATA_DIR, n + ".csv")) for n in TABLES}
    sv = os.path.join(DATA_DIR, "synthetic_survey_responses_ar.csv")
    d["survey"] = pd.read_csv(sv, encoding="utf-8-sig") if os.path.exists(sv) else pd.DataFrame()
    ml = os.path.join(DATA_DIR, "ML_Predictions.csv")
    d["ml"] = pd.read_csv(ml) if os.path.exists(ml) else None
    _CACHE.update(d)
    _CACHE["projects"] = _build_projects(d)
    _CACHE["avail_units"] = _build_available_units(d)
    return _CACHE


def _build_available_units(d):
    """Unsold units with an estimated current ticket price (Jan-2024 base x project's contract/base ratio)."""
    u, s, p = d["Dim_Unit"], d["Fact_Sales"], d["projects"] if "projects" in d else None
    sold_ids = set(s[s.Cancel_Date.isna()]["Unit_ID"])
    u = u.copy()
    u["base_price"] = u.Area_m2 * u.Price_per_m2_Jan2024
    ss = s[s.Cancel_Date.isna()].merge(u[["Unit_ID", "base_price"]], on="Unit_ID")
    ratio = (ss.Contract_Value_EGP / ss.base_price).groupby(ss.Project_ID).median()
    u["price_ratio"] = u.Project_ID.map(ratio).fillna(ratio.median())
    u["ticket"] = u.base_price * u.price_ratio
    u["is_sold"] = u.Unit_ID.isin(sold_ids)
    return u


def _build_projects(d):
    """One row per project with every metric the agents/dashboard need."""
    inv, fun, col, con = (d["Fact_Inventory_Monthly"], d["Fact_Project_Sales_Funnel"],
                          d["Fact_Collections_Monthly"], d["Fact_Construction_Monthly"])
    months = sorted(inv["Month"].unique())
    last, last6, prev6, last12 = months[-1], months[-6:], months[-12:-6], months[-12:]

    p = (d["Dim_Project"][["Project_ID", "Project_Name", "District_ID", "Developer_ID", "Units_Planned", "Launch_Date"]]
         .merge(d["Dim_Developer"][["Developer_ID", "Developer_Name", "Quality_Index"]], on="Developer_ID")
         .merge(d["Dim_District"][["District_ID", "District_Name", "Lifecycle_Archetype"]], on="District_ID")
         .merge(d["Project_Investment_Inputs"][["Project_ID", "sold_units_observed", "sell_through_pct",
                                                "observed_avg_contract_egp", "investment_cost_per_unit_egp"]],
                on="Project_ID"))
    # Source of truth for unit counts = Dim_Unit (matches Fact_Inventory_Monthly.Available_End). Units_Planned in Dim_Project
    # misses unplanned releases (e.g. P038 +300-unit phase, anomaly E_IV_001), so it understated unsold units.
    n_units = d["Dim_Unit"].groupby("Project_ID").size()
    p["units_planned_original"] = p.Units_Planned
    p["Units_Planned"] = p.Project_ID.map(n_units).fillna(p.Units_Planned).astype(int)
    p["units_unplanned_release"] = p.Units_Planned - p.units_planned_original
    p["sell_through_pct"] = p.sold_units_observed / p.Units_Planned * 100
    p["margin_pct"] = (p.observed_avg_contract_egp - p.investment_cost_per_unit_egp) / p.observed_avg_contract_egp * 100
    p["unsold_units"] = p.Units_Planned - p.sold_units_observed
    p["unsold_ratio"] = p.unsold_units / p.Units_Planned
    p["age_months"] = ((pd.Timestamp(last) - pd.to_datetime(p.Launch_Date)).dt.days / 30.4).round(1)
    ppm2 = d["Dim_Unit"].groupby("Project_ID")["Price_per_m2_Jan2024"].mean()
    p["price_per_m2"] = p.Project_ID.map(ppm2)

    # inventory
    cur = inv[inv.Month == last].set_index("Project_ID")
    p["available_units"] = p.Project_ID.map(cur["Available_End"])
    p["inventory_age_days"] = p.Project_ID.map(cur["Avg_Inventory_Age_Days"])
    s6 = inv[inv.Month.isin(last6)].groupby("Project_ID")["Units_Sold"].sum()
    s_prev = inv[inv.Month.isin(prev6)].groupby("Project_ID")["Units_Sold"].sum()
    p["sales_last6"] = p.Project_ID.map(s6).fillna(0)
    p["sales_prev6"] = p.Project_ID.map(s_prev).fillna(0)
    p["sales_momentum"] = (p.sales_last6 - p.sales_prev6) / p.sales_prev6.clip(lower=1)
    p["monthly_velocity"] = p.sales_last6 / 6
    p["months_of_inventory"] = (p.unsold_units / p.monthly_velocity.clip(lower=0.1)).clip(upper=120)
    # exit-market depth: how long the market needs to absorb the unsold overhang that competes with an investor's resale listing,
    # at the current sales pace - own project (60%) and whole district (40%). Lower = deeper, easier exit.
    dv = p.groupby("District_ID")[["unsold_units", "monthly_velocity"]].transform("sum")
    p["district_months_of_inventory"] = (dv.unsold_units / dv.monthly_velocity.clip(lower=0.1)).clip(upper=120)
    p["exit_depth_months"] = EXIT_WEIGHT_PROJECT * p.months_of_inventory + (1 - EXIT_WEIGHT_PROJECT) * p.district_months_of_inventory
    p["exit_depth_label"] = pd.cut(p.exit_depth_months, [-1, 12, 36, 10 ** 6], labels=["deep", "medium", "thin"]).astype(str)
    p["sales_value_12m"] = p.Project_ID.map(inv[inv.Month.isin(last12)].groupby("Project_ID")["Sales_Value_EGP"].sum())
    p["cancelled_12m"] = p.Project_ID.map(inv[inv.Month.isin(last12)].groupby("Project_ID")["Units_Cancelled"].sum())
    sold12 = inv[inv.Month.isin(last12)].groupby("Project_ID")["Units_Sold"].sum()
    p["cancel_rate"] = (p.cancelled_12m / (sold12.reindex(p.Project_ID).values + 1e-9)).clip(0, 1)

    # funnel (last 6 months)
    f = fun[fun.Month.isin(last6)].groupby("Project_ID").agg(
        leads=("Leads", "sum"), viewings=("Viewings", "sum"), bookings=("Bookings", "sum"),
        resp_hours=("Average_Lead_Response_Hours", "mean"), marketing=("Marketing_Spend_EGP", "sum"),
        l_lt5=("Budget_Segment_Leads_Under_5M", "sum"), l_5_10=("Budget_Segment_Leads_5M_to_10M", "sum"),
        l_gt10=("Budget_Segment_Leads_Over_10M", "sum"))
    p = p.merge(f, left_on="Project_ID", right_index=True, how="left")
    p["lead_to_booking"] = p.bookings / p.leads.clip(lower=1)

    # collections (last 12 months)
    c = col[col.Month.isin(last12)].groupby("Project_ID").agg(due=("Amount_Due_EGP", "sum"),
                                                                collected=("Amount_Collected_EGP", "sum"))
    p["collection_rate_raw"] = p.Project_ID.map(c.collected / c.due.replace(0, np.nan))
    p["collection_rate"] = p.collection_rate_raw.clip(upper=1.0)   # >100% = catch-up of overdue; capped so it cannot inflate scores
    p["overdue_latest_egp"] = p.Project_ID.map(col[col.Month == last].set_index("Project_ID")["Overdue_Balance_EGP"])

    # construction
    k = con[con.Month == last].set_index("Project_ID")
    p["completion_pct"] = p.Project_ID.map(k["Actual_Pct"])
    p["handover_pct"] = p.Project_ID.map(k["Handover_Pct"])
    p["delay_pts"] = (p.Project_ID.map(k["Planned_Pct"]) - p.completion_pct).clip(lower=0)
    p["is_completed"] = p.completion_pct >= 99.5
    # delivery timing derived from the construction curves (planning curve vs actual progress) - not contractual dates
    cmonths = sorted(con.Month.unique())
    prev_m = cmonths[-7] if len(cmonths) >= 7 else cmonths[0]
    k0 = con[con.Month == prev_m].set_index("Project_ID")
    rate_act = ((k["Actual_Pct"] - k0["Actual_Pct"]) / max(len(cmonths[cmonths.index(prev_m):]) - 1, 1)).clip(lower=0)
    p["construction_rate_pts_month"] = p.Project_ID.map(rate_act)
    plan_rate = d["Dim_Project"].set_index("Project_ID")["Planned_Rate_Pts_Month"]
    mtc_act = ((100 - p.completion_pct).clip(lower=0) / p.construction_rate_pts_month.clip(lower=0.5)).clip(upper=48)
    mtc_plan = ((100 - p.Project_ID.map(k["Planned_Pct"])).clip(lower=0) / p.Project_ID.map(plan_rate).clip(lower=0.5)).clip(upper=48)
    p["months_to_complete"] = mtc_act.where(~p.is_completed, 0.0)
    p["months_to_handover"] = p.months_to_complete + HANDOVER_LAG_MONTHS
    base_m = pd.Timestamp(last)
    p["forecast_completion_month"] = [(base_m + pd.DateOffset(months=int(round(x)))).strftime("%Y-%m") for x in p.months_to_complete]
    p["planned_completion_month"] = [(base_m + pd.DateOffset(months=int(round(x)))).strftime("%Y-%m") for x in mtc_plan.where(~p.is_completed, 0.0)]
    p["forecast_delay_months"] = (mtc_act - mtc_plan).clip(lower=0).where(~p.is_completed, 0.0)
    p.loc[p.is_completed, ["forecast_completion_month", "planned_completion_month"]] = None   # already completed: no future date
    cc = con.groupby("Project_ID")[["Planned_Cost_EGP", "Actual_Cost_EGP"]].sum()
    p["cost_overrun_pct"] = p.Project_ID.map((cc.Actual_Cost_EGP / cc.Planned_Cost_EGP.replace(0, np.nan) - 1) * 100)

    # district context (latest)
    ds = d["Fact_District_Monthly_Status"]
    ds = ds[ds.Month == ds.Month.max()].set_index("District_ID")
    sv = d["Fact_District_Service_Indicators"]
    sv = sv[sv.Month == sv.Month.max()].set_index("District_ID")
    p["service_gap_count"] = p.District_ID.map(ds["Service_Gap_Count"])
    p["service_coverage_pct"] = p.District_ID.map(ds["Essential_Service_Coverage_Pct"])
    p["occupancy_pct"] = p.District_ID.map(ds["Occupancy_Pct"])
    p["transport_score"] = p.District_ID.map(ds["Transport_Accessibility_Score"])
    p["utilities_reliability_pct"] = p.District_ID.map(sv["Utilities_Reliability_Pct"])

    # Unsold Inventory Risk Score 0-100 (weights documented in assumptions)
    p["risk_score"] = 100 * (0.30 * _mm(p.unsold_ratio) + 0.20 * _mm(p.inventory_age_days)
                             + 0.15 * _mm(1 - p.collection_rate.fillna(p.collection_rate.mean()))
                             + 0.15 * _mm(p.delay_pts) + 0.10 * _mm(p.cancel_rate)
                             + 0.10 * _mm(p.service_gap_count))
    return p


RISK_WEIGHTS_TXT = ("risk_score (0-100) = 30% unsold ratio + 20% inventory age + 15% collection shortfall "
                    "+ 15% construction delay + 10% cancellation rate + 10% district service gaps")


def _annual_price_growth():
    pi = _load()["Dim_Price_Index"].sort_values("Month")
    return float(pi.Price_Index.iloc[-1] / pi.Price_Index.iloc[-13] - 1) if len(pi) > 12 else 0.0


# ----------------------------------------------------------------- Agent 1: Investment Advisor
def _rent_flows(price, horizon, g, gross_yield_pct, service_pct, vacancy_pct, start_month):
    """Net rental income per year (index 1..horizon). Rent starts after handover; a unit is rentable only after `start_month`.
    Rent and service charges are a % of the unit's CURRENT value (mid-year price grown at g), so they follow the price scenario.
    Service charges / maintenance are owed from handover even when the unit is empty."""
    out = [0.0]
    for t in range(1, horizon + 1):
        active = max(0.0, min(12.0 * t, 12.0 * horizon) - max(12.0 * (t - 1), start_month))
        active = min(active, 12.0)
        value = price * (1 + g) ** (t - 0.5)
        gross = value * gross_yield_pct / 100 * active / 12
        out.append(gross * (1 - vacancy_pct / 100) - value * service_pct / 100 * active / 12)
    return out


def _cash_flows(price, down_pct, plan_years, horizon, g, exit_discount, tx_pct, delay_haircut, rent=None):
    """Annual investor cash flows: down payment t0, equal yearly instalments, net rent (optional), resale at horizon."""
    flows = [-price * down_pct]
    inst = price * (1 - down_pct) / max(plan_years, 1)
    for t in range(1, horizon + 1):
        flows.append(-inst if t <= plan_years else 0.0)
    if rent:
        flows = [f + r for f, r in zip(flows, rent)]
    resale = price * (1 + g) ** horizon * (1 - exit_discount - delay_haircut) * (1 - tx_pct / 100)
    remaining = price * (1 - down_pct) * max(plan_years - horizon, 0) / max(plan_years, 1)
    flows[-1] += resale - remaining
    net_profit = sum(flows)
    return flows, resale, net_profit


def investment_advisor(budget_egp, risk_tolerance="medium", top_n=3, district_id=None, horizon_years=5,
                       unit_type=None, liquidity_need="medium", weights=None, appreciation_pct=None,
                       rental_yield_pct=None, service_charge_pct=None, vacancy_pct=None, inflation_pct=None):
    """Compare projects by budget, horizon, risk, liquidity need and unit type, with 3 scenarios per project.
    Return = rent (after handover, net of vacancy and service charges) + resale; shown nominal and real (inflation-adjusted).
    rental_yield_pct (one gross yield for every unit type), service_charge_pct, vacancy_pct, inflation_pct override the defaults."""
    D = _load()
    clamp = lambda v, lo, hi, dflt: dflt if v is None or v == "" else float(min(max(float(v), lo), hi))
    svc_pct = clamp(service_charge_pct, 0, 10, SERVICE_CHARGE_PCT)
    vac_pct = clamp(vacancy_pct, 0, 100, VACANCY_PCT)
    infl_pct = clamp(inflation_pct, -5, 100, INFLATION_PCT)
    yield_override = None if rental_yield_pct in (None, "") else clamp(rental_yield_pct, 0, 30, DEFAULT_RENTAL_YIELD_PCT)
    p = D["projects"].copy()
    au = D["avail_units"]
    horizon_years = int(max(1, min(10, horizon_years or 5)))
    if district_id:
        p = p[p.District_ID == district_id]

    # purchasable units only: unsold, affordable, of the requested type
    av = au[(~au.is_sold) & (au.ticket <= budget_egp)]
    if unit_type:
        av = av[av.Unit_Type.str.lower() == str(unit_type).lower()]
    av = av[av.Project_ID.isin(p.Project_ID)]
    if av.empty:
        mn = au[~au.is_sold].ticket.min()
        return _pack("investment_advisor",
                     T(f"لا توجد وحدات متاحة للبيع ضمن ميزانية {budget_egp:,.0f} جنيه" + (f" ونوع {unit_type}" if unit_type else ""),
                       f"No units are available for sale within a budget of EGP {budget_egp:,.0f}" + (f" of type {unit_type}" if unit_type else "")),
                     {"top_projects": [], "district_comparison": [], "candidates_in_budget": 0}, [], [],
                     [T("الميزانية أقل من أقل سعر وحدة متاحة، أو النوع المطلوب غير متاح", "The budget is below the cheapest available unit, or the requested type is not available")],
                     [T(f"أرخص وحدة متاحة في البيانات: {mn:,.0f} جنيه", f"Cheapest available unit in data: {mn:,.0f} EGP")])
    g_obs = _annual_price_growth() if appreciation_pct is None else appreciation_pct / 100 / SCENARIO_APPRECIATION_FACTOR["optimistic"]
    av = av.assign(gross_yield=yield_override if yield_override is not None
                   else av.Unit_Type.map(RENTAL_YIELD_BY_TYPE_PCT).fillna(DEFAULT_RENTAL_YIELD_PCT))
    agg = av.groupby("Project_ID").agg(units_in_budget=("Unit_ID", "count"), ticket=("ticket", "median"),
                                       down_pct=("Down_Payment_Pct", "mean"), plan_years=("Plan_Years", "mean"),
                                       gross_rent_yield_pct=("gross_yield", "mean"),
                                       types=("Unit_Type", lambda s: ", ".join(sorted(set(s))))).reset_index()
    pool = p.merge(agg, on="Project_ID")
    pool["net_rent_yield_pct"] = pool.gross_rent_yield_pct * (1 - vac_pct / 100) - svc_pct   # steady-state, once handed over

    # scenarios
    scen = {k: [] for k in SCENARIO_APPRECIATION_FACTOR}
    exit_pcts = []
    for r in pool.itertuples():
        exit_disc = min(0.10, 0.02 + 0.0015 * r.exit_depth_months)   # market depth (own project + district overhang), not the project alone
        exit_pcts.append(exit_disc * 100)
        delay_hc = min(0.08, (r.delay_pts or 0) * 0.004)
        row = {}
        for name, fac in SCENARIO_APPRECIATION_FACTOR.items():
            hc = {"conservative": delay_hc, "base": delay_hc / 2, "optimistic": 0.0}[name]
            rent = _rent_flows(r.ticket, horizon_years, g_obs * fac, r.gross_rent_yield_pct, svc_pct, vac_pct, r.months_to_handover)
            fl, resale, prof = _cash_flows(r.ticket, r.down_pct, round(r.plan_years), horizon_years,
                                           g_obs * fac, exit_disc, TRANSACTION_COST_PCT, hc, rent=rent)
            irr = _irr(fl)
            irr_real = ((1 + irr) / (1 + infl_pct / 100) - 1) if irr is not None else None
            row[name] = {"net_profit_egp": prof, "irr_pct": irr * 100 if irr is not None else np.nan,
                         "irr_real_pct": irr_real * 100 if irr_real is not None else np.nan,
                         "net_profit_real_egp": sum(cf / (1 + infl_pct / 100) ** t for t, cf in enumerate(fl)),
                         "rent_income_egp": sum(rent), "resale_value_egp": resale, "net_cash_flows": [round(x) for x in fl]}
        for k in scen:
            scen[k].append(row[k])
    pool["exit_discount_pct"] = exit_pcts
    pool["scenarios"] = [{k: scen[k][i] for k in scen} for i in range(len(pool))]
    pool["irr_base_pct"] = [s["base"]["irr_pct"] for s in pool.scenarios]
    pool["irr_cons_pct"] = [s["conservative"]["irr_pct"] for s in pool.scenarios]
    pool["net_profit_base_egp"] = [s["base"]["net_profit_egp"] for s in pool.scenarios]
    pool["irr_real_base_pct"] = [s["base"]["irr_real_pct"] for s in pool.scenarios]
    pool["net_profit_real_base_egp"] = [s["base"]["net_profit_real_egp"] for s in pool.scenarios]
    pool["rent_income_base_egp"] = [s["base"]["rent_income_egp"] for s in pool.scenarios]
    pool["price_per_m2_ref"] = pool.price_per_m2
    peer_ppm2 = D["projects"].groupby("District_ID")["price_per_m2"].mean()
    pool["price_vs_district_pct"] = (pool.price_per_m2 / pool.District_ID.map(peer_ppm2) - 1) * 100

    # Investment Attractiveness Score: components 0..1 (NaN -> component missing -> score flagged incomplete)
    comp = pd.DataFrame({
        "return": _mm(pool.irr_base_pct.where(pool.irr_base_pct.notna())),
        "risk": 1 - pool.risk_score / 100,
        "liquidity": 1 - _mm(pool.months_of_inventory),
        "services": _mm(pool.service_coverage_pct),
        "construction": 0.5 * _mm(pool.completion_pct) + 0.5 * (1 - _mm(pool.delay_pts)),
        "price_value": 1 - _mm(pool.price_vs_district_pct),
    }, index=pool.index)
    comp.loc[pool.irr_base_pct.isna(), "return"] = np.nan
    w = dict(DEFAULT_INV_WEIGHTS)
    if weights:
        w.update({k: float(v) for k, v in weights.items() if k in w})
    else:  # profile-based tilt (only when the user did not give explicit weights)
        if risk_tolerance == "low":
            w["risk"] *= 1.6; w["return"] *= 0.8
        elif risk_tolerance == "high":
            w["return"] *= 1.5; w["risk"] *= 0.5
        if liquidity_need == "high":
            w["liquidity"] *= 2.0
        elif liquidity_need == "low":
            w["liquidity"] *= 0.5
    tot = sum(w.values())
    w = {k: v / tot for k, v in w.items()}
    W = pd.Series(w)
    avail = comp.notna()
    pool["attractiveness_score"] = 100 * (comp.fillna(0) * W).sum(axis=1) / (avail * W).sum(axis=1)
    pool["score_complete"] = avail.all(axis=1)
    pool["missing_components"] = [", ".join(c for c in comp.columns if not avail.loc[i, c]) for i in pool.index]
    pool["components"] = [{k: round(100 * float(v), 1) if pd.notna(v) else None for k, v in comp.loc[i].items()} for i in pool.index]

    top = pool.sort_values("attractiveness_score", ascending=False).head(top_n)
    ar = {"return": T("العائد المتوقع", "Expected return"), "risk": T("انخفاض المخاطر", "Low risk"),
          "liquidity": T("السيولة (سهولة التخارج)", "Liquidity (ease of exit)"), "services": T("جودة الخدمات", "Service quality"),
          "construction": T("التقدم الإنشائي", "Construction progress"), "price_value": T("سعر المتر مقابل الحي", "Price per m² vs district")}
    med = comp.median()
    sw = []
    for i in top.index:
        st = [ar[c] for c in comp.columns if pd.notna(comp.loc[i, c]) and comp.loc[i, c] >= max(0.7, med[c] + 0.15)]
        wk = [ar[c] for c in comp.columns if pd.notna(comp.loc[i, c]) and comp.loc[i, c] <= min(0.3, med[c] - 0.15)]
        sw.append({"strengths": st, "weaknesses": wk})
    top = top.assign(strengths=[x["strengths"] for x in sw], weaknesses=[x["weaknesses"] for x in sw])

    cols = ["Project_ID", "Project_Name", "District_ID", "District_Name", "Developer_Name", "types", "units_in_budget",
            "ticket", "price_per_m2_ref", "price_vs_district_pct", "down_pct", "plan_years", "irr_base_pct", "irr_cons_pct",
            "irr_real_base_pct", "net_profit_base_egp", "net_profit_real_base_egp", "rent_income_base_egp",
            "gross_rent_yield_pct", "net_rent_yield_pct", "months_to_handover", "planned_completion_month",
            "forecast_completion_month", "forecast_delay_months", "exit_depth_months", "exit_depth_label", "exit_discount_pct",
            "months_of_inventory", "completion_pct", "delay_pts", "service_coverage_pct",
            "risk_score", "attractiveness_score", "score_complete", "missing_components",
            "components", "strengths", "weaknesses", "scenarios"]
    districts = (pool.groupby(["District_ID", "District_Name", "Lifecycle_Archetype"])
                 .agg(projects=("Project_ID", "count"), avg_irr_base_pct=("irr_base_pct", "mean"),
                      avg_risk=("risk_score", "mean"), avg_score=("attractiveness_score", "mean"),
                      service_coverage_pct=("service_coverage_pct", "first"))
                 .reset_index().sort_values("avg_score", ascending=False))
    evidence = []
    for r in top.itertuples():
        b = r.scenarios["base"]; c = r.scenarios["conservative"]; o = r.scenarios["optimistic"]
        sep = T("، ", ", ")
        evidence.append({"project": r.Project_ID, "why": T(
            f"{r.Project_ID} ({r.District_ID}): وحدة بسعر ~{r.ticket:,.0f} ({r.units_in_budget} وحدة متاحة ضمن الميزانية)، "
            f"درجة الجاذبية {r.attractiveness_score:.0f}/100، مخاطرة {r.risk_score:.0f}/100، تقدم إنشائي {r.completion_pct:.0f}% (تأخر {r.delay_pts:.1f} نقطة)، "
            f"أشهر المخزون {r.months_of_inventory:.0f}. عائد سنوي (IRR) متحفظ/أساسي/متفائل: "
            f"{_f(c['irr_pct'])}% / {_f(b['irr_pct'])}% / {_f(o['irr_pct'])}%، صافي الربح الأساسي بعد {horizon_years} سنوات {b['net_profit_egp']:,.0f} جنيه "
            f"(منها إيجار صافٍ {b['rent_income_egp']:,.0f})؛ العائد الحقيقي بعد التضخم {_f(b['irr_real_pct'])}%. "
            f"الإيجار يبدأ بعد ~{r.months_to_handover:.0f} شهرًا من الشراء (عند التسليم)، وعائده الإيجاري الصافي المستقر {r.net_rent_yield_pct:.1f}%. "
            f"عمق سوق التخارج: ~{r.exit_depth_months:.0f} شهرًا لتصريف المعروض ({_depth(r.exit_depth_label)}) وخصم تخارج {r.exit_discount_pct:.1f}%. "
            + (f"نقاط القوة: {sep.join(r.strengths)}. " if r.strengths else "") + (f"نقاط الضعف: {sep.join(r.weaknesses)}." if r.weaknesses else ""),
            f"{r.Project_ID} ({r.District_ID}): a unit priced ~EGP {r.ticket:,.0f} ({r.units_in_budget} units available within budget). "
            f"Attractiveness {r.attractiveness_score:.0f}/100, risk {r.risk_score:.0f}/100, construction {r.completion_pct:.0f}% ({r.delay_pts:.1f} pts behind plan), "
            f"{r.months_of_inventory:.0f} months of inventory. Annual IRR conservative/base/optimistic: "
            f"{_f(c['irr_pct'])}% / {_f(b['irr_pct'])}% / {_f(o['irr_pct'])}%; base net profit after {horizon_years} years EGP {b['net_profit_egp']:,.0f} "
            f"(of which net rent {b['rent_income_egp']:,.0f}); real IRR after inflation {_f(b['irr_real_pct'])}%. "
            f"Rent starts ~{r.months_to_handover:.0f} months after purchase (handover), steady-state net rental yield {r.net_rent_yield_pct:.1f}%. "
            f"Exit-market depth: ~{r.exit_depth_months:.0f} months to absorb the overhang ({_depth(r.exit_depth_label)}), exit discount {r.exit_discount_pct:.1f}%. "
            + (f"Strengths: {sep.join(r.strengths)}. " if r.strengths else "") + (f"Weaknesses: {sep.join(r.weaknesses)}." if r.weaknesses else ""))})
    return _pack(
        "investment_advisor",
        T(f"أفضل {len(top)} مشروعات لميزانية {budget_egp:,.0f} جنيه، أفق {horizon_years} سنوات، مخاطرة {risk_tolerance}"
          + (f"، نوع {unit_type}" if unit_type else ""),
          f"Top {len(top)} projects for a budget of EGP {budget_egp:,.0f}, {horizon_years}-year horizon, {risk_tolerance} risk"
          + (f", type {unit_type}" if unit_type else "")),
        {"top_projects": top[cols].to_dict("records"), "district_comparison": districts.to_dict("records"),
         "candidates_in_budget": int(len(pool)), "weights_used": w,
         "scenario_growth_pct_per_year": {k: round(g_obs * f * 100, 1) for k, f in SCENARIO_APPRECIATION_FACTOR.items()},
         "return_assumptions": {"gross_rental_yield_pct_by_type": ({t: yield_override for t in RENTAL_YIELD_BY_TYPE_PCT} if yield_override is not None
                                                                   else RENTAL_YIELD_BY_TYPE_PCT),
                                "service_charge_pct": svc_pct, "vacancy_pct": vac_pct, "inflation_pct": infl_pct,
                                "handover_lag_months": HANDOVER_LAG_MONTHS, "exit_weight_project": EXIT_WEIGHT_PROJECT,
                                "transaction_cost_pct": TRANSACTION_COST_PCT}},
        evidence,
        [T("الميزانية = سعر الوحدة الواحدة؛ تُعرض الوحدات المتاحة للبيع فقط (المشروعات المباعة بالكامل تُستبعد)",
           "Budget = price of a single unit; only units actually available for sale are shown (fully sold projects are excluded)"),
         T("سعر الوحدة = سعر الأساس (يناير 2024) × نسبة العقود الفعلية للمشروع، وليس سعرًا معروضًا حاليًا",
           "Unit price = Jan-2024 base price × the project's actual contract ratio, not a currently listed price"),
         T(f"النمو السنوي للأسعار من مؤشر الأسعار آخر 12 شهرًا = {g_obs * 100:.1f}%؛ السيناريوهات تأخذ ", f"Annual price growth from the price index over the last 12 months = {g_obs * 100:.1f}%; scenarios take ")
         + " / ".join(f"{k} {int(f * 100)}%" for k, f in SCENARIO_APPRECIATION_FACTOR.items()) + T(" منه (افتراضات سيناريو)", " of it (scenario assumptions)"),
         T(f"سداد المستثمر: مقدم + أقساط سنوية متساوية حسب خطة وحدات المشروع؛ التخارج بإعادة بيع في نهاية الأفق، تكلفة معاملة {TRANSACTION_COST_PCT}%، "
           "خصم تخارج حتى 10% حسب عمق سوق التخارج، وخصم تأخير من نقاط التأخر الإنشائي",
           f"Investor pays a down payment + equal annual instalments per the project's plan; exit by resale at the end of the horizon, transaction cost {TRANSACTION_COST_PCT}%, "
           "exit discount up to 10% by exit-market depth, and a delay discount from construction delay points"),
         T(f"الإيجار (افتراض سيناريو وليس قياسًا): عائد إيجاري إجمالي سنوي من قيمة الوحدة الحالية — "
           + "، ".join(f"{k} {v:.1f}%" for k, v in ({t: yield_override for t in RENTAL_YIELD_BY_TYPE_PCT} if yield_override is not None else RENTAL_YIELD_BY_TYPE_PCT).items())
           + f"؛ شاغر {vac_pct:.0f}%، ورسوم خدمات وصيانة {svc_pct:.1f}% سنويًا من القيمة. الإيجار يبدأ بعد اكتمال الإنشاء + {HANDOVER_LAG_MONTHS} أشهر تسليم "
             "(من معدل التقدم الإنشائي آخر 6 أشهر)، والإيجار والرسوم يتبعان سعر السيناريو. مرجع الشقق: Global Property Guide الربع الرابع 2025 (القاهرة الجديدة 6.5-7.8%)",
           f"Rent (scenario assumption, not measured): annual gross rental yield on the unit's current value — "
           + ", ".join(f"{k} {v:.1f}%" for k, v in ({t: yield_override for t in RENTAL_YIELD_BY_TYPE_PCT} if yield_override is not None else RENTAL_YIELD_BY_TYPE_PCT).items())
           + f"; vacancy {vac_pct:.0f}%, service charges & maintenance {svc_pct:.1f}% of value a year. Rent starts after construction completes + {HANDOVER_LAG_MONTHS} months of handover "
             "(from the last-6-month construction pace); rent and charges follow the scenario price. Apartment benchmark: Global Property Guide Q4 2025 (New Cairo 6.5-7.8%)"),
         T(f"التضخم {infl_pct:.1f}% سنويًا ثابت على كل الأفق (آخر قراءة للتضخم الحضري: 13.9% سبتمبر 2026، الجهاز المركزي للتعبئة والإحصاء)؛ العائد الحقيقي = (1+IRR الاسمي)÷(1+التضخم)−1 وصافي الربح الحقيقي بخصم التدفقات بالتضخم",
           f"Inflation {infl_pct:.1f}% a year, held constant over the horizon (latest urban reading: 13.9%, Sep 2026, CAPMAS); real IRR = (1 + nominal IRR) ÷ (1 + inflation) − 1, real profit deflates each cash flow by inflation"),
         T(f"عمق سوق التخارج = {EXIT_WEIGHT_PROJECT * 100:.0f}% أشهر تصريف مخزون المشروع + {(1 - EXIT_WEIGHT_PROJECT) * 100:.0f}% أشهر تصريف مخزون الحي (غير المباع ÷ سرعة البيع آخر 6 أشهر)؛ ≤12 شهرًا عميق، ≤36 متوسط، أكثر من ذلك ضحل",
           f"Exit-market depth = {EXIT_WEIGHT_PROJECT * 100:.0f}% own-project months of inventory + {(1 - EXIT_WEIGHT_PROJECT) * 100:.0f}% district months of inventory (unsold ÷ sales pace of the last 6 months); ≤12 months deep, ≤36 medium, above that thin"),
         T("الأوزان (قابلة للتعديل عبر weights): ", "Weights (adjustable via weights): ") + T("، ", ", ").join(f"{ar[k]} {v * 100:.0f}%" for k, v in w.items()),
         T("كل البيانات اصطناعية؛ الإيجار والتضخم وعمق التخارج افتراضات/مشتقات معلنة وقابلة للتعديل وليست قياسات سوق",
           "All data is synthetic; rent, inflation and exit depth are declared, adjustable assumptions / derived estimates, not market measurements")],
        [T("العائد يعتمد على استمرار نمو الأسعار التاريخي؛ تراجع السوق غير مُمثَّل إلا في السيناريو المتحفظ",
           "Return depends on historical price growth continuing; a market decline is only represented in the conservative scenario"),
         T("المشروعات ذات المخزون القديم أو التحصيل الضعيف قد تحتاج وقتًا أطول للتخارج",
           "Projects with aged inventory or weak collections may take longer to exit"),
         T("التأخر الإنشائي قد يؤخر التسليم وإعادة البيع", "Construction delays can postpone handover and resale"),
         T("الإيجار جزء كبير من العائد وهو افتراض؛ فرق نقطة واحدة في العائد الإيجاري يغيّر الـ IRR، فاختبر القيم في خانة الافتراضات",
           "Rent is a large part of the return and is an assumption; one point of rental yield moves the IRR, so test other values in the assumptions panel"),
         T("التضخم ثابت على الأفق كله؛ لو انخفض فالعائد الحقيقي أعلى من المعروض ولو ارتفع فهو أقل",
           "Inflation is held constant over the horizon; if it falls the real return is higher than shown, if it rises it is lower")],
        [] + ([T(f"الدرجة غير مكتملة لـ {int((~pool.score_complete).sum())} مشروع: مكونات غير متاحة", f"Score incomplete for {int((~pool.score_complete).sum())} project(s): components not available")]
           if (~pool.score_complete).any() else []))


def _depth(label):
    return T({"deep": "عميق", "medium": "متوسط", "thin": "ضحل"}.get(label, label), label)


def _f(v):
    return T("غير متاح", "n/a") if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.1f}"


# ----------------------------------------------------------------- Agent 2: Developer Performance
def _dev_trends(d, dev_projects):
    """Last 6 months vs previous 6 months per metric for a set of projects."""
    inv, col, con = d["Fact_Inventory_Monthly"], d["Fact_Collections_Monthly"], d["Fact_Construction_Monthly"]
    months = sorted(inv.Month.unique())
    wins = {"previous": months[-12:-6], "recent": months[-6:]}
    out = {}
    for w, ms in wins.items():
        i = inv[inv.Project_ID.isin(dev_projects) & inv.Month.isin(ms)]
        c = col[col.Project_ID.isin(dev_projects) & col.Month.isin(ms)]
        k = con[con.Project_ID.isin(dev_projects) & (con.Month == ms[-1])]
        out[w] = {"units_sold": i.Units_Sold.sum(), "sales_value_egp": i.Sales_Value_EGP.sum(),
                  "collection_rate_pct": c.Amount_Collected_EGP.sum() / max(c.Amount_Due_EGP.sum(), 1) * 100,
                  "cancel_rate_pct": i.Units_Cancelled.sum() / max(i.Units_Sold.sum(), 1) * 100,
                  "avg_inventory_age_days": i[i.Month == ms[-1]].Avg_Inventory_Age_Days.mean(),
                  "construction_delay_pts": (k.Planned_Pct - k.Actual_Pct).clip(lower=0).mean()}
    better_high = {"units_sold": True, "sales_value_egp": True, "collection_rate_pct": True,
                   "cancel_rate_pct": False, "avg_inventory_age_days": False, "construction_delay_pts": False}
    rows = []
    for m, hi in better_high.items():
        a, b = out["previous"][m], out["recent"][m]
        if pd.isna(a) or pd.isna(b):
            continue
        rel = (b - a) / abs(a) if a else 0
        small = abs(b - a) < (1.0 if m.endswith("pct") or m.endswith("pts") else 0) or abs(rel) < 0.05
        status = "stable" if small else ("improved" if (b > a) == hi else "declined")
        rows.append({"metric": m, "previous_6m": a, "recent_6m": b, "change": b - a, "status": status})
    return rows


def developer_performance(developer_id=None, stage=None, min_units=None, max_units=None, max_age_months=None, weights=None):
    """Rank developers fairly: vs same-district peers, adjusted for project age, with sub-scores and trends."""
    D = _load()
    allp = D["projects"].copy()
    # fairness: compare each project with peers of its own district, then remove the effect of project age
    allp["st_vs_district"] = allp.sell_through_pct - allp.groupby("District_ID")["sell_through_pct"].transform("mean")
    allp["age_vs_district"] = allp.age_months - allp.groupby("District_ID")["age_months"].transform("mean")
    slope, icpt = np.polyfit(allp.age_vs_district, allp.st_vs_district, 1)
    allp["sales_vs_expected"] = allp.st_vs_district - (slope * allp.age_vs_district + icpt)
    allp["delay_vs_district"] = allp.delay_pts - allp.groupby("District_ID")["delay_pts"].transform("mean")
    allp["on_time"] = (allp.delay_pts <= 2).astype(float)
    allp["size_class"] = pd.cut(allp.Units_Planned, [0, 150, 250, 10 ** 9], labels=["small", "medium", "large"]).astype(str)
    p = allp
    if stage == "completed":
        p = p[p.is_completed]
    elif stage in ("under_construction", "ongoing"):
        p = p[~p.is_completed]
    if min_units:
        p = p[p.Units_Planned >= min_units]
    if max_units:
        p = p[p.Units_Planned <= max_units]
    if max_age_months:
        p = p[p.age_months <= max_age_months]
    filt = {k: v for k, v in dict(stage=stage, min_units=min_units, max_units=max_units, max_age_months=max_age_months).items() if v}

    base_assump = [T("المقارنة العادلة: نسبة البيع تُقاس مقابل متوسط نفس الحي ثم يُخصم أثر عمر المشروع (انحدار خطي)، والتأخير مقابل متوسط الحي",
                     "Fair comparison: sell-through is measured against the same district's average, then project-age effect is removed (linear regression); delay is measured against the district average"),
                   T("الدرجة = 25% مبيعات نسبية + 25% ربحية + 20% تحصيل + 15% تسليم/التزام + 15% مخزون (أوزان مبدئية غير مثبتة علميًا؛ كل مكوّن 0-100 باتجاه موحد)",
                     "Score = 25% relative sales + 25% profitability + 20% collections + 15% delivery/commitment + 15% inventory (initial weights, not scientifically validated; each component 0-100, same direction)"),
                   T("الربحية = هامش مفترض (70%) + انحراف التكلفة (30%)؛ التسليم = التأخر الإنشائي مقابل الحي + نسبة المشروعات المتأخرة ≤ 2 نقطة (بديل لـ On-time delivery)",
                     "Profitability = assumed margin (70%) + cost overrun (30%); delivery = construction delay vs district + share of projects delayed ≤ 2 pts (proxy for on-time delivery)"),
                   T("المخزون = عمر المخزون + نسبة غير المباع؛ Quality_Index للمرجعية فقط ولا يدخل في الدرجة",
                     "Inventory = inventory age + unsold ratio; Quality_Index is for reference only and is not part of the score"),
                   T("التطوّر عبر الزمن = آخر 6 أشهر مقابل الـ 6 قبلها (تغير أقل من 5% = مستقر)",
                     "Trend over time = last 6 months vs the previous 6 (change under 5% = stable)")]
    risks = [T("عدد مشروعات كل مطور صغير، فالترتيب حساس لمشروع واحد ضعيف",
               "Each developer has few projects, so the ranking is sensitive to one weak project"),
             T("المقارنة بين مطورين في مشروعات مختلفة المرحلة محدودة الدلالة؛ استخدم المرشحات للمقارنة المتجانسة",
               "Comparing developers whose projects are at different stages has limited meaning; use the filters for a like-for-like comparison"),
             T("كل البيانات اصطناعية", "All data is synthetic")]
    extra = [T("مواعيد التسليم التعاقدية من العقود (المعروض: موعد الإنجاز المخطط من منحنى الخطة وتوقع الإنجاز من معدل التقدم الفعلي آخر 6 أشهر؛ التسليم في الوقت يُقاس ببديل نقاط التأخر)",
               "Contractual delivery dates from contracts (shown instead: planned completion from the planning curve and forecast completion from the last-6-month actual pace; on-time delivery still uses a delay-points proxy)"),
             T("مستهدفات المبيعات (نسبة تحقيق المستهدف غير قابلة للحساب)", "Sales targets (Target Achievement % not computable)")]
    if p.empty:
        return _pack("developer_performance", T("لا توجد مشروعات تطابق المرشحات", "No projects match the filters"), {"ranking": [], "filters": filt}, [], base_assump, risks, extra)

    g = p.groupby(["Developer_ID", "Developer_Name"]).agg(
        projects=("Project_ID", "count"), units_planned=("Units_Planned", "sum"), units_sold=("sold_units_observed", "sum"),
        unsold_units=("unsold_units", "sum"), sales_vs_expected=("sales_vs_expected", "mean"),
        sell_through_vs_district=("st_vs_district", "mean"), delay_vs_district=("delay_vs_district", "mean"),
        on_time_share=("on_time", "mean"), collection_rate=("collection_rate", "mean"), cancel_rate=("cancel_rate", "mean"),
        avg_margin_pct=("margin_pct", "mean"), cost_overrun_pct=("cost_overrun_pct", "mean"),
        avg_inventory_age=("inventory_age_days", "mean"), unsold_ratio=("unsold_ratio", "mean"),
        avg_risk=("risk_score", "mean"), avg_project_age_months=("age_months", "mean"),
        quality_index_reference=("Quality_Index", "first")).reset_index()
    g["sell_through_pct"] = g.units_sold / g.units_planned * 100
    ov = g.cost_overrun_pct
    prof = 0.7 * _mm(g.avg_margin_pct) + 0.3 * (1 - _mm(ov)).where(ov.notna(), 0.7 * _mm(g.avg_margin_pct) / 0.3 * 0.3)
    sub = pd.DataFrame({
        "sales": 100 * _mm(g.sales_vs_expected),
        "profitability": 100 * (0.7 * _mm(g.avg_margin_pct) + 0.3 * (1 - _mm(ov)).fillna(_mm(g.avg_margin_pct))),
        "collection": 100 * _mm(g.collection_rate),
        "delivery": 100 * (0.5 * (1 - _mm(g.delay_vs_district)) + 0.5 * _mm(g.on_time_share)),
        "inventory": 100 * (0.5 * (1 - _mm(g.avg_inventory_age)) + 0.5 * (1 - _mm(g.unsold_ratio))),
    })
    w = dict(DEV_WEIGHTS)
    if weights:
        w.update({k: float(v) for k, v in weights.items() if k in w})
        t = sum(w.values()); w = {k: v / t for k, v in w.items()}
    g["score"] = sum(sub[k] * v for k, v in w.items())
    for k in sub.columns:
        g["sub_" + k] = sub[k]
    g["weakest_component"] = sub.idxmin(axis=1)
    g["red_flags"] = [[f"{c}: {sub.loc[i, c]:.0f}/100" for c in sub.columns if sub.loc[i, c] < 30] for i in g.index]
    g = g.sort_values("score", ascending=False).reset_index(drop=True)
    g["rank"] = g.index + 1
    g["sales_volume_rank"] = g.units_sold.rank(ascending=False, method="min").astype(int)

    if developer_id:
        row = g[g.Developer_ID == developer_id]
        if row.empty:
            return _pack("developer_performance", T(f"المطور {developer_id} غير موجود (أو لا مشروعات تطابق المرشحات)", f"Developer {developer_id} not found (or no projects match the filters)"),
                         {"ranking": [], "filters": filt}, [], base_assump, risks, extra)
        r = row.iloc[0]
        proj = p[p.Developer_ID == developer_id].sort_values("risk_score", ascending=False)
        pcols = ["Project_ID", "District_ID", "Units_Planned", "age_months", "size_class", "completion_pct", "sell_through_pct",
                 "st_vs_district", "sales_vs_expected", "delay_pts", "planned_completion_month", "forecast_completion_month",
                 "forecast_delay_months", "collection_rate", "margin_pct", "unsold_units", "risk_score"]
        trends = _dev_trends(D, list(proj.Project_ID))
        imp = [t["metric"] for t in trends if t["status"] == "improved"]
        dec = [t["metric"] for t in trends if t["status"] == "declined"]
        subs = {c: r["sub_" + c] for c in sub.columns}
        strengths = [c for c, v in subs.items() if v >= 70]
        issues = [c for c, v in subs.items() if v < 40]
        none_ = T("لا شيء", "none")
        evidence = [T(f"الترتيب {int(r['rank'])} من {len(g)} بدرجة {r.score:.0f}/100 (ترتيبه حسب حجم البيع {int(r.sales_volume_rank)} - لا يُعتمد عليه وحده)",
                      f"Rank {int(r['rank'])} of {len(g)} with a score of {r.score:.0f}/100 (rank by sales volume {int(r.sales_volume_rank)} - not to be relied on alone)"),
                    T("الدرجات الفرعية: ", "Sub-scores: ") + T("، ", ", ").join(f"{c} {v:.0f}" for c, v in subs.items()),
                    T(f"نسبة بيع {r.sell_through_pct:.0f}% ({r.sell_through_vs_district:+.1f} نقطة عن أحيائه، {r.sales_vs_expected:+.1f} بعد تعديل العمر)",
                      f"Sell-through {r.sell_through_pct:.0f}% ({r.sell_through_vs_district:+.1f} pts vs its districts, {r.sales_vs_expected:+.1f} after age adjustment)"),
                    T(f"تحصيل {r.collection_rate * 100:.0f}%، إلغاءات {r.cancel_rate * 100:.1f}%، أضعف مشروع {proj.iloc[0].Project_ID} (مخاطرة {proj.iloc[0].risk_score:.0f})",
                      f"Collections {r.collection_rate * 100:.0f}%, cancellations {r.cancel_rate * 100:.1f}%, weakest project {proj.iloc[0].Project_ID} (risk {proj.iloc[0].risk_score:.0f})"),
                    T("تحسّن: ", "Improved: ") + (", ".join(imp) or none_) + T(" | تراجع: ", " | Declined: ") + (", ".join(dec) or none_)]
        res = {"developer": r.to_dict(), "projects": proj[pcols].to_dict("records"), "subscores": subs, "trends": trends,
               "strengths": strengths, "issues": issues, "filters": filt, "weights_used": w,
               "comparison_limits": [T("مقارنة داخل نفس الحي فقط", "Compared within the same district only"), T(f"عدد المشروعات {int(r.projects)} فقط", f"Only {int(r.projects)} project(s)"), T("نضج الحي (Lifecycle) يؤثر في معدل البيع", "District maturity (lifecycle) affects the sales rate")]}
        return _pack("developer_performance", T(f"{r.Developer_Name}: المركز {int(r['rank'])} من {len(g)} (درجة {r.score:.0f})", f"{r.Developer_Name}: rank {int(r['rank'])} of {len(g)} (score {r.score:.0f})"),
                     res, evidence, base_assump, risks, extra)
    top = g.iloc[0]
    flagged = [f"{r.Developer_Name}: {', '.join(r.red_flags)}" for r in g.itertuples() if r.red_flags]
    return _pack("developer_performance", T(f"ترتيب {len(g)} مطورين (أفضلهم {top.Developer_Name})", f"Ranking of {len(g)} developers (best: {top.Developer_Name})"),
                 {"ranking": g.to_dict("records"), "filters": filt, "weights_used": w},
                 [T(f"{r.Developer_Name}: درجة {r.score:.0f}، بيع نسبي {r.sell_through_vs_district:+.1f}، ترتيب حجم البيع {r.sales_volume_rank}",
                    f"{r.Developer_Name}: score {r.score:.0f}, relative sell-through {r.sell_through_vs_district:+.1f}, sales-volume rank {r.sales_volume_rank}")
                  for r in g.head(3).itertuples()] + ([T("تنبيهات مكوّنات ضعيفة: ", "Weak-component alerts: ") + " | ".join(flagged)] if flagged else []),
                 base_assump, risks, extra)


# ----------------------------------------------------------------- Agent 3: Unsold Inventory
def _pv_price(P, d, n, r):
    n = max(int(round(n)), 1)
    return P * (d + (1 - d) / n * sum(1 / (1 + r) ** k for k in range(1, n + 1)))


def _econ(N, P, c, d, n, v, r, mkt_month, limited=None):
    """Value of clearing N unsold units: PV of cash (price, plan), time discount, minus marketing run-rate."""
    v = max(v, 0.05)
    if limited:  # temporary discount window then back to baseline speed
        x, months, v1, v0 = limited
        in_window = min(N, v1 * months)
        rest = N - in_window
        t = months * (in_window / max(v1 * months, 1e-9)) + (rest / max(v0, 0.05) if rest > 0 else 0)
        avg_price = P * (1 - x * in_window / N)
    else:
        t, avg_price = N / v, P
    t = min(t, 120)
    pv = _pv_price(avg_price, d, n, r)
    tf = 1 / (1 + r) ** (t / 24)
    value = N * (pv - c) * tf - mkt_month * t
    return {"months_to_clear": t, "net_value_egp": value, "margin_pct": (avg_price - c) / avg_price * 100,
            "avg_price_egp": avg_price}


SURVEY_EN = {"الترفيه": "Entertainment", "التسوق": "Shopping", "المواصلات": "Transportation", "النظافة": "Cleanliness", "الأمن": "Security",
             "المياه أو الكهرباء": "Water or electricity", "الخدمات الطبية": "Medical services", "لا توجد مشكلة مؤثرة": "No impactful problem",
             "التعليم": "Education", "الطرق أو الإنارة": "Roads or lighting", "الإنترنت": "Internet",
             "الترفيه والمساحات الخضراء": "Entertainment & green spaces", "التسوق والاحتياجات اليومية": "Shopping & daily needs",
             "المدارس أو التعليم": "Schools or education", "الخدمات الطبية أو الصيدليات": "Medical services or pharmacies",
             "النظافة وجمع المخلفات": "Cleanliness & waste collection", "الإنترنت والاتصالات": "Internet & telecom", "الأمن والسلامة": "Security & safety",
             "السعر": "Price", "موقع المنطقة": "Area location", "نظام السداد": "Payment plan", "القرب من العمل أو الدراسة": "Proximity to work or study",
             "جودة الوحدة": "Unit quality", "الاستثمار وإعادة البيع": "Investment & resale", "موعد التسليم": "Handover date", "سمعة الشركة": "Developer reputation"}


def _sv_tr(d):
    """Translate survey category labels (Arabic in the data) when the agent is called with lang='en'."""
    return {SURVEY_EN.get(k, k): v for k, v in d.items()} if _LANG.get() == "en" else d


def _ml_for(pid):
    ml = _load()["ml"]
    if ml is None or "Project_ID" not in ml.columns:
        return None
    r = ml[ml.Project_ID == pid]
    return None if r.empty else r.iloc[0].to_dict()


def _survey_for(district_id):
    s = _load()["survey"]
    if s.empty:
        return None
    e = s[(s.is_eligible == True) & (s.district_id == district_id)]  # noqa: E712
    if len(e) < 15:
        return {"n": int(len(e))}
    q11 = e.q11_problem_changed_decision.dropna()
    q11 = q11[~q11.str.startswith("لا ينطبق")]
    hurt = q11.str.contains("أؤجل|منطقة أخرى").mean() if len(q11) else np.nan
    return {"n": int(len(e)), "decision_hurt_share": hurt,
            "top_problems": _sv_tr(e.q5_most_impactful_problem.value_counts().head(3).to_dict()),
            "top_improvement": _sv_tr(e.q15_top_improvement_priority.value_counts().head(2).to_dict()),
            "top_factors": _sv_tr(e.q10_top_3_decision_factors.dropna().str.split("; ").explode().value_counts().head(4).to_dict())}


def unsold_inventory(project_id=None, top_n=5, discount_rate_pct=None, response_assumptions=None, _sens=True):
    """Explain unsold-units risk, compare alternatives with financial impact, and rank them."""
    D = _load()
    p = D["projects"]
    ml = D["ml"]
    if not project_id:
        top = p.sort_values("risk_score", ascending=False).head(top_n).copy()
        if ml is not None and "Unsold_Risk_Probability" in ml.columns:
            top = top.merge(ml[["Project_ID", "Unsold_Risk_Probability"]], on="Project_ID", how="left")
        cols = [c for c in ["Project_ID", "District_ID", "Developer_Name", "unsold_units", "unsold_ratio", "inventory_age_days",
                            "months_of_inventory", "monthly_velocity", "risk_score", "Unsold_Risk_Probability"] if c in top.columns]
        return _pack("unsold_inventory", T(f"أعلى {len(top)} مشروعات في مخاطر الوحدات غير المباعة", f"Top {len(top)} projects by unsold-units risk"),
                     {"top_risky_projects": top[cols].to_dict("records"),
                      "unsold_risk_rate_pct": float((p.risk_score >= 60).mean() * 100)},
                     [T(f"{r.Project_ID}: غير مباع {r.unsold_units} وحدة ({r.unsold_ratio * 100:.0f}%)، عمر مخزون {r.inventory_age_days:.0f} يوم، "
                        f"يحتاج ~{r.months_of_inventory:.0f} شهرًا للتصريف بسرعة البيع الحالية",
                        f"{r.Project_ID}: {r.unsold_units} unsold units ({r.unsold_ratio * 100:.0f}%), inventory age {r.inventory_age_days:.0f} days, "
                        f"needs ~{r.months_of_inventory:.0f} months to clear at the current sales pace") for r in top.itertuples()],
                     [RISK_WEIGHTS_TXT, T("أشهر المخزون = غير المباع ÷ متوسط البيع الشهري آخر 6 أشهر (تقدير يفترض ثبات السرعة)",
                                          "Months of inventory = unsold ÷ average monthly sales over the last 6 months (assumes a constant pace)")],
                     [T("كل البيانات اصطناعية", "All data is synthetic")],
                     [] if ml is not None else ["ML predictions file (data/ML_Predictions.csv) - model risk probabilities not used yet"])
    row = p[p.Project_ID == project_id]
    if row.empty:
        return _pack("unsold_inventory", T(f"المشروع {project_id} غير موجود", f"Project {project_id} not found"), {"project": None, "likely_causes": [], "suggested_actions": [], "alternatives": []}, [], [], [])
    r = row.iloc[0]
    peers = p[(p.District_ID == r.District_ID) & (p.Project_ID != project_id)]
    au = D["avail_units"]
    pu = au[au.Project_ID == project_id]
    unsold, sold = pu[~pu.is_sold], pu[pu.is_sold]
    dist_ppm2 = au[au.District_ID == r.District_ID].Price_per_m2_Jan2024.mean()

    causes, actions, evidence, extra_missing = [], [], [], [T("مرونة الطلب السعرية (غير مقاسة؛ أسعار المنافسين مقرَّبة بسعر المتر في مشروعات نفس الحي)", "Price elasticity (not measured; competitor prices are approximated by price per m² of same-district projects)"), T("سبب إلغاء كل حجز", "Reason for cancellation of each booking")]
    addressed = {}  # cause key -> alternative that targets it

    def add(cause, sev, key, action):
        causes.append({"cause": cause, "severity": sev, "key": key}); actions.append(action)

    # 1) price vs district
    price_gap = (pu.Price_per_m2_Jan2024.mean() / dist_ppm2 - 1) * 100
    evidence.append(T(f"سعر المتر أعلى/أقل من متوسط الحي بـ {price_gap:+.1f}%", f"Price per m² vs district average: {price_gap:+.1f}%"))
    if price_gap > 5:
        add(T("تسعير أعلى من الحي", "Priced above the district"), "high" if price_gap > 10 else "medium", "price",
            T(f"مراجعة التسعير: خفض تدريجي حتى {round(min(price_gap, 10))}% للوحدات غير المباعة", f"Review pricing: gradual cut of up to {round(min(price_gap, 10))}% on unsold units"))
    # 2) payment plan & down payment
    up = sp = ud = sd = np.nan
    if len(unsold) and len(sold):
        up, sp = unsold.Plan_Years.mean(), sold.Plan_Years.mean()
        ud, sd = unsold.Down_Payment_Pct.mean(), sold.Down_Payment_Pct.mean()
        evidence.append(T(f"خطة سداد غير المباع {up:.1f} سنة ومقدم {ud * 100:.0f}% مقابل المباع {sp:.1f} سنة ومقدم {sd * 100:.0f}%", f"Unsold units: {up:.1f}-year plan, {ud * 100:.0f}% down payment vs sold units: {sp:.1f} years, {sd * 100:.0f}%"))
        if up < sp - 0.5:
            add(T("مدة التقسيط أقصر من الوحدات المباعة", "Instalment period shorter than for sold units"), "medium", "plan", T(f"تمديد التقسيط إلى {np.ceil(sp):.0f} سنوات", f"Extend instalments to {np.ceil(sp):.0f} years"))
        if ud > sd + 0.02:
            add(T("المقدم أعلى من الوحدات المباعة", "Down payment higher than for sold units"), "medium", "down", T(f"تخفيض المقدم إلى {max(sd, 0.05) * 100:.0f}%", f"Reduce the down payment to {max(sd, 0.05) * 100:.0f}%"))
    # 3) unit size mix
    if len(unsold) and len(sold):
        ua, sa = unsold.Area_m2.mean(), sold.Area_m2.mean()
        evidence.append(T(f"متوسط مساحة غير المباع {ua:.0f} م² مقابل المباع {sa:.0f} م²", f"Average unsold unit area {ua:.0f} m² vs sold {sa:.0f} m²"))
        if ua > sa * 1.15:
            add(T("مساحات الوحدات غير المباعة أكبر من المطلوب", "Unsold units are larger than demanded"), "medium", "mix", T("تعديل مزيج الوحدات: طرح/إعادة تقسيم مساحات أصغر أو تسعيرها بحسب الشريحة", "Adjust the unit mix: offer/re-divide into smaller units or price by budget segment"))
    # 4) funnel
    peer_conv = peers.lead_to_booking.mean()
    evidence.append(T(f"تحويل العملاء المحتملين لحجوزات {r.lead_to_booking * 100:.1f}% مقابل {peer_conv * 100:.1f}% في نفس الحي", f"Lead-to-booking conversion {r.lead_to_booking * 100:.1f}% vs {peer_conv * 100:.1f}% in the same district"))
    if r.lead_to_booking < peer_conv * 0.85:
        add(T("ضعف تحويل العملاء لحجوزات", "Weak lead-to-booking conversion"), "medium", "funnel", T("تحسين متابعة العملاء وعروض المعاينة", "Improve lead follow-up and viewing offers"))
    if r.resp_hours > p.resp_hours.median() * 1.3:
        add(T("بطء الرد على العملاء المحتملين", "Slow response to leads"), "low", "response", T(f"تقليل زمن الرد (الحالي {r.resp_hours:.1f} ساعة)", f"Reduce response time (currently {r.resp_hours:.1f} hours)"))
    if r.leads < peers.leads.mean() * 0.7:
        add(T("قلة العملاء المحتملين", "Too few leads"), "medium", "leads", T("زيادة التسويق أو تغيير القنوات", "Increase marketing or change channels"))
    # 5) budget segment
    if len(unsold):
        ticket = unsold.ticket.mean()
        seg, share = (("Under 5M", r.l_lt5) if ticket < 5e6 else ("5M-10M", r.l_5_10) if ticket < 1e7 else ("Over 10M", r.l_gt10))
        share = share / max(r.leads, 1)
        evidence.append(T(f"متوسط سعر الوحدة غير المباعة {ticket:,.0f} (شريحة {seg}) ونسبة العملاء في هذه الشريحة {share * 100:.0f}%", f"Average unsold unit price {ticket:,.0f} (segment {seg}); share of leads in this segment {share * 100:.0f}%"))
        if share < 0.2:
            add(T("عدم توافق سعر الوحدات مع ميزانيات العملاء", "Unit prices do not match client budgets"), "medium", "segment", T(f"إعادة استهداف الحملات أو تعديل مساحات/أسعار شريحة {seg}", f"Retarget campaigns or adjust sizes/prices for segment {seg}"))
    # 6) delivery / construction
    evidence.append(T(f"الإنجاز الإنشائي {r.completion_pct:.0f}% (تأخر {r.delay_pts:.1f} نقطة عن المخطط)", f"Construction {r.completion_pct:.0f}% complete ({r.delay_pts:.1f} pts behind plan)"))
    if r.delay_pts >= 5 and not r.is_completed:
        add(T("تأخر الإنشاء يضعف ثقة المشترين", "Construction delay weakens buyer confidence"), "high" if r.delay_pts >= 10 else "medium", "delivery", T("إعلان جدول تسليم واضح وتقارير تقدم شهرية للمشترين", "Publish a clear handover schedule and monthly progress reports to buyers"))
    # 7) competition
    if len(peers) and len(unsold):
        cheaper = peers[(peers.price_per_m2 < r.price_per_m2 * 0.95) & (peers.sell_through_pct > r.sell_through_pct)]
        evidence.append(T(f"مشروعات منافسة في الحي أرخص وأعلى بيعًا: {len(cheaper)} من {len(peers)}", f"Competing projects in the district that are cheaper and sell faster: {len(cheaper)} of {len(peers)}"))
        if len(cheaper) >= 1 and len(cheaper) >= len(peers) / 2:
            add(T("منافسة من مشروعات الحي الأرخص والأعلى بيعًا", "Competition from cheaper, faster-selling projects in the district"), "medium", "competition", T("مقارنة العروض التنافسية وتقديم ميزة تفاضلية", "Benchmark competing offers and add a differentiating feature"))
    # 7b) unplanned release (units beyond the plan, e.g. P038 phase of 300 units)
    ev = pu[pu.Release_Batch == "EventBatch"]
    if len(ev):
        n_ev, n_ev_unsold = len(ev), int((~ev.is_sold).sum())
        evidence.append(T(f"إصدار مرحلة غير مخططة: {n_ev} وحدة، غير مباع منها {n_ev_unsold}", f"Unplanned release: {n_ev} units, of which {n_ev_unsold} unsold"))
        if n_ev_unsold >= 0.3 * max(len(unsold), 1):
            add(T("إصدار مرحلة غير مخططة أضاف مخزونًا كبيرًا", "An unplanned release added a large amount of inventory"), "high", "release",
                T("تأجيل طرح باقي المرحلة أو تسويقها على دفعات بدل ضخها دفعة واحدة", "Defer the rest of the phase or market it in batches instead of all at once"))
    # 8) district services + survey
    evidence.append(T(f"فجوات الخدمات في الحي: {r.service_gap_count:.0f}، تغطية الخدمات {r.service_coverage_pct:.0f}%", f"District service gaps: {r.service_gap_count:.0f}, service coverage {r.service_coverage_pct:.0f}%"))
    if r.service_gap_count >= 3 or r.service_coverage_pct < 70:
        add(T("ضعف الخدمات في الحي يقلل الطلب", "Weak district services reduce demand"), "medium", "services", T("إبراز خطة الخدمات القادمة أو الاتفاق مع جهات الخدمات", "Highlight the upcoming services plan or agree with service providers"))
    sv = _survey_for(r.District_ID)
    if sv and sv.get("top_problems"):
        tp = T("، ", ", ").join(f"{k} ({v})" for k, v in sv["top_problems"].items())
        hs = sv.get("decision_hurt_share")
        evidence.append(T(f"استبيان الحي ({sv['n']} استجابة): أهم المشكلات {tp}"
                          + (f"؛ {hs * 100:.0f}% ممن فكروا في الشراء أجّلوا أو اختاروا منطقة أخرى بسبب المشكلات" if hs == hs else ""),
                          f"District survey ({sv['n']} responses): top problems {tp}"
                          + (f"; {hs * 100:.0f}% of those who considered buying postponed or chose another area because of these problems" if hs == hs else "")))
        if sv.get("decision_hurt_share") == sv.get("decision_hurt_share") and sv["decision_hurt_share"] >= 0.4:
            add(T("المشكلات المعلنة في الاستبيان أخّرت قرارات الشراء", "Problems reported in the survey delayed purchase decisions"), "medium", "survey",
                T(f"معالجة أولوية السكان الأولى: {next(iter(sv['top_improvement']), 'الخدمات')}", f"Address residents' top priority: {next(iter(sv['top_improvement']), 'services')}"))
    elif sv is None:
        extra_missing.append(T("ردود الاستبيان", "Survey responses"))
    # 9) ML
    mlr = _ml_for(project_id)
    if mlr:
        pr = mlr.get("Unsold_Risk_Probability")
        evidence.append(T(f"نموذج ML: احتمال استمرار عدم البيع {pr * 100:.0f}%", f"ML model: probability the units stay unsold {pr * 100:.0f}%") + (f" (WAPE {mlr['Evaluation_WAPE']})" if "Evaluation_WAPE" in mlr else ""))
    else:
        extra_missing.append(T("توقع الـ ML لمخاطر عدم بيع هذا المشروع (ملف data/ML_Predictions.csv غير متوفر)", "ML unsold-risk prediction for this project (data/ML_Predictions.csv not provided)"))
    if not causes:
        causes.append({"cause": T("لا توجد أسباب واضحة في البيانات", "No clear causes in the data"), "severity": "low", "key": "none"})
        actions.append(T("متابعة الأداء الشهري", "Monitor monthly performance"))

    # ---- alternatives with financial impact
    A = dict(RESPONSE_ASSUMPTIONS); A.update(response_assumptions or {})
    rate = (discount_rate_pct if discount_rate_pct is not None else DISCOUNT_RATE_PCT) / 100
    N = int(r.unsold_units)
    alternatives, baseline = [], None
    if N > 0 and len(unsold):
        P = unsold.ticket.mean(); c = r.investment_cost_per_unit_egp
        d, n = unsold.Down_Payment_Pct.mean(), unsold.Plan_Years.mean()
        v0 = max(r.monthly_velocity, 0.05)
        mkt = (r.marketing or 0) / 6
        base = _econ(N, P, c, d, n, v0, rate, mkt)
        baseline = {"option": T("الوضع الحالي", "Current state"), **base, "velocity_units_per_month": v0, "delta_vs_baseline_egp": 0.0,
                    "avg_unit_price_egp": P, "unit_cost_egp": c, "unsold_units": N}

        def breakeven(P2, d2, n2):
            """Sales-speed uplift (%) this lever would need just to match the current-state value."""
            hi = v0 * 60
            if _econ(N, P2, c, d2, n2, hi, rate, mkt)["net_value_egp"] < base["net_value_egp"]:
                return None
            lo = v0
            for _ in range(40):
                mid = (lo + hi) / 2
                if _econ(N, P2, c, d2, n2, mid, rate, mkt)["net_value_egp"] >= base["net_value_egp"]:
                    hi = mid
                else:
                    lo = mid
            return (hi / v0 - 1) * 100

        def alt(name, desc, key, v, P2=P, d2=d, n2=n, limited=None, quant=True):
            e = _econ(N, P2, c, d2, n2, v, rate, mkt, limited)
            be = None if limited else breakeven(P2, d2, n2)
            alternatives.append({"option": name, "description": desc, "quantified": quant,
                                 "addresses_cause": key in {x["key"] for x in causes}, "velocity_units_per_month": v, **e,
                                 "assumed_speed_uplift_pct": (v / v0 - 1) * 100, "breakeven_speed_uplift_pct": be,
                                 "delta_vs_baseline_egp": e["net_value_egp"] - base["net_value_egp"],
                                 "months_saved": base["months_to_clear"] - e["months_to_clear"]})
        x = A["price_cut_pct"]
        alt(T(f"خفض السعر {x}%", f"Price cut {x}%"), T(f"خفض سعر الوحدات غير المباعة {x}% (مرونة مفترضة {A['price_elasticity']})", f"Cut the price of unsold units by {x}% (assumed elasticity {A['price_elasticity']})"), "price",
            v0 * (1 + A["price_elasticity"] * x / 100), P2=P * (1 - x / 100))
        y = A["extend_years"]
        alt(T(f"تمديد التقسيط +{y} سنة", f"Extend instalments +{y} years"), T(f"من {n:.0f} إلى {n + y:.0f} سنوات بنفس السعر", f"From {n:.0f} to {n + y:.0f} years at the same price"), "plan",
            v0 * (1 + A["uplift_per_extra_year"] * y), n2=n + y)
        z = min(A["down_payment_cut_pts"], max(d * 100 - 2, 0))
        alt(T(f"خفض المقدم -{z:.0f} نقطة", f"Down payment -{z:.0f} pts"), T(f"من {d * 100:.0f}% إلى {d * 100 - z:.0f}%", f"From {d * 100:.0f}% to {d * 100 - z:.0f}%"), "down",
            v0 * (1 + A["uplift_per_dp_pt"] * z), d2=d - z / 100)
        lx, lm = A["limited_discount_pct"], A["limited_months"]
        v1 = v0 * (1 + A["limited_uplift"])
        alt(T(f"خصم محدود {lx}% لمدة {lm} شهور", f"Limited {lx}% discount for {lm} months"), T("خصم على الوحدات المباعة خلال النافذة ثم عودة السعر", "Discount on units sold during the window, then the price returns"), "price",
            v1, limited=(lx / 100, lm, v1, v0))
        alt(T("تحسين الخدمات / مزيج الوحدات", "Improve services / unit mix"), T("تحسين الوصول للخدمات أو تعديل مساحات الوحدات (غير قابل للقياس مالياً بدون بيانات تكلفة وأثر)", "Improve access to services or adjust unit sizes (cannot be quantified financially without cost and impact data)"),
            "services", v0, quant=False)
        alternatives[-1].update({"net_value_egp": None, "delta_vs_baseline_egp": None, "months_to_clear": None, "margin_pct": None, "avg_price_egp": None, "months_saved": None})
        if "mix" in {x["key"] for x in causes}:
            alternatives[-1]["addresses_cause"] = True
        q = sorted([a for a in alternatives if a["quantified"]], key=lambda a: a["delta_vs_baseline_egp"], reverse=True)
        rel = [a for a in q if a["addresses_cause"]]
        for i, a in enumerate(q):
            a["rank"] = i + 1
        best = (rel or q)[0] if q else None
        if best is not None and best["delta_vs_baseline_egp"] <= 0:
            best = next((a for a in q if a["delta_vs_baseline_egp"] > 0), None)   # only recommend what adds value
        for a in alternatives:
            a["recommended"] = bool(best and a["option"] == best["option"])
        least_bad = q[0] if q else None
        alternatives = q + [a for a in alternatives if not a["quantified"]]

    # sensitivity: does ANY financial alternative add value across discount rates x response strengths?
    sensitivity, sens_txt = [], ""
    if _sens and alternatives and project_id:
        base_resp = {k: A[k] for k in ("price_elasticity", "uplift_per_extra_year", "uplift_per_dp_pt", "limited_uplift")}
        for dr in (8, 12, 18, 25):
            for k in (0.5, 1, 2):
                sub = unsold_inventory(project_id, discount_rate_pct=dr,
                                       response_assumptions={**A, **{n: v * k for n, v in base_resp.items()}}, _sens=False)
                qa = [a for a in sub["results"]["alternatives"] if a.get("quantified")]
                if qa:
                    b = max(qa, key=lambda a: a["delta_vs_baseline_egp"])
                    sensitivity.append({"discount_rate_pct": dr, "response_strength": k, "best_option": b["option"],
                                        "best_delta_egp": b["delta_vs_baseline_egp"], "adds_value": b["delta_vs_baseline_egp"] > 0})
        n_ok = sum(x["adds_value"] for x in sensitivity)
        sens_txt = T(f"اختبار الحساسية: يوجد بديل يرفع القيمة في {n_ok} من {len(sensitivity)} سيناريو (معدل خصم 8-25% × قوة استجابة 0.5-2 ضعف)",
                     f"Sensitivity test: an alternative adds value in {n_ok} of {len(sensitivity)} scenarios (discount rate 8-25% × response strength 0.5-2x)")
        if n_ok in (0, len(sensitivity)):
            sens_txt += T(" - النتيجة ثابتة عبر كل الافتراضات المختبرة", " - the result is the same across all tested assumptions")
        else:
            sens_txt += T(" - التوصية تعتمد على الافتراضات، فتعامل معها بحذر", " - the recommendation depends on the assumptions; treat it with caution")
    unsold_types = unsold.Unit_Type.value_counts().to_dict() if len(unsold) else {}
    best_txt, recommendation = "", None
    if alternatives:
        b = next((a for a in alternatives if a.get("recommended")), None)
        if b:
            recommendation = T(f"البديل الأفضل وفق الافتراضات: {b['option']} (فرق القيمة {b['delta_vs_baseline_egp'] / 1e6:+,.1f} مليون جنيه)",
                               f"Best alternative under the assumptions: {b['option']} (value difference EGP {b['delta_vs_baseline_egp'] / 1e6:+,.1f}M)")
        else:
            lb = least_bad
            recommendation = T("لا يوجد بديل مالي يرفع القيمة وفق الافتراضات الحالية؛ أقل البدائل تكلفة: "
                               f"{lb['option']} ({lb['delta_vs_baseline_egp'] / 1e6:+,.1f} مليون). يُنصح بمعالجة الأسباب غير السعرية أولاً "
                               "(الخدمات، الإنشاء، مزيج الوحدات) أو اختبار البدائل على دفعة صغيرة لقياس الاستجابة الفعلية",
                               "No financial alternative adds value under the current assumptions; the least costly is: "
                               f"{lb['option']} (EGP {lb['delta_vs_baseline_egp'] / 1e6:+,.1f}M). Address the non-price causes first "
                               "(services, construction, unit mix) or test alternatives on a small batch to measure the real response")
        if sens_txt:
            recommendation = (recommendation + " | " + sens_txt) if recommendation else sens_txt
        best_txt = " - " + recommendation.split(";")[0].split(": ", 1)[0] if False else ""
    return _pack(
        "unsold_inventory",
        T(f"{project_id}: {int(r.unsold_units)} وحدة غير مباعة ({r.unsold_ratio * 100:.0f}%) - مخاطرة {r.risk_score:.0f}/100", f"{project_id}: {int(r.unsold_units)} unsold units ({r.unsold_ratio * 100:.0f}%) - risk {r.risk_score:.0f}/100"),
        {"project": {k: r[k] for k in ["Project_ID", "Project_Name", "District_ID", "Developer_Name", "unsold_units", "unsold_ratio",
                                       "inventory_age_days", "months_of_inventory", "monthly_velocity", "risk_score"]},
         "unsold_by_type": unsold_types, "likely_causes": causes, "suggested_actions": actions,
         "baseline": baseline, "alternatives": alternatives, "recommendation": recommendation, "survey": sv, "ml": mlr, "sensitivity": sensitivity},
        evidence,
        [T("الأسباب مستنتجة من قواعد بسيطة مقارنة بنفس الحي، وليست إثباتًا لسبب فعلي", "Causes are inferred from simple rules versus the same district; they are not proof of an actual cause"),
         T(f"أثر البدائل: معاملات استجابة افتراضية وليست مقاسة (خفض السعر: مرونة {A['price_elasticity']}؛ +{A['uplift_per_extra_year'] * 100:.0f}% سرعة لكل سنة تقسيط؛ "
           f"+{A['uplift_per_dp_pt'] * 100:.1f}% لكل نقطة مقدم؛ خصم محدود +{A['limited_uplift'] * 100:.0f}% خلال النافذة)، قابلة للتعديل عبر response_assumptions",
           f"Alternatives' impact uses assumed, not measured, response coefficients (price cut: elasticity {A['price_elasticity']}; +{A['uplift_per_extra_year'] * 100:.0f}% speed per extra instalment year; "
           f"+{A['uplift_per_dp_pt'] * 100:.1f}% per down-payment point; limited discount +{A['limited_uplift'] * 100:.0f}% during the window), adjustable via response_assumptions"),
         T(f"القيمة = القيمة الحالية لصافي (السعر وفق خطة السداد − التكلفة المفترضة) × الوحدات غير المباعة بخصم {rate * 100:.0f}% سنويًا، ناقص مصروف التسويق حتى التصريف",
           f"Value = present value of net (price per payment plan − assumed cost) × unsold units, discounted at {rate * 100:.0f}% a year, minus marketing spend until sell-out"),
         T("سرعة البيع الأساسية = متوسط آخر 6 أشهر؛ السعر = سعر الأساس × نسبة العقود الفعلية", "Baseline sales pace = average of the last 6 months; price = base price × actual contract ratio"), RISK_WEIGHTS_TXT],
        [T("لا يمكن تقدير أثر خفض السعر على المبيعات دون بيانات مرونة الطلب، فالأرقام مقارنة نسبية بين البدائل وليست توقعًا",
           "The effect of a price cut on sales cannot be estimated without demand-elasticity data, so the figures are a relative comparison between alternatives, not a forecast"),
         T(f"عمر المخزون {r.inventory_age_days:.0f} يوم يزيد تكلفة التمويل", f"Inventory age of {r.inventory_age_days:.0f} days increases financing cost"),
         T("تخفيض السعر قد يؤثر في أسعار الوحدات المباعة والمشروعات المجاورة", "A price cut may affect the prices of sold units and neighbouring projects")],
        extra_missing)


# ----------------------------------------------------------------- KPIs (dashboard backend)
def portfolio_kpis():
    D = _load()
    p, inv = D["projects"], D["Fact_Inventory_Monthly"]
    last12 = sorted(inv.Month.unique())[-12:]
    i12 = inv[inv.Month.isin(last12)]
    return _clean({
        "total_sales_value_12m_egp": i12.Sales_Value_EGP.sum(),
        "units_sold_12m": i12.Units_Sold.sum(),
        "overall_sell_through_pct": p.sold_units_observed.sum() / p.Units_Planned.sum() * 100,
        "unsold_units": p.unsold_units.sum(),
        "avg_inventory_age_days": p.inventory_age_days.mean(),
        "collection_rate_pct": p.collection_rate.mean() * 100,
        "overdue_balance_egp": p.overdue_latest_egp.sum(),
        "cancellation_rate_pct": p.cancel_rate.mean() * 100,
        "lead_to_booking_pct": p.bookings.sum() / p.leads.sum() * 100,
        "avg_construction_delay_pts": p.delay_pts.mean(),
        "avg_assumed_margin_pct": p.margin_pct.mean(),
        "avg_risk_score": p.risk_score.mean(),
        "high_risk_projects": int((p.risk_score >= 60).sum()),
        "unsold_risk_rate_pct": float((p.risk_score >= 60).mean() * 100),
        "months_of_inventory": p.unsold_units.sum() / max(p.monthly_velocity.sum(), 0.1),
        "data_status": DATA_STATUS,
    })


def district_kpis():
    p = _load()["projects"]
    g = (p.groupby(["District_ID", "District_Name", "Lifecycle_Archetype"])
         .agg(projects=("Project_ID", "count"), units_planned=("Units_Planned", "sum"),
              units_sold=("sold_units_observed", "sum"), unsold_units=("unsold_units", "sum"),
              sales_value_12m_egp=("sales_value_12m", "sum"), avg_inventory_age=("inventory_age_days", "mean"),
              collection_rate=("collection_rate", "mean"), avg_risk=("risk_score", "mean"),
              avg_margin_pct=("margin_pct", "mean"), service_coverage_pct=("service_coverage_pct", "first"),
              service_gap_count=("service_gap_count", "first"), occupancy_pct=("occupancy_pct", "first"))
         .reset_index())
    g["sell_through_pct"] = g.units_sold / g.units_planned * 100
    return _clean(g.to_dict("records"))


def project_kpis():
    cols = ["Project_ID", "Project_Name", "District_ID", "Developer_ID", "Developer_Name", "Units_Planned", "sold_units_observed",
            "sell_through_pct", "unsold_units", "inventory_age_days", "months_of_inventory", "sales_value_12m", "lead_to_booking",
            "collection_rate", "completion_pct", "delay_pts", "cost_overrun_pct", "cancel_rate", "margin_pct", "risk_score"]
    return _clean(_load()["projects"][cols].to_dict("records"))


# ----------------------------------------------------------------- free-text front door
_AR = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")
_TYPES = {"شقة": "Apartment", "شقق": "Apartment", "apartment": "Apartment", "فيلا": "Villa", "فلل": "Villa", "villa": "Villa",
          "تاون": "Townhouse", "townhouse": "Townhouse", "تجاري": "Commercial", "محل": "Commercial", "commercial": "Commercial",
          "مكتب": "Office", "إداري": "Office", "اداري": "Office", "office": "Office"}


def parse_request(text):
    t = (text or "").translate(_AR).lower()
    out = {}
    m = re.search(r"(\d+(?:\.\d+)?)\s*(مليون|million|m\b|م\b)", t)
    if m:
        out["budget_egp"] = float(m.group(1)) * 1e6
    else:
        m = re.search(r"(\d{6,})", t.replace(",", ""))
        if m:
            out["budget_egp"] = float(m.group(1))
    m = re.search(r"(\d+)\s*(سنوات|سنة|سنين|years?|yr)", t)
    if m:
        out["horizon_years"] = int(m.group(1))
    m = re.search(r"(?:أفضل|افضل|top)\s*(\d+)", t)
    if m:
        out["top_n"] = int(m.group(1))
    if re.search(r"مخاطر\w*\s*(منخفض|قليل|ضعيف)|low risk|متحفظ", t):
        out["risk_tolerance"] = "low"
    elif re.search(r"مخاطر\w*\s*(مرتفع|عالي|كبير)|high risk|مغامر", t):
        out["risk_tolerance"] = "high"
    for k, v in _TYPES.items():
        if k in t:
            out["unit_type"] = v
            break
    if re.search(r"سيولة\s*(عالي|مرتفع|سريع)|تخارج سريع", t):
        out["liquidity_need"] = "high"
    ids = {"project_id": re.search(r"\bp\s?0*(\d{1,3})\b", t), "developer_id": re.search(r"\bdev\s?(\d+)\b", t),
           "district_id": re.search(r"\bd\s?0*(\d{1,2})\b", t)}
    if ids["project_id"]:
        out["project_id"] = "P%03d" % int(ids["project_id"].group(1))
    if ids["developer_id"]:
        out["developer_id"] = "DEV" + ids["developer_id"].group(1)
    if ids["district_id"]:
        out["district_id"] = "D%02d" % int(ids["district_id"].group(1))
    return out


def ask(text, lang="ar", **defaults):
    """Free-text request -> one or more agents. Returns {parsed, answers:[agent outputs]}."""
    pr = parse_request(text)
    t = (text or "").lower()
    want_unsold = bool(re.search(r"غير المباع|مخزون|unsold|خصم|تقسيط|الوحدات الراكدة|راكد", t))
    want_dev = bool(re.search(r"مطور|شركة|شركات|أداء|اداء|developer", t))
    want_inv = bool(re.search(r"استثمار|ميزانية|عائد|invest|budget|return", t)) or "budget_egp" in pr
    which = [n for n, w in [("investment_advisor", want_inv), ("developer_performance", want_dev), ("unsold_inventory", want_unsold)] if w]
    if not which:
        which = ["investment_advisor"] if "budget_egp" in pr else ["unsold_inventory"]
    answers, used = [], {}
    for name in which:
        if name == "investment_advisor":
            kw = {k: pr[k] for k in ("budget_egp", "horizon_years", "top_n", "risk_tolerance", "unit_type", "liquidity_need", "district_id") if k in pr}
            kw.setdefault("budget_egp", defaults.get("budget_egp", 10_000_000))
        elif name == "developer_performance":
            kw = {k: pr[k] for k in ("developer_id",) if k in pr}
        else:
            kw = {k: pr[k] for k in ("project_id", "top_n") if k in pr}
        kw = {**{k: v for k, v in defaults.items() if k in ("project_id", "district_id") and name != "developer_performance" and k not in kw and (k != "district_id" or name == "investment_advisor") and (k != "project_id" or name == "unsold_inventory")}, **kw}
        used[name] = kw
        answers.append(run_agent(name, lang=lang, **kw))
    return {"parsed": pr, "params_used": used, "answers": answers}


# ----------------------------------------------------------------- unified entry
AGENTS = {
    "investment_advisor": investment_advisor,
    "developer_performance": developer_performance,
    "unsold_inventory": unsold_inventory,
}


def run_agent(name, lang="ar", **params):
    """lang: 'ar' (default) or 'en' - language of the generated evidence / assumptions / risks / recommendations."""
    if name not in AGENTS:
        raise ValueError(f"Unknown agent '{name}'. Available: {list(AGENTS)}")
    tok = _LANG.set("en" if str(lang).lower().startswith("en") else "ar")
    try:
        return AGENTS[name](**params)
    finally:
        _LANG.reset(tok)


if __name__ == "__main__":
    import json
    print(json.dumps(run_agent("investment_advisor", budget_egp=10_000_000, horizon_years=5), ensure_ascii=False, indent=2)[:3000])
