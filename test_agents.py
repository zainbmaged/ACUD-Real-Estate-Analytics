"""Run:  python -m pytest -q      (from the acud folder)
Checks every agent on known cases from the synthetic data and verifies that no recommendation is unsupported."""
import json
import os
import numpy as np
import pandas as pd
import pytest
import agents
import dashboard

REQUIRED = ["agent", "summary", "results", "evidence", "assumptions", "risks", "missing_data", "data_status"]


def _ok(r):
    assert all(k in r for k in REQUIRED)
    json.dumps(r, allow_nan=False)            # JSON-safe, no NaN
    assert r["missing_data"] and r["data_status"].startswith("Synthetic")


# ------------------------------------------------------------- shared contract
@pytest.mark.parametrize("name,kw", [
    ("investment_advisor", dict(budget_egp=10_000_000)),
    ("investment_advisor", dict(budget_egp=6_000_000, risk_tolerance="low", unit_type="Apartment", liquidity_need="high")),
    ("investment_advisor", dict(budget_egp=100)),
    ("developer_performance", {}), ("developer_performance", dict(developer_id="DEV5")),
    ("developer_performance", dict(stage="completed")), ("developer_performance", dict(developer_id="NOPE")),
    ("unsold_inventory", {}), ("unsold_inventory", dict(project_id="P030")), ("unsold_inventory", dict(project_id="P006")),
    ("unsold_inventory", dict(project_id="P999")),
])
def test_contract(name, kw):
    _ok(agents.run_agent(name, **kw))


def test_unknown_agent():
    with pytest.raises(ValueError):
        agents.run_agent("nope")


# ------------------------------------------------------------- investment advisor
def test_budget_respected_and_only_available_units():
    r = agents.run_agent("investment_advisor", budget_egp=6_000_000, top_n=10)
    au = agents._load()["avail_units"]
    for p in r["results"]["top_projects"]:
        assert p["ticket"] <= 6_000_000
        assert p["units_in_budget"] >= 1
        assert au[(au.Project_ID == p["Project_ID"]) & (~au.is_sold)].shape[0] >= 1


def test_sold_out_project_never_recommended():
    r = agents.run_agent("investment_advisor", budget_egp=50_000_000, top_n=40)
    assert "P006" not in [p["Project_ID"] for p in r["results"]["top_projects"]]   # P006 is 100% sold


def test_scenarios_ordered_and_flows_consistent():
    r = agents.run_agent("investment_advisor", budget_egp=10_000_000, horizon_years=5)
    for p in r["results"]["top_projects"]:
        s = p["scenarios"]
        assert s["conservative"]["net_profit_egp"] < s["base"]["net_profit_egp"] < s["optimistic"]["net_profit_egp"]
        assert len(s["base"]["net_cash_flows"]) == 6
        assert abs(sum(s["base"]["net_cash_flows"]) - s["base"]["net_profit_egp"]) < 10


def test_unit_type_filter():
    r = agents.run_agent("investment_advisor", budget_egp=50_000_000, unit_type="Villa", top_n=10)
    assert r["results"]["top_projects"] and all("Villa" in p["types"] for p in r["results"]["top_projects"])


def test_weights_change_ranking_and_normalise():
    a = agents.run_agent("investment_advisor", budget_egp=30_000_000, top_n=5, weights={"return": 1, "risk": 0, "liquidity": 0, "services": 0, "construction": 0, "price_value": 0})
    b = agents.run_agent("investment_advisor", budget_egp=30_000_000, top_n=5, weights={"risk": 1, "return": 0, "liquidity": 0, "services": 0, "construction": 0, "price_value": 0})
    assert abs(sum(a["results"]["weights_used"].values()) - 1) < 0.01
    ia, ib = [p["Project_ID"] for p in a["results"]["top_projects"]], [p["Project_ID"] for p in b["results"]["top_projects"]]
    assert ia != ib
    irr = [p["irr_base_pct"] for p in a["results"]["top_projects"]]
    assert irr == sorted(irr, reverse=True)           # return-only weights => ordered by return


def test_impossible_budget_returns_message_not_error():
    r = agents.run_agent("investment_advisor", budget_egp=100)
    assert r["results"]["top_projects"] == [] and "لا توجد" in r["summary"]


# ------------------------------------------------------------- developer performance
def test_dev_ranking_complete_and_subscores_bounded():
    r = agents.run_agent("developer_performance")
    rk = r["results"]["ranking"]
    assert len(rk) == 8 and [x["rank"] for x in rk] == list(range(1, 9))
    for x in rk:
        for k in ["sub_sales", "sub_profitability", "sub_collection", "sub_delivery", "sub_inventory"]:
            assert -1e-6 <= x[k] <= 100 + 1e-6
    assert abs(sum(r["results"]["weights_used"].values()) - 1) < 0.01
    assert r["results"]["weights_used"]["sales"] == 0.25


def test_dev_not_ranked_by_volume_alone():
    rk = agents.run_agent("developer_performance")["results"]["ranking"]
    assert [x["Developer_ID"] for x in rk] != [x["Developer_ID"] for x in sorted(rk, key=lambda x: -x["units_sold"])]


def test_dev_detail_has_trends_and_known_weak_project():
    r = agents.run_agent("developer_performance", developer_id="DEV5")
    R = r["results"]
    assert R["projects"][0]["Project_ID"] == "P030"        # known weakest project of DEV5
    assert {t["status"] for t in R["trends"]} <= {"improved", "declined", "stable"} and len(R["trends"]) >= 5


def test_dev_filters_apply():
    a = agents.run_agent("developer_performance")["results"]["ranking"]
    b = agents.run_agent("developer_performance", stage="completed")["results"]["ranking"]
    assert sum(x["projects"] for x in b) < sum(x["projects"] for x in a)


# ------------------------------------------------------------- unsold inventory
def test_unsold_top_list_sorted():
    r = agents.run_agent("unsold_inventory", top_n=5)
    s = [x["risk_score"] for x in r["results"]["top_risky_projects"]]
    assert s == sorted(s, reverse=True) and r["results"]["top_risky_projects"][0]["Project_ID"] == "P030"


def test_unsold_known_case_P030():
    R = agents.run_agent("unsold_inventory", project_id="P030")["results"]
    keys = {c["key"] for c in R["likely_causes"]}
    assert {"delivery", "services"} <= keys                       # 20.9-pt construction delay + 32% service coverage
    assert len(R["alternatives"]) >= 5 and R["baseline"]["unsold_units"] == 68


def test_unsold_no_unsupported_recommendation():
    """Recommended alternative must add value; otherwise the agent must say no option pays for itself."""
    for pid in ["P030", "P035", "P008", "P012"]:
        R = agents.run_agent("unsold_inventory", project_id=pid)["results"]
        rec = [a for a in R["alternatives"] if a["recommended"]]
        for a in rec:
            assert a["delta_vs_baseline_egp"] > 0
        if not rec and R["alternatives"]:
            assert "لا يوجد بديل" in R["recommendation"]


def test_unsold_economics_react_to_assumptions():
    lo = agents.run_agent("unsold_inventory", project_id="P030", discount_rate_pct=5)["results"]
    hi = agents.run_agent("unsold_inventory", project_id="P030", discount_rate_pct=25)["results"]
    d = lambda R, n: next(a for a in R["alternatives"] if a["option"].startswith(n))["delta_vs_baseline_egp"]
    assert d(lo, "تمديد") > d(hi, "تمديد")                        # longer plans hurt more when money is expensive
    base = agents.run_agent("unsold_inventory", project_id="P030")["results"]
    big = agents.run_agent("unsold_inventory", project_id="P030", response_assumptions={"price_elasticity": 8})["results"]
    assert d(big, "خفض السعر") > d(base, "خفض السعر")             # higher assumed elasticity => price cut looks better


def test_unsold_sold_out_project_has_no_alternatives():
    R = agents.run_agent("unsold_inventory", project_id="P006")["results"]
    assert R["alternatives"] == [] and R["project"]["unsold_units"] == 0


def test_ml_hook_is_reported_missing():
    r = agents.run_agent("unsold_inventory", project_id="P030")
    if agents._load()["ml"] is None:
        assert any("ML" in m for m in r["missing_data"])


# ------------------------------------------------------------- free text
def test_ask_parses_arabic():
    p = agents.parse_request("قارن أفضل 3 مشروعات بميزانية 8 مليون لمدة 5 سنوات مخاطر منخفضة شقق")
    assert p["budget_egp"] == 8e6 and p["horizon_years"] == 5 and p["top_n"] == 3 and p["risk_tolerance"] == "low" and p["unit_type"] == "Apartment"
    q = agents.parse_request("المخزون غير المباع في p30 للمطور DEV5")
    assert q["project_id"] == "P030" and q["developer_id"] == "DEV5"


def test_ask_routes_to_both_agents():
    r = agents.ask("قارن أفضل 3 مشروعات للاستثمار بميزانية 8 مليون، واقترح حلولًا للمخزون غير المباع في P030")
    assert [a["agent"] for a in r["answers"]] == ["investment_advisor", "unsold_inventory"]
    assert r["answers"][1]["results"]["project"]["Project_ID"] == "P030"


# ------------------------------------------------------------- dashboard consistency
def test_dashboard_matches_agents():
    k = agents.portfolio_kpis()
    assert dashboard.overview()["kpis"]["unsold_units"] == k["unsold_units"]
    inv = dashboard.inventory_risk()
    assert inv["kpis"]["unsold_units"] == k["unsold_units"]
    assert inv["projects"][0]["Project_ID"] == agents.run_agent("unsold_inventory", top_n=1)["results"]["top_risky_projects"][0]["Project_ID"]


@pytest.mark.parametrize("fn,args", [(dashboard.meta, ()), (dashboard.overview, ()), (dashboard.sales_financial, ()), (dashboard.sales_financial, ("D03",)),
                                     (dashboard.inventory_risk, ()), (dashboard.developers, ()), (dashboard.smart_city, ()),
                                     (dashboard.sustainability, ()), (dashboard.project_detail, ("P030",))])
def test_dashboard_json_safe(fn, args):
    json.dumps(fn(*args), allow_nan=False)


def test_project_detail_unknown():
    assert dashboard.project_detail("P999") is None


# ---------------------------------------------------------------- reconciliation fixes
def test_unit_counts_reconcile_across_sources():
    D = agents._load()
    p = D["projects"]
    inv = D["Fact_Inventory_Monthly"]
    last = inv[inv.Month == inv.Month.max()]
    assert p.unsold_units.sum() == last.Available_End.sum() == len(D["avail_units"][~D["avail_units"].is_sold]) == 2201
    assert p.loc[p.Project_ID == "P038", "units_unplanned_release"].iloc[0] == 300


def test_one_number_everywhere():
    k = agents.portfolio_kpis()["unsold_units"]
    assert k == 2201 == dashboard.overview()["kpis"]["unsold_units"]


def test_collection_rate_capped_but_raw_kept():
    p = agents._load()["projects"]
    assert p.collection_rate.max() <= 1.0
    assert p.collection_rate_raw.max() > 1.0


def test_unplanned_release_is_a_cause_for_p038():
    r = agents.unsold_inventory("P038")["results"]
    assert "release" in {c["key"] for c in r["likely_causes"]}


def test_sensitivity_sweep_present_and_consistent():
    r = agents.unsold_inventory("P030")["results"]
    sens = r["sensitivity"]
    assert len(sens) == 12
    if not any(x["adds_value"] for x in sens):
        assert not any(a.get("recommended") for a in r["alternatives"])


# ---------------------------------------------------------------- bilingual outputs
import re as _re
_AR = _re.compile("[\u0600-\u06ff]")


@pytest.mark.parametrize("name,kw", [("investment_advisor", dict(budget_egp=8e6, risk_tolerance="low")),
                                     ("developer_performance", {}), ("developer_performance", dict(developer_id="DEV5")),
                                     ("unsold_inventory", {}), ("unsold_inventory", dict(project_id="P038")),
                                     ("unsold_inventory", dict(project_id="P030"))])
def test_english_output_has_no_arabic(name, kw):
    out = json.dumps(agents.run_agent(name, lang="en", **kw), ensure_ascii=False)
    assert not _AR.search(out)


def test_arabic_is_default_and_language_does_not_leak():
    ar1 = agents.run_agent("unsold_inventory", project_id="P038")["summary"]
    en = agents.run_agent("unsold_inventory", project_id="P038", lang="en")["summary"]
    ar2 = agents.run_agent("unsold_inventory", project_id="P038")["summary"]
    assert _AR.search(ar1) and ar1 == ar2 and not _AR.search(en)


def test_same_numbers_in_both_languages():
    a = agents.run_agent("unsold_inventory", project_id="P038")["results"]
    e = agents.run_agent("unsold_inventory", project_id="P038", lang="en")["results"]
    assert a["project"]["unsold_units"] == e["project"]["unsold_units"]
    assert [x["net_value_egp"] for x in a["alternatives"]] == [x["net_value_egp"] for x in e["alternatives"]]


def test_ask_respects_lang():
    o = agents.ask("budget 8 million, low risk", lang="en")
    assert not _AR.search(json.dumps(o, ensure_ascii=False))


# ---------------------------------------------------------------- sustainability data (synthetic fill-ins) + branding
def test_sustainability_inputs_present_and_consistent():
    s = dashboard.sustainability()
    k = s["kpis"]
    assert 3 < k["renewable_share_pct_12m"] < 15
    assert k["electricity_cost_12m_egp"] > 0 and k["water_cost_12m_egp"] > 0
    assert 80 < k["liters_per_resident_day"] < 300
    assert k["portfolio_payback_years"] is not None and 1 < k["portfolio_payback_years"] < 15
    assert s["completeness"].startswith("complete")
    D = agents._load()
    res = D["Fact_Residents_Monthly"]
    assert (res.Residents > res.Occupied_Units).where(res.Occupied_Units > 1, True).all()          # >1 resident per occupied unit
    sol = D["Fact_Solar_Monthly"].merge(D["Fact_Utilities_Monthly"], on=["District_ID", "Month"])
    assert (sol.Solar_Generation_kWh <= sol.Electricity_kWh).all()


def test_pilot_savings_are_measured_not_copied_from_expectation():
    dep = pd.DataFrame(dashboard.sustainability()["deployments"])
    pilots = dep[dep.Solution_ID != "S01"]
    assert pilots.measured_saving_pct.notna().all() and (pilots.measured_saving_pct > 0).all()
    assert not (pilots.measured_saving_pct.round(1) == pilots.expected_saving_pct.round(1)).all()
    assert (dep.payback_years.dropna() > 0).all()


def test_dashboard_has_logo_brand_colours_and_no_synthetic_banner():
    h = open(os.path.join(os.path.dirname(__file__), "dashboard.html"), encoding="utf-8").read()
    assert "data:image/png;base64," in h and "#ED1C24" in h
    assert "const synth = () => \"\"" in h


def test_missing_data_follows_language():
    for name, kw in [("investment_advisor", dict(budget_egp=8e6)), ("developer_performance", {}), ("unsold_inventory", dict(project_id="P038"))]:
        assert all(_AR.search(x) for x in agents.run_agent(name, **kw)["missing_data"])
        assert not any(_AR.search(x) for x in agents.run_agent(name, lang="en", **kw)["missing_data"])


# ------------------------------------------------------------- rent / inflation / exit depth (added assumptions)
def _top(**kw):
    return agents.run_agent("investment_advisor", budget_egp=10_000_000, horizon_years=5, top_n=3, **kw)["results"]["top_projects"]


def test_rent_raises_return_and_is_overridable():
    base = _top()
    no_rent = _top(rental_yield_pct=0, service_charge_pct=0)
    assert all(b["irr_base_pct"] > n["irr_base_pct"] for b, n in zip(sorted(base, key=lambda x: x["Project_ID"]), sorted(no_rent, key=lambda x: x["Project_ID"])))
    hi = _top(rental_yield_pct=12)
    assert max(x["irr_base_pct"] for x in hi) > max(x["irr_base_pct"] for x in base)
    for p in base:
        assert p["rent_income_base_egp"] > 0 and 0 < p["net_rent_yield_pct"] < p["gross_rent_yield_pct"]


def test_service_charges_and_vacancy_reduce_net_rent():
    a, b = _top(vacancy_pct=0, service_charge_pct=0), _top(vacancy_pct=40, service_charge_pct=3)
    ra, rb = {p["Project_ID"]: p["rent_income_base_egp"] for p in a}, {p["Project_ID"]: p["rent_income_base_egp"] for p in b}
    assert all(rb[k] < ra[k] for k in ra if k in rb)


def test_inflation_gives_real_return_below_nominal_and_cash_flows_stay_consistent():
    r = agents.run_agent("investment_advisor", budget_egp=10_000_000, horizon_years=5, inflation_pct=14)["results"]["top_projects"]
    lo = agents.run_agent("investment_advisor", budget_egp=10_000_000, horizon_years=5, inflation_pct=5)["results"]["top_projects"]
    for p, q in zip(r, lo):
        s = p["scenarios"]["base"]
        assert s["irr_real_pct"] < s["irr_pct"]
        assert abs((1 + s["irr_pct"] / 100) / 1.14 - 1 - s["irr_real_pct"] / 100) < 1e-4      # real = (1+nominal)/(1+infl) - 1
        assert s["net_profit_real_egp"] < s["net_profit_egp"]
        assert abs(sum(s["net_cash_flows"]) - s["net_profit_egp"]) < 10
        assert q["scenarios"]["base"]["irr_real_pct"] > s["irr_real_pct"]
        assert p["scenarios"]["conservative"]["irr_pct"] < s["irr_pct"] < p["scenarios"]["optimistic"]["irr_pct"]


def test_rent_starts_only_after_handover():
    flows = agents._rent_flows(1_000_000, 5, 0.1, 8.0, 1.0, 0.0, start_month=30)    # handover in month 30
    assert flows[1] == 0 and flows[2] == 0 and flows[3] > 0                          # years 1-2 (months 0-24) empty, year 3 half-year
    assert flows[3] < flows[4]                                                       # partial first year < full year
    assert agents._rent_flows(1e6, 5, 0.1, 8, 1, 0, 100)[1:] == [0.0] * 5            # beyond the horizon: no rent


def test_handover_timing_is_derived_from_construction_progress():
    p = agents._load()["projects"].set_index("Project_ID")
    assert (p.months_to_handover >= agents.HANDOVER_LAG_MONTHS).all()
    done, building = p[p.is_completed], p[~p.is_completed]
    assert (done.months_to_handover == agents.HANDOVER_LAG_MONTHS).all()
    assert (building.months_to_handover > agents.HANDOVER_LAG_MONTHS).all()
    assert building.loc["P027"].months_to_handover > building.loc["P017"].months_to_handover   # 46% built vs 87% built
    assert done.forecast_completion_month.isna().all() and building.forecast_completion_month.notna().all()


def test_exit_depth_blends_project_and_district_and_drives_exit_discount():
    p = agents._load()["projects"]
    exp = agents.EXIT_WEIGHT_PROJECT * p.months_of_inventory + (1 - agents.EXIT_WEIGHT_PROJECT) * p.district_months_of_inventory
    assert np.allclose(p.exit_depth_months, exp)
    assert set(p.exit_depth_label) <= {"deep", "medium", "thin"}
    for t in _top():
        assert abs(t["exit_discount_pct"] - min(10.0, 2 + 0.15 * t["exit_depth_months"])) < 0.01


def test_added_assumptions_are_declared_and_removed_from_missing():
    for lang in ("ar", "en"):
        r = agents.run_agent("investment_advisor", budget_egp=10_000_000, lang=lang)
        txt = " ".join(r["assumptions"]).lower()
        assert ("إيجار" in txt and "تضخم" in txt) if lang == "ar" else ("rent" in txt and "inflation" in txt)
        miss = " ".join(r["missing_data"]).lower()
        assert not any(k in miss for k in (("إيجار", "التضخم", "عمق سوق") if lang == "ar" else ("rent", "inflation", "exit market depth")))
    ra = agents.run_agent("investment_advisor", budget_egp=10_000_000)["results"]["return_assumptions"]
    assert ra["inflation_pct"] == agents.INFLATION_PCT and ra["gross_rental_yield_pct_by_type"]["Apartment"] == 6.5


def test_bad_assumption_values_are_clamped_not_crashing():
    r = agents.run_agent("investment_advisor", budget_egp=10_000_000, vacancy_pct=500, rental_yield_pct=-3, inflation_pct=9999, service_charge_pct=99)
    assert r["results"]["return_assumptions"]["vacancy_pct"] == 100 and r["results"]["return_assumptions"]["inflation_pct"] == 100
    json.dumps(r, allow_nan=False)


def test_developer_projects_show_planned_and_forecast_completion():
    r = agents.run_agent("developer_performance", developer_id="DEV5")["results"]["projects"]
    assert all({"planned_completion_month", "forecast_completion_month", "forecast_delay_months"} <= set(x) for x in r)


def test_smart_city_expected_demand_is_derived_and_guarded():
    d = pd.DataFrame(dashboard.smart_city()["districts"]).set_index("District_ID")
    assert (d.expected_residents_buildout >= d.current_residents).all()
    assert d.loc[["D07", "D08"], "projected_utilization_pct"].isna().all()           # not populated yet -> no ratio on a floor value
    ok = d.projected_utilization_pct.notna()
    assert np.allclose(d[ok].projected_utilization_pct, d[ok].Service_Capacity_Utilization_Pct * d[ok].expected_residents_buildout / d[ok].current_residents)
    assert (d[ok].capacity_gap_pct >= 0).all() and d.loc[d.capacity_stress_buildout == True, "capacity_gap_pct"].gt(0).all()  # noqa: E712
    assert dashboard.smart_city()["missing"] and "Expected demand" not in json.dumps(dashboard.smart_city()["missing"])
