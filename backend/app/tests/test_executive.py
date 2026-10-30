"""Tests for app.analytics.executive. Hand-built TEST fixtures; verified values are worked out by hand in comments."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.analytics import executive as ex  # noqa: E402
from app.analytics import service_gap as sg  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]


def build(dissat: dict[str, int], n_per=20, sell=None, extra=None):
    """dissat: district -> number of 'transport' ratings of 1 (rest are 4). Other services neutral (3)."""
    rows = []
    for d, k in dissat.items():
        for i in range(n_per):
            r = {"response_id": f"{d}-{i}", "is_eligible": True, "district_id": d, "project_id": f"P{d[1:]}", "data_status": "test_observed",
                 "q4_recent_service_problems": "المواصلات" if i < k else "لم أواجه مشكلة",
                 "q5_most_impactful_problem": "المواصلات" if i < k else "لا توجد مشكلة مؤثرة",
                 "q6_problem_frequency": "كثيرًا" if i < k else "لم تؤثر", "q7_problem_severity": ("تأثير كبير" if i < k // 2 else "تأثير بسيط") if i < k else "لا توجد مشكلة",
                 "q11_problem_changed_decision": "لا", "q13_area_attractiveness_1_5": 3, "q14_recommend_area": "غالبًا نعم"}
            for key in sg.SERVICES:
                r[f"q8_importance_{key}_1_5"], r[f"q9_current_level_{key}_1_5"] = 4, 3
            r["q9_current_level_transport_1_5"] = 1 if i < k else 4
            rows.append(r)
    ds = list(dissat)
    districts = pd.DataFrame({"District_ID": ds, "District_Name": ds, "Lifecycle_Archetype": ["a", "a", "a", "b", "b", "b"][: len(ds)]})
    projects = pd.DataFrame({"Project_ID": [f"P{d[1:]}" for d in ds], "District_ID": ds, "Price_Multiplier": 1.0, "Units_Planned": 100})
    status = None if sell is None else pd.DataFrame({"District_ID": ds, "Month": "2025-12-01", "Sell_Through_Pct": [sell[d] for d in ds], "Units_Total_District": 100})
    return pd.DataFrame(rows), districts, projects, status


# ---- exact test, hand-calculated
def test_exact_mwu_hand_calculated():
    u, p, m = ex.exact_mwu(np.array([.5, .4, .6]), np.array([.1, .2, .15]))
    assert u == 9.0 and m == "exact_permutation"       # every weaker value beats every stronger value: U = 3*3
    assert p == pytest.approx(2 / 20)                  # two most extreme of C(6,3)=20 assignments -> 0.1
    _, p0, _ = ex.exact_mwu(np.zeros(5), np.zeros(5))
    assert p0 == 1.0                                   # all ties: no evidence


# ---- group comparison on a case with a known answer
def test_group_comparison_known_direction_and_unit_is_district():
    d = {"D1": 10, "D2": 8, "D3": 12, "D4": 2, "D5": 4, "D6": 3}        # bottom-box .50 .40 .60 | .10 .20 .15
    s, dist, proj, st = build(d, sell={"D1": 10, "D2": 20, "D3": 30, "D4": 80, "D5": 90, "D6": 95})
    clean, _ = sg.validate_survey(s, dist, proj)
    pos = {"D1": "weaker", "D2": "weaker", "D3": "weaker", "D4": "stronger", "D5": "stronger", "D6": "stronger"}
    g = ex.group_comparison(clean, pos)
    t = next(x for x in g["services"] if x["service"] == "transport")
    assert t["weaker_mean"] == 0.5 and t["stronger_mean"] == 0.15 and t["diff_pp"] == 35.0
    assert t["rank_biserial"] == 1.0 and t["p_exact"] == 0.1 and g["min_possible_p"] == 0.1
    assert t["supported"] is False                    # p=0.1 can never reach 0.05 with 3 v 3 districts: no overclaiming
    assert t["weaker_pooled"] == ex._share(30, 60) | {}  # pooled respondents shown descriptively: 10+8+12 of 60
    # a service with no variation is not supported
    s2 = next(x for x in g["services"] if x["service"] == "security")
    assert s2["diff_pp"] == 0.0 and s2["supported"] is False


def test_group_comparison_refuses_tiny_groups():
    s, dist, proj, _ = build({"D1": 5, "D2": 5, "D3": 5})
    clean, _ = sg.validate_survey(s, dist, proj)
    assert ex.group_comparison(clean, {"D1": "weaker", "D2": "stronger", "D3": "stronger"})["status"] == "insufficient_data"


def test_sales_positions_need_enough_districts_and_use_median():
    ctx = pd.DataFrame({"District_ID": [f"D{i}" for i in range(1, 7)], "sell_through_pct": [10, 20, 30, 80, 90, 95.0]})
    pos, med = ex.sales_positions(ctx)
    assert med == 55.0 and [pos[f"D{i}"] for i in range(1, 7)] == ["weaker"] * 3 + ["stronger"] * 3
    assert ex.sales_positions(ctx.head(4)) == ({}, None)


# ---- pain points: independent recomputation
def test_pain_points_denominators_and_separate_measures():
    s, dist, proj, _ = build({"D1": 10, "D2": 0}, n_per=20)
    clean, _ = sg.validate_survey(s, dist, proj)
    t = next(p for p in ex.pain_points(clean) if p["service"] == "transport")
    assert t["prevalence_q4"]["k"] == 10 and t["prevalence_q4"]["n"] == 40 and t["prevalence_q4"]["share"] == 0.25
    assert t["main_problem_q5"]["k"] == 10 and t["main_problem_q5"]["n"] == 40
    assert t["severe_among_main"]["k"] == 5 and t["severe_among_main"]["n"] == 10      # i < k//2 -> 5 severe of 10
    assert t["frequent_among_main"]["share"] == 1.0
    other = next(p for p in ex.pain_points(clean) if p["service"] == "healthcare")
    assert other["prevalence_q4"]["k"] == 0 and other["severe_among_main"]["n"] == 0 and other["severe_among_main"]["share"] is None


def test_satisfaction_missing_and_na_not_confused():
    s, dist, proj, _ = build({"D1": 10}, n_per=20)
    s.loc[s.index[:4], "q9_current_level_transport_1_5"] = None            # 4 unanswered
    clean, _ = sg.validate_survey(s, dist, proj)
    t = next(x for x in ex.satisfaction_by_service(clean) if x["service"] == "transport")
    assert t["n_valid"] == 16 and t["missing_rate"] == 0.2 and t["not_applicable_rate"] is None
    assert t["bottom_box"] == pytest.approx(6 / 16, abs=1e-3)               # rows 4..9 are rating 1


# ---- classification: no force-fitting, sensitivity reported
def test_classification_is_conservative_without_support():
    s, dist, proj, st = build({"D1": 2, "D2": 2, "D3": 2, "D4": 2, "D5": 2, "D6": 2}, sell={f"D{i}": 50 + i for i in range(1, 7)})
    out = ex.executive_brief(s, dist, proj, status=st)
    assert all(c["class"] in ("monitor",) for c in out["service_priority"])
    assert out["hypothesis_conclusion"]["label"] in ("no_credible_evidence", "weak_or_mixed", "insufficient_data")
    assert out["hypothesis_conclusion"]["causal_claim"] is False


def test_filters_change_denominators_and_empty_filter_is_handled():
    s, dist, proj, st = build({"D1": 10, "D2": 0, "D3": 0, "D4": 0, "D5": 0, "D6": 0}, sell={f"D{i}": 10 * i for i in range(1, 7)})
    full = ex.executive_brief(s, dist, proj, status=st)
    one = ex.executive_brief(s, dist, proj, status=st, district_id="d1")
    weak = ex.executive_brief(s, dist, proj, status=st, sales_position="weaker")
    assert full["respondents_in_filter"] == 120 and one["respondents_in_filter"] == 20 and weak["respondents_in_filter"] == 60
    t = lambda o: next(x for x in o["satisfaction_by_service"] if x["service"] == "transport")
    assert t(full)["n_valid"] == 120 and t(one)["n_valid"] == 20 and t(one)["bottom_box"] == 0.5
    # group tests are not filter dependent
    assert one["group_comparison"]["services"] == full["group_comparison"]["services"]
    empty = ex.executive_brief(s, dist, proj, status=st, district_id="D9")
    assert empty["respondents_in_filter"] == 0 and empty["insights"][0]["text"].startswith("No respondents")
    again = ex.executive_brief(s, dist, proj, status=st)
    assert again["respondents_in_filter"] == 120                             # clearing filters restores the baseline


# ---- repository data: independent recomputation straight from the CSV
@pytest.fixture(scope="module")
def repo():
    sys.path.insert(0, str(ROOT / "backend"))
    import run_service_gap_analysis as run
    data = run.load(ROOT / "data")
    return data, ex.executive_brief(**data)


def test_repo_kpis_match_independent_pandas(repo):
    data, out = repo
    s = data["survey"]
    e = s[s["is_eligible"].astype(str).str.lower() == "true"]
    for row in out["satisfaction_by_service"]:
        col = f"q9_current_level_{row['service']}_1_5"
        v = pd.to_numeric(e[col], errors="coerce").dropna()
        assert row["n_valid"] == len(v) and row["bottom_box"] == pytest.approx((v <= 2).mean(), abs=5e-4) and row["top_box"] == pytest.approx((v >= 4).mean(), abs=5e-4)
    q4 = e["q4_recent_service_problems"].dropna()
    transport = next(p for p in out["pain_points"] if p["service"] == "transport")
    assert transport["prevalence_q4"]["k"] == int(q4.str.contains("المواصلات").sum()) and transport["prevalence_q4"]["n"] == len(q4)
    sub = e[e["q5_most_impactful_problem"] == "المواصلات"]
    assert transport["severe_among_main"]["k"] == int(sub["q7_problem_severity"].isin(ex.SEVERE).sum())
    assert out["respondents_in_filter"] == len(e) == 945


def test_repo_conclusions_are_labelled_synthetic_and_non_causal(repo):
    _, out = repo
    assert out["analysis_status"] == "demonstration_only" and out["data_mode"] == "synthetic_demonstration"
    assert out["sales_position"]["official_classification_found"] is False
    assert out["hypothesis_conclusion"]["causal_claim"] is False
    assert any("synthetic" in i["text"].lower() for i in out["insights"])
    assert all(a["illustrative_only"] for a in out["recommended_actions"])
    assert {"observation", "interpretation", "decision"} <= {i["level"] for i in out["insights"]}


def test_repo_recommendations_trace_and_priority_classes_valid(repo):
    _, out = repo
    ids = {f["finding_id"] for f in out["area_service_matrix"]}
    for a in out["recommended_actions"]:
        assert set(a["evidence"]["supports_finding_ids"]) <= ids
        assert a["priority_class"] in {"high_priority", "medium_priority", "further_investigation", "monitor", "outside_service_scope"}
    assert set(out["priority_sensitivity"]["unstable_services"]) <= {c["service"] for c in out["service_priority"]}


def test_committed_outputs_are_not_stale():
    """outputs/*.json must equal a fresh run on the repository data (regenerate with run_service_gap_analysis.py)."""
    import json
    sys.path.insert(0, str(ROOT / "backend"))
    import run_service_gap_analysis as run
    data = run.load(ROOT / "data")
    fresh = json.loads(json.dumps(ex.executive_brief(**data), ensure_ascii=False, default=str))
    committed = json.loads((ROOT / "outputs" / "executive_brief.json").read_text(encoding="utf-8"))
    assert committed == fresh
    fresh_gap = json.loads(json.dumps(sg.analyze(**data), ensure_ascii=False, default=str))
    assert json.loads((ROOT / "outputs" / "service_gap_analysis.json").read_text(encoding="utf-8")) == fresh_gap
