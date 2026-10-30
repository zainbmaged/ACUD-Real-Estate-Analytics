"""Tests for app.analytics.service_gap. Fixtures are tiny hand-built TEST data, not survey evidence."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.analytics import service_gap as sg  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]


def make_survey(spec: dict[str, list[int | None]], status="observed_test", transport_imp=5):
    """spec: district -> list of transport q9 ratings (None = unanswered). Other services get neutral 3s."""
    rows, i = [], 0
    for d, ratings in spec.items():
        for r in ratings:
            i += 1
            row = {"response_id": f"T{i:04d}", "is_eligible": True, "district_id": d, "project_id": f"P{d[-1]}", "data_status": status,
                   "q11_problem_changed_decision": "لا"}
            for k in sg.SERVICES:
                row[f"q8_importance_{k}_1_5"] = 4
                row[f"q9_current_level_{k}_1_5"] = 3
            row["q8_importance_transport_1_5"] = transport_imp
            row["q9_current_level_transport_1_5"] = r
            rows.append(row)
    return pd.DataFrame(rows)


def dims(ds):
    districts = pd.DataFrame({"District_ID": ds, "District_Name": ds, "Lifecycle_Archetype": ["x"] * len(ds)})
    projects = pd.DataFrame({"Project_ID": [f"P{d[-1]}" for d in ds], "District_ID": ds, "Price_Multiplier": 1.0, "Units_Planned": 100})
    return districts, projects


# ---- statistics helpers
def test_wilson_known_value():
    lo, hi = sg.wilson_ci(50, 100)
    assert lo == pytest.approx(0.4038, abs=1e-3) and hi == pytest.approx(0.5962, abs=1e-3)
    assert sg.wilson_ci(0, 0) == (None, None)
    lo, hi = sg.wilson_ci(0, 20)
    assert lo == 0.0 and 0 < hi < 0.2


def test_benjamini_hochberg_matches_reference():
    p = [0.01, 0.04, 0.03, 0.005, None]
    q = sg.benjamini_hochberg(p)
    assert q[4] is None
    assert q[3] == pytest.approx(0.02) and q[0] == pytest.approx(0.02)
    assert q[2] == pytest.approx(0.04) and q[1] == pytest.approx(0.04)
    assert all(qq >= pp for qq, pp in zip(q[:4], p[:4]))


# ---- validation / missing data
def test_validation_excludes_and_counts_bad_rows():
    s = make_survey({"D1": [1, 2, 3, 4], "D2": [3, 3]})
    s = pd.concat([s, s.iloc[[0]]])                              # duplicate response_id
    s.loc[s.index[1], "q9_current_level_transport_1_5"] = 9      # out of range
    s.loc[s.index[2], "project_id"] = "P2"                       # project belongs to D2, row says D1
    s.loc[s.index[3], "is_eligible"] = False
    d, p = dims(["D1", "D2"])
    clean, rep = sg.validate_survey(s, d, p)
    assert rep["excluded"]["duplicate_response_id"] == 1
    assert rep["excluded"]["project_district_mismatch"] == 1
    assert rep["ineligible_screened_out"] == 1
    assert rep["invalid_rating_values_set_to_null"] == 1
    assert clean["q9_current_level_transport_1_5"].isna().sum() == 1  # invalid -> null, not imputed


def test_missing_required_columns_gives_insufficient_data():
    d, p = dims(["D1"])
    out = sg.analyze(pd.DataFrame({"foo": [1]}), d, p)
    assert out["analysis_status"] == "insufficient_data" and out["recommended_actions"] == []


def test_nulls_excluded_from_denominator_not_counted_as_dissatisfied():
    s = make_survey({"D1": [1, 1, None, None, 5, 5]})
    d, p = dims(["D1"])
    clean, _ = sg.validate_survey(s, d, p)
    m = sg._cell_metrics(sg._service_cells(clean, "transport"))
    assert m["n_valid"] == 4 and m["n_null_excluded"] == 2
    assert m["bottom_box"] == 0.5 and m["top_box"] == 0.5


# ---- metric correctness
def test_gap_and_important_unmet():
    s = make_survey({"D1": [1, 2, 3, 4, 5]}, transport_imp=5)
    d, p = dims(["D1"])
    clean, _ = sg.validate_survey(s, d, p)
    m = sg._cell_metrics(sg._service_cells(clean, "transport"))
    assert m["gap_mean"] == 2.0                      # mean(5 - [1..5]) = 2
    assert m["important_unmet"] == 0.4               # ratings 1,2 with importance 5
    assert m["distribution"] == {"1": 1, "2": 1, "3": 1, "4": 1, "5": 1}


def test_area_vs_rest_and_denominators_do_not_leak_between_areas():
    s = make_survey({"D1": [1] * 40, "D2": [4] * 40, "D3": [4] * 40})
    d, p = dims(["D1", "D2", "D3"])
    clean, _ = sg.validate_survey(s, d, p)
    rows, _ = sg.build_area_service_table(clean, None)
    r = next(x for x in rows if x["district_id"] == "D1" and x["service"] == "transport")
    assert r["bottom_box"] == 1.0 and r["rest_bottom_box"] == 0.0 and r["diff_vs_rest"] == 1.0
    assert r["significant_vs_rest"] and r["scope"] == "area_specific"
    assert r["independent"]["status"] == "no_independent_data"
    assert r["evidence_grade"] == "consistent_but_limited"     # cannot be 'strong' without independent data
    r2 = next(x for x in rows if x["district_id"] == "D2" and x["service"] == "transport")
    assert r2["evidence_grade"] == "no_concern_detected"


def test_small_sample_is_insufficient_or_preliminary_not_confident():
    s = make_survey({"D1": [1] * 10, "D2": [1] * 20, "D3": [4] * 40})
    d, p = dims(["D1", "D2", "D3"])
    clean, _ = sg.validate_survey(s, d, p)
    rows, _ = sg.build_area_service_table(clean, None)
    g = {x["district_id"]: x["evidence_grade"] for x in rows if x["service"] == "transport"}
    assert g["D1"] == "insufficient_data" and g["D2"] == "preliminary_association"


def test_independent_indicator_can_corroborate_or_contradict():
    s = make_survey({"D1": [1] * 40, "D2": [1] * 40, "D3": [4] * 40, "D4": [4] * 40})
    d, p = dims(["D1", "D2", "D3", "D4"])
    clean, _ = sg.validate_survey(s, d, p)
    ind = pd.DataFrame({"District_ID": ["D1", "D2", "D3", "D4"], "Month": "2025-12-01", "Public_Transport_Coverage_Pct": [20, 95, 60, 70]})
    rows, _ = sg.build_area_service_table(clean, ind)
    g = {x["district_id"]: x for x in rows if x["service"] == "transport"}
    assert g["D1"]["independent"]["status"] == "documented_relatively_low"
    assert g["D1"]["evidence_grade"] == "strong_descriptive"
    assert g["D2"]["independent"]["status"] == "relatively_high"
    assert g["D2"]["evidence_grade"] == "weak_or_contradictory"       # survey vs indicator disagree


# ---- priority sensitivity
def test_priority_has_no_weights_and_reports_sensitivity():
    s = make_survey({"D1": [1] * 40 + [4] * 20, "D2": [2] * 40 + [4] * 20})
    d, p = dims(["D1", "D2"])
    clean, _ = sg.validate_survey(s, d, p)
    _, port = sg.build_area_service_table(clean, None)
    pr = sg.service_priority(port)
    assert pr["ranking"][0]["service"] == "transport" and pr["ranking"][0]["tier"] == "A_priority"
    assert "No composite weights" in pr["rule"] and "spearman_rank_bottom_box_vs_gap" in pr["sensitivity"]


# ---- commercial association guard rails
def test_commercial_association_needs_enough_districts():
    rows = [{"district_id": f"D{i}", "bottom_box": 0.2} for i in range(1, 4)]
    ctx = pd.DataFrame({"District_ID": ["D1", "D2", "D3"], "sell_through_pct": [10.0, 20.0, 30.0]})
    out = sg.commercial_association(rows, ctx)
    assert out["status"] == "insufficient_data" and out["target_data_available"] is False


def test_commercial_association_no_causal_claim_and_detects_nothing_in_noise():
    rng = np.random.default_rng(1)
    ids = [f"D{i}" for i in range(1, 11)]
    rows = [{"district_id": d, "bottom_box": float(x)} for d, x in zip(ids, rng.uniform(0.1, 0.4, 10))]
    ctx = pd.DataFrame({"District_ID": ids, "sell_through_pct": rng.uniform(40, 99, 10)})
    out = sg.commercial_association(rows, ctx)
    assert out["status"] == "preliminary_association_only"
    assert out["primary"]["ci95"] is None and "Not causal" in out["interpretation"]
    assert "no meaningful association" in out["primary"]["direction"]


# ---- end to end on the repository CSVs
@pytest.fixture(scope="module")
def result():
    sys.path.insert(0, str(ROOT / "backend"))
    import run_service_gap_analysis as run
    return sg.analyze(**run.load(ROOT / "data"))


def test_repo_data_is_flagged_as_synthetic_and_never_valid(result):
    assert result["data_mode"] == "synthetic_demonstration" and result["analysis_status"] == "demonstration_only"
    assert any("SYNTHETIC" in x for x in result["limitations"])
    assert all(r["illustrative_only"] for r in result["recommended_actions"])


def test_recommendations_trace_to_findings_and_use_no_infrastructure_action(result):
    ids = {f["finding_id"] for f in result["service_findings"]}
    for r in result["recommended_actions"]:
        assert set(r["supports_finding_ids"]) <= ids
        assert r["recommended_action"] in {"investigate_service_deficiency", "investigate_cause_before_intervention", "investigate_accessibility_or_quality",
                                             "review_operational_quality", "collect_additional_resident_feedback", "request_commercial_investigation"}
        assert r["alternative_explanations"] and r["evidence_limitations"] and r["follow_up_measurement"]
    infra_unknown = [r for r in result["recommended_actions"] if r["recommended_action"] == "investigate_cause_before_intervention"]
    assert all(r["missing_data"] for r in infra_unknown)


def test_repo_counts_reconcile(result):
    dq = result["data_quality"]
    assert dq["rows_received"] == 1000 and dq["eligible_clean_rows"] == 945 and dq["ineligible_screened_out"] == 55
    assert len(result["service_findings"]) == 100
    for p in result["portfolio_service_metrics"]:
        assert sum(p["distribution"].values()) == p["n_valid"] == 945


def test_area_filter_narrows_findings_but_keeps_full_statistics():
    sys.path.insert(0, str(ROOT / "backend"))
    import run_service_gap_analysis as run
    full = sg.analyze(**run.load(ROOT / "data"))
    one = sg.analyze(**run.load(ROOT / "data"), area_id="d03")
    assert {f["district_id"] for f in one["service_findings"]} == {"D03"}
    a = next(f for f in full["service_findings"] if f["finding_id"] == "F-D03-transport")
    b = next(f for f in one["service_findings"] if f["finding_id"] == "F-D03-transport")
    assert a["bottom_box"] == b["bottom_box"] and a["q_bh"] == b["q_bh"]
    assert all("D03" in r["affected_areas"] for r in one["recommended_actions"])


# ---- regression: constant input must never yield a "significant" permutation p-value
def test_constant_variable_gives_undefined_correlation_not_false_significance():
    rows = [{"district_id": f"D{i}", "bottom_box": 0.2} for i in range(1, 11)]          # no variation at all
    ctx = pd.DataFrame({"District_ID": [f"D{i}" for i in range(1, 11)], "sell_through_pct": list(range(10, 110, 10))})
    out = sg.commercial_association(rows, ctx)
    assert out["primary"]["rho"] is None and out["primary"]["p_permutation"] is None
    assert "not computable" in out["primary"]["direction"]
