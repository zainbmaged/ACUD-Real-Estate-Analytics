"""API-level checks against a temporary SQLite database seeded from the repository CSVs (PostgreSQL is not used here)."""
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

BACKEND = Path(__file__).resolve().parents[1]
ROOT = BACKEND.parent


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    db = tmp_path_factory.mktemp("db") / "acud_test.db"
    env = {**os.environ, "DATABASE_URL": f"sqlite:///{db}"}
    for script in ("seed_real_estate.py", "seed_survey_responses.py"):
        subprocess.run([sys.executable, script], cwd=BACKEND, env=env, check=True, capture_output=True)
    os.environ["DATABASE_URL"] = env["DATABASE_URL"]
    sys.path.insert(0, str(BACKEND))
    from fastapi.testclient import TestClient
    from app.main import app
    return TestClient(app)


@pytest.fixture(scope="module")
def eligible():
    s = pd.read_csv(ROOT / "data" / "synthetic_survey_responses_ar.csv", encoding="utf-8-sig")
    return s[s["is_eligible"].astype(str).str.lower() == "true"]


BASE = "/api/v1/survey-sales"


def test_legacy_dashboard_matches_independent_recomputation(client, eligible):
    d = client.get(f"{BASE}/dashboard").json()
    assert d["survey"]["responses"] == len(eligible) == 945
    t = next(g for g in d["survey"]["service_gaps"] if g["service"] == "المواصلات")
    imp, cur = eligible["q8_importance_transport_1_5"], eligible["q9_current_level_transport_1_5"]
    assert t["importance"] == round(imp.mean(), 2) and t["current_level"] == round(cur.mean(), 2) and t["gap"] == round((imp - cur).mean(), 2)
    assert d["survey"]["attractiveness"] == round(eligible["q13_area_attractiveness_1_5"].mean(), 2)


def test_legacy_filters_and_complaint_share_regression(client, eligible):
    d3 = client.get(f"{BASE}/dashboard", params={"district_id": "D03"}).json()
    sub = eligible[eligible["district_id"] == "D03"]
    assert d3["survey"]["responses"] == len(sub)
    row = d3["linked_comparison"]["districts"][0]
    top = sub["q5_most_impactful_problem"].value_counts()
    assert row["top_complaint_count"] == int(top.iloc[0]) and row["top_complaint_share_pct"] == round(top.iloc[0] / top.sum() * 100, 1)   # share beside the count: counts alone are not comparable across areas
    both = client.get(f"{BASE}/dashboard", params={"district_id": "D03", "project_id": "P001"}).json()   # P001 belongs to D01
    assert both["survey"]["responses"] == 0 and both["survey"]["service_gaps"] == []
    assert client.get(f"{BASE}/dashboard").json()["survey"]["responses"] == 945                                   # clearing filters restores baseline


def test_executive_endpoint_matches_library_and_recomputation(client, eligible):
    r = client.get(f"{BASE}/executive").json()
    assert r["analysis_status"] == "demonstration_only" and r["respondents_in_filter"] == 945
    tr = next(x for x in r["satisfaction_by_service"] if x["service"] == "transport")
    v = eligible["q9_current_level_transport_1_5"]
    assert tr["n_valid"] == int(v.notna().sum()) and tr["bottom_box"] == pytest.approx((v <= 2).mean(), abs=5e-4)
    sg = client.get(f"{BASE}/service-gap").json()
    assert len(sg["service_findings"]) == 100
    assert sg["commercial_context"]["primary"]["rho"] == r["commercial_context"]["primary"]["rho"]


def test_executive_filters(client, eligible):
    d3 = client.get(f"{BASE}/executive", params={"district_id": "D03"}).json()
    assert d3["respondents_in_filter"] == int((eligible["district_id"] == "D03").sum())
    assert all(a["areas"] == ["D03"] for a in d3["recommended_actions"] if a["areas"])
    weak = client.get(f"{BASE}/executive", params={"sales_position": "weaker"}).json()
    assert weak["districts_in_filter"] == weak["group_comparison"]["weaker_districts"]
    assert weak["respondents_in_filter"] == int(eligible["district_id"].isin(weak["districts_in_filter"]).sum())
    assert d3["group_comparison"]["services"] == weak["group_comparison"]["services"]          # group tests ignore filters
    none = client.get(f"{BASE}/executive", params={"district_id": "D99"}).json()
    assert none["respondents_in_filter"] == 0
    assert client.get(f"{BASE}/executive", params={"sales_position": "bogus"}).status_code == 422
