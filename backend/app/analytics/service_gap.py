"""Resident Survey & Service Gap analytics (pure pandas/scipy; no database or web imports).

Pipeline:  validated survey -> metrics -> findings -> evidence grade -> recommendations.

Metric definitions (all documented again in `METRIC_DEFINITIONS`, which is returned in the output):

* Scale: q9 "current level" and q8 "importance" are 1-5 ordinal, higher = better / more important.
* bottom-box  = share of valid answers with q9 in {1, 2}  (dissatisfied)
* top-box     = share of valid answers with q9 in {4, 5}
* gap_mean    = mean(q8 - q9) over respondents answering both (paired)
* important_unmet = share with q8 >= 4 AND q9 <= 2
* Denominator = eligible respondents with a valid (non-null, 1..5) answer for that service in that area.
  Null answers are excluded and counted; they are never treated as dissatisfaction or as zero.
* Proportion CI = Wilson 95 % interval.
* Area-vs-rest test = Fisher exact on bottom-box counts, Benjamini-Hochberg adjusted over all area x service tests.

There is NO sales target in the source data, so "target achievement" is not computed. The commercial
outcome used is latest cumulative sell-through % from Fact_District_Monthly_Status.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

MODULE = "resident_service_gap_analysis"
SCHEMA_VERSION = "1.0"

SERVICES: dict[str, dict[str, str]] = {
    "transport": {"en": "Transport", "ar": "المواصلات"},
    "healthcare": {"en": "Healthcare", "ar": "الخدمات الطبية"},
    "education": {"en": "Education", "ar": "التعليم"},
    "security": {"en": "Security & safety", "ar": "الأمن"},
    "shopping": {"en": "Shopping & daily needs", "ar": "التسوق"},
    "cleanliness": {"en": "Cleanliness & waste collection", "ar": "النظافة"},
    "roads_lighting": {"en": "Roads & lighting", "ar": "الطرق والإنارة"},
    "internet": {"en": "Internet & telecom", "ar": "الإنترنت"},
    "utilities": {"en": "Water & electricity", "ar": "المرافق"},
    "recreation": {"en": "Recreation & green space", "ar": "الترفيه"},
}

# Independent (non-survey) availability indicators, from Fact_District_Service_Indicators.
# Only services with a real counterpart are mapped; the rest are "no independent data", never "adequate".
INDEPENDENT_INDICATORS: dict[str, str] = {
    "transport": "Public_Transport_Coverage_Pct",
    "healthcare": "Healthcare_Access_Score",
    "education": "School_Access_Score",
    "shopping": "Retail_Access_Score",
    "utilities": "Utilities_Reliability_Pct",
}

# Policy thresholds. These are management-adjustable decision rules, not estimated quantities.
PARAMS: dict[str, Any] = {
    "bottom_box_threshold": 0.25,   # share dissatisfied above which a service is a concern
    "min_n_reportable": 30,         # minimum valid answers for a graded finding
    "min_n_preliminary": 15,        # below this: insufficient data
    "importance_high": 4.0,         # mean importance regarded as "important"
    "min_diff_vs_rest": 0.10,       # practical-significance floor (percentage points / 100)
    "alpha": 0.05,                  # FDR level (Benjamini-Hochberg)
    "indicator_low_quantile": 0.25, # relative-low cut for independent indicators across districts
    "indicator_high_quantile": 0.75,
}

METRIC_DEFINITIONS = [
    {"metric": "bottom_box", "formula": "count(q9 in {1,2}) / count(valid q9)", "source": "survey.q9_current_level_<service>_1_5",
     "level": "district x service", "note": "Ordinal scale: reported as a proportion, not a mean. Wilson 95% CI."},
    {"metric": "top_box", "formula": "count(q9 in {4,5}) / count(valid q9)", "source": "survey.q9_current_level_<service>_1_5", "level": "district x service", "note": "Neutral (3) is neither top nor bottom."},
    {"metric": "gap_mean", "formula": "mean(q8 - q9) over respondents with both answers", "source": "survey.q8_/q9_ columns", "level": "district x service",
     "note": "Importance-minus-performance. Not a target gap: no service target exists in the data. Treats ordinal scores as interval: use alongside bottom_box."},
    {"metric": "important_unmet", "formula": "count(q8>=4 and q9<=2) / count(both valid)", "source": "survey.q8_/q9_ columns", "level": "district x service", "note": "Dissatisfaction among people for whom the service matters."},
    {"metric": "diff_vs_rest", "formula": "bottom_box(area) - bottom_box(all other areas)", "source": "derived", "level": "district x service",
     "note": "Fisher exact test; Benjamini-Hochberg across all district x service tests; practical floor 10 pp."},
    {"metric": "omnibus_difference", "formula": "Kruskal-Wallis H across districts per service; effect size epsilon^2 = (H-k+1)/(n-k)", "source": "survey q9", "level": "service", "note": "Tests whether districts differ at all; BH across services."},
    {"metric": "independent_indicator", "formula": "mean of last 12 months of district indicator; 'relatively low' = bottom quartile of districts", "source": "Fact_District_Service_Indicators",
     "level": "district x service", "note": "Relative ranking only: no absolute adequacy standard exists in the data. Services without an indicator are 'no_independent_data'."},
    {"metric": "decision_loss_rate", "formula": "count(q11 postponed or chose other area) / count(q11 excluding not-applicable)", "source": "survey.q11_problem_changed_decision", "level": "district",
     "note": "Self-reported, not observed lost sales."},
    {"metric": "commercial_association", "formula": "Spearman rho (district mean bottom_box vs latest sell-through %), exact permutation p", "source": "survey + Fact_District_Monthly_Status",
     "level": "district (n = number of districts)", "note": "Association only. Survey respondents are not independent commercial observations; n is the number of districts."},
]

DATA_STATUS_SYNTHETIC = {"synthetic_linked_demo", "synthetic", "synthetic demonstration only", "synthetic scenario"}
RESPONSE_REQUIRED = {"response_id", "is_eligible", "district_id", "project_id"}
SCALE_RANGE = (1, 5)
DECISION_NA_PREFIX = "لا ينطبق"
DECISION_LOSS_VALUES = ("نعم، جعلتني أؤجل القرار", "نعم، جعلتني أختار منطقة أخرى")


# --------------------------------------------------------------------------- statistics helpers
def wilson_ci(successes: int, n: int, z: float = 1.959964) -> tuple[float | None, float | None]:
    if n <= 0:
        return None, None
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def benjamini_hochberg(pvalues: list[float | None]) -> list[float | None]:
    idx = [i for i, p in enumerate(pvalues) if p is not None]
    out: list[float | None] = [None] * len(pvalues)
    m = len(idx)
    if m == 0:
        return out
    order = sorted(idx, key=lambda i: pvalues[i])
    prev = 1.0
    for rank in range(m, 0, -1):
        i = order[rank - 1]
        prev = min(prev, pvalues[i] * m / rank)  # type: ignore[operator]
        out[i] = prev
    return out


def _r(x: float | None, nd: int = 3) -> float | None:
    return None if x is None or (isinstance(x, float) and not math.isfinite(x)) else round(float(x), nd)


# --------------------------------------------------------------------------- validation
def validate_survey(survey: pd.DataFrame, districts: pd.DataFrame, projects: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Return (clean eligible respondents, data-quality report). Invalid rows are excluded AND counted."""
    report: dict[str, Any] = {"rows_received": int(len(survey)), "issues": [], "excluded": {}}
    missing = sorted(RESPONSE_REQUIRED - set(survey.columns))
    if missing:
        report["issues"].append({"code": "missing_required_columns", "detail": missing})
        report["status"] = "invalid_schema"
        return survey.iloc[0:0].copy(), report

    df = survey.copy()
    for key in ("district_id", "project_id"):
        df[key] = df[key].astype(str).str.strip().str.upper()
    df["is_eligible"] = df["is_eligible"].astype(str).str.lower().isin(["true", "1", "yes"])

    dup = int(df["response_id"].duplicated().sum())
    if dup:
        df = df.drop_duplicates("response_id", keep="first")
        report["excluded"]["duplicate_response_id"] = dup
    report["ineligible_screened_out"] = int((~df["is_eligible"]).sum())
    df = df[df["is_eligible"]].copy()

    bad_district = ~df["district_id"].isin(set(districts["District_ID"].astype(str).str.upper()))
    if bad_district.any():
        report["excluded"]["unknown_district_id"] = int(bad_district.sum())
        df = df[~bad_district]
    proj_map = projects.assign(Project_ID=projects["Project_ID"].astype(str).str.upper(), District_ID=projects["District_ID"].astype(str).str.upper()).drop_duplicates("Project_ID").set_index("Project_ID")["District_ID"]
    expected = df["project_id"].map(proj_map)
    mismatch = expected.notna() & (expected != df["district_id"])
    unknown_project = expected.isna()
    if mismatch.any():
        report["excluded"]["project_district_mismatch"] = int(mismatch.sum())
    if unknown_project.any():
        report["excluded"]["unknown_project_id"] = int(unknown_project.sum())
    df = df[~mismatch & ~unknown_project]

    missing_services, invalid_values, null_ratings = [], 0, 0
    for key in SERVICES:
        for prefix in ("q8_importance_", "q9_current_level_"):
            col = f"{prefix}{key}_1_5"
            if col not in df:
                missing_services.append(col)
                continue
            num = pd.to_numeric(df[col], errors="coerce")
            bad = num.notna() & ((num < SCALE_RANGE[0]) | (num > SCALE_RANGE[1]) | (num != num.round()))
            invalid_values += int(bad.sum())
            null_ratings += int(num.isna().sum())
            df[col] = num.where(~bad)
    if missing_services:
        report["issues"].append({"code": "missing_rating_columns", "detail": missing_services})
    report["invalid_rating_values_set_to_null"] = invalid_values
    report["null_rating_cells_not_imputed"] = null_ratings
    report["eligible_clean_rows"] = int(len(df))
    status_col = df["data_status"].astype(str).str.lower() if "data_status" in df else pd.Series(dtype=str)
    report["data_status_values"] = sorted(status_col.unique().tolist())
    report["is_synthetic"] = bool(len(status_col) == 0 or status_col.isin(DATA_STATUS_SYNTHETIC).any())
    report["status"] = "ok" if not report["issues"] and len(df) else ("no_valid_rows" if not len(df) else "partial")
    return df, report


# --------------------------------------------------------------------------- core metrics
def _service_cells(df: pd.DataFrame, key: str) -> pd.DataFrame:
    imp = pd.to_numeric(df.get(f"q8_importance_{key}_1_5"), errors="coerce")
    cur = pd.to_numeric(df.get(f"q9_current_level_{key}_1_5"), errors="coerce")
    return pd.DataFrame({"district_id": df["district_id"], "imp": imp, "cur": cur})


def _cell_metrics(cells: pd.DataFrame) -> dict[str, Any]:
    cur = cells["cur"].dropna()
    n = int(len(cur))
    k = int((cur <= 2).sum())
    lo, hi = wilson_ci(k, n)
    both = cells.dropna(subset=["imp", "cur"])
    nb = int(len(both))
    gap = both["imp"] - both["cur"]
    gap_ci = (None, None)
    if nb > 1:
        se = float(gap.std(ddof=1)) / math.sqrt(nb)
        gap_ci = (float(gap.mean()) - 1.96 * se, float(gap.mean()) + 1.96 * se)
    unmet = int(((both["imp"] >= 4) & (both["cur"] <= 2)).sum())
    ulo, uhi = wilson_ci(unmet, nb)
    return {
        "n_valid": n, "n_null_excluded": int(len(cells) - n),
        "bottom_box": _r(k / n if n else None), "bottom_box_ci95": [_r(lo), _r(hi)], "bottom_box_count": k,
        "top_box": _r(float((cur >= 4).mean()) if n else None),
        "neutral": _r(float((cur == 3).mean()) if n else None),
        "importance_mean": _r(float(cells["imp"].dropna().mean()) if cells["imp"].notna().any() else None, 2),
        "current_mean": _r(float(cur.mean()) if n else None, 2), "current_median": _r(float(cur.median()) if n else None, 1),
        "gap_mean": _r(float(gap.mean()) if nb else None, 2), "gap_ci95": [_r(gap_ci[0], 2), _r(gap_ci[1], 2)],
        "important_unmet": _r(unmet / nb if nb else None), "important_unmet_ci95": [_r(ulo), _r(uhi)],
        "distribution": {str(v): int((cur == v).sum()) for v in range(1, 6)},
    }


def independent_indicator_context(indicators: pd.DataFrame | None, district_ids: list[str]) -> dict[str, dict[str, dict[str, Any]]]:
    """service -> district -> {value, status}. status in documented_relatively_low | mid | relatively_high | no_independent_data."""
    out: dict[str, dict[str, dict[str, Any]]] = {s: {} for s in SERVICES}
    if indicators is None or indicators.empty or "District_ID" not in indicators:
        for s in SERVICES:
            for d in district_ids:
                out[s][d] = {"indicator": None, "value": None, "status": "no_independent_data"}
        return out
    ind = indicators.copy()
    ind["District_ID"] = ind["District_ID"].astype(str).str.upper()
    if "Month" in ind:
        ind["Month"] = pd.to_datetime(ind["Month"], errors="coerce")
        ind = ind[ind["Month"] > ind["Month"].max() - pd.DateOffset(months=12)]
    for s in SERVICES:
        col = INDEPENDENT_INDICATORS.get(s)
        if not col or col not in ind:
            for d in district_ids:
                out[s][d] = {"indicator": None, "value": None, "status": "no_independent_data"}
            continue
        vals = pd.to_numeric(ind[col], errors="coerce").groupby(ind["District_ID"]).mean()
        lo = vals.quantile(PARAMS["indicator_low_quantile"])
        hi = vals.quantile(PARAMS["indicator_high_quantile"])
        for d in district_ids:
            v = vals.get(d)
            if v is None or not np.isfinite(v):
                out[s][d] = {"indicator": col, "value": None, "status": "no_independent_data"}
            else:
                status = "documented_relatively_low" if v <= lo else ("relatively_high" if v >= hi else "mid")
                out[s][d] = {"indicator": col, "value": _r(float(v), 2), "status": status}
    return out


def grade_evidence(m: dict[str, Any], diff: dict[str, Any] | None, corroboration: str, scope: str) -> tuple[str, list[str]]:
    """Return (grade, reasons). Criteria are explicit; see GRADE_CRITERIA."""
    n, thr = m["n_valid"], PARAMS["bottom_box_threshold"]
    lb = (m["bottom_box_ci95"] or [None])[0]
    pt = m["bottom_box"]
    reasons: list[str] = []
    if n < PARAMS["min_n_preliminary"] or pt is None:
        return "insufficient_data", [f"n={n} below {PARAMS['min_n_preliminary']}"]
    if pt < thr:
        return "no_concern_detected", [f"bottom-box {pt:.0%} below {thr:.0%}"]
    if corroboration == "relatively_high":
        return "weak_or_contradictory", ["survey dissatisfaction but independent indicator is relatively high (perception vs measured gap)"]
    if n < PARAMS["min_n_reportable"] or lb is None or lb < thr:
        reasons.append(f"n={n}" if n < PARAMS["min_n_reportable"] else f"CI lower bound {lb:.0%} < {thr:.0%}")
        return "preliminary_association", reasons
    distinct = bool(diff and diff.get("significant"))
    if corroboration == "documented_relatively_low" and (distinct or scope == "portfolio_wide"):
        return "strong_descriptive", ["CI above threshold", "independent indicator relatively low", "distinct from other districts" if distinct else "widespread across districts"]
    reasons.append("CI above threshold")
    reasons.append("no independent availability data" if corroboration == "no_independent_data" else "independent indicator not relatively low")
    return "consistent_but_limited", reasons


GRADE_CRITERIA = {
    "strong_descriptive": "n>=30, Wilson CI lower bound >= threshold, independent indicator relatively low, and (BH-significant >=10pp vs other districts OR portfolio-wide).",
    "consistent_but_limited": "n>=30 and CI lower bound >= threshold, but no independent availability data or indicator not relatively low.",
    "preliminary_association": "Point estimate >= threshold but n<30 or CI lower bound below threshold.",
    "weak_or_contradictory": "Survey dissatisfaction but independent indicator is relatively high.",
    "insufficient_data": "n<15 valid answers.",
    "no_concern_detected": "Bottom-box below threshold. Absence of a flag is not proof of adequacy.",
}


def build_area_service_table(df: pd.DataFrame, indicators: pd.DataFrame | None) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Area x service metrics with area-vs-rest tests; plus per-service portfolio metrics with omnibus tests."""
    district_ids = sorted(df["district_id"].unique())
    context = independent_indicator_context(indicators, district_ids)
    rows: list[dict[str, Any]] = []
    portfolio: list[dict[str, Any]] = []
    pvals: list[float | None] = []
    for key, names in SERVICES.items():
        cells = _service_cells(df, key)
        all_m = _cell_metrics(cells)
        groups = [g["cur"].dropna().to_numpy() for _, g in cells.groupby("district_id") if g["cur"].notna().sum() >= 2]
        omni: dict[str, Any] = {"test": "kruskal_wallis", "p": None, "epsilon_sq": None}
        if len(groups) >= 2 and sum(len(g) for g in groups) > len(groups) and len({v for g in groups for v in g}) > 1:
            h, p = stats.kruskal(*groups)
            n_tot, k = sum(len(g) for g in groups), len(groups)
            omni.update({"p": float(p), "H": _r(float(h)), "epsilon_sq": _r(max(0.0, (h - k + 1) / (n_tot - k)), 4), "k_districts": k})
        portfolio.append({"service": key, "label_en": names["en"], "label_ar": names["ar"], **all_m, "omnibus": omni})
        for did in district_ids:
            mine, rest = cells[cells["district_id"] == did], cells[cells["district_id"] != did]
            m = _cell_metrics(mine)
            rm = _cell_metrics(rest)
            a, n1 = m["bottom_box_count"], m["n_valid"]
            b, n2 = rm["bottom_box_count"], rm["n_valid"]
            p_f, orr = None, None
            if n1 > 0 and n2 > 0:
                orr_raw, p_f = stats.fisher_exact([[a, n1 - a], [b, n2 - b]])
                orr = None if not np.isfinite(orr_raw) else orr_raw
            diff = None if (m["bottom_box"] is None or rm["bottom_box"] is None) else m["bottom_box"] - rm["bottom_box"]
            pvals.append(None if p_f is None else float(p_f))
            rows.append({
                "district_id": did, "service": key, "label_en": names["en"], "label_ar": names["ar"], **m,
                "rest_bottom_box": rm["bottom_box"], "diff_vs_rest": _r(diff), "odds_ratio_vs_rest": _r(orr, 2),
                "p_fisher": _r(p_f, 5), "independent": context[key][did],
            })
    adj = benjamini_hochberg(pvals)
    for row, q in zip(rows, adj):
        row["q_bh"] = _r(q, 5)
    omni_q = benjamini_hochberg([p["omnibus"]["p"] for p in portfolio])
    for p, q in zip(portfolio, omni_q):
        p["omnibus"]["q_bh"] = _r(q, 5)
        p["omnibus"]["p"] = _r(p["omnibus"]["p"], 5)
        p["omnibus"]["districts_differ"] = bool(q is not None and q < PARAMS["alpha"])
    by_service = {p["service"]: p for p in portfolio}
    for row in rows:
        row["significant_vs_rest"] = bool(row["q_bh"] is not None and row["q_bh"] < PARAMS["alpha"] and (row["diff_vs_rest"] or 0) >= PARAMS["min_diff_vs_rest"])
        port_lb = by_service[row["service"]]["bottom_box_ci95"][0]
        # scope: portfolio_wide if the service is a concern across districts generally
        row["scope"] = "area_specific" if row["significant_vs_rest"] else ("portfolio_wide" if (port_lb is not None and port_lb >= PARAMS["bottom_box_threshold"]) else "not_flagged")
        corr = row["independent"]["status"]
        grade, why = grade_evidence(row, {"significant": row["significant_vs_rest"]}, corr, row["scope"])
        row["evidence_grade"], row["grade_reasons"] = grade, why
        row["finding_id"] = f"F-{row['district_id']}-{row['service']}"
    return rows, portfolio


def service_priority(portfolio: list[dict[str, Any]]) -> dict[str, Any]:
    """Transparent rule-based tiers with rank-sensitivity across three alternative gap definitions (no arbitrary weights)."""
    thr, imp_hi = PARAMS["bottom_box_threshold"], PARAMS["importance_high"]
    df = pd.DataFrame([{k: p[k] for k in ("service", "bottom_box", "gap_mean", "important_unmet", "importance_mean", "n_valid")} | {"lb": p["bottom_box_ci95"][0]} for p in portfolio])
    for col in ("bottom_box", "gap_mean", "important_unmet"):
        df[f"rank_{col}"] = df[col].rank(ascending=False, method="min")
    ranks = df[["rank_bottom_box", "rank_gap_mean", "rank_important_unmet"]]
    rho = stats.spearmanr(ranks["rank_bottom_box"], ranks["rank_gap_mean"])[0] if len(df) > 2 else None
    tops = [set(df.nsmallest(3, c)["service"]) for c in ("rank_bottom_box", "rank_gap_mean", "rank_important_unmet")]
    stable_top3 = set.intersection(*tops)
    out = []
    for _, r in df.iterrows():
        important = bool(r["importance_mean"] is not None and r["importance_mean"] >= imp_hi)
        if r["lb"] is not None and r["lb"] >= thr and important:
            tier = "A_priority"
        elif (r["bottom_box"] or 0) >= thr or ((r["gap_mean"] or 0) >= 1.0 and important):
            tier = "B_watch"
        else:
            tier = "C_no_flag"
        out.append({"service": r["service"], "tier": tier, "important": important, "bottom_box": r["bottom_box"], "bottom_box_ci_lower": _r(r["lb"]),
                    "gap_mean": r["gap_mean"], "important_unmet": r["important_unmet"], "importance_mean": r["importance_mean"], "n_valid": int(r["n_valid"]),
                    "ranks": {"bottom_box": int(r["rank_bottom_box"]), "gap_mean": int(r["rank_gap_mean"]), "important_unmet": int(r["rank_important_unmet"])}})
    order = {"A_priority": 0, "B_watch": 1, "C_no_flag": 2}
    out.sort(key=lambda x: (order[x["tier"]], x["ranks"]["bottom_box"]))
    return {"ranking": out, "rule": f"A: bottom-box CI lower bound >= {thr:.0%} and mean importance >= {imp_hi}; B: bottom-box point >= {thr:.0%} or (gap >= 1.0 and important); else C. No composite weights used.",
            "sensitivity": {"spearman_rank_bottom_box_vs_gap": _r(rho), "top3_in_all_three_definitions": sorted(stable_top3), "rank_stable": len(stable_top3) >= 2}}


# --------------------------------------------------------------------------- district context & commercial relevance
def district_context(df: pd.DataFrame, districts: pd.DataFrame, projects: pd.DataFrame, status: pd.DataFrame | None,
                     funnel: pd.DataFrame | None) -> pd.DataFrame:
    base = districts[["District_ID"] + [c for c in ("District_Name", "Lifecycle_Archetype") if c in districts]].copy()
    base["District_ID"] = base["District_ID"].astype(str).str.upper()
    base = base.set_index("District_ID")
    n_resp = df.groupby("district_id").size()
    base["respondents"] = n_resp.reindex(base.index).fillna(0).astype(int)
    q11 = df.get("q11_problem_changed_decision")
    if q11 is not None:
        applicable = df[~q11.astype(str).str.startswith(DECISION_NA_PREFIX) & q11.notna()]
        loss = applicable[applicable["q11_problem_changed_decision"].isin(DECISION_LOSS_VALUES)]
        base["decision_applicable_n"] = applicable.groupby("district_id").size().reindex(base.index).fillna(0).astype(int)
        base["decision_loss_n"] = loss.groupby("district_id").size().reindex(base.index).fillna(0).astype(int)
        base["decision_loss_rate"] = (base["decision_loss_n"] / base["decision_applicable_n"].replace(0, np.nan)).round(3)
    if status is not None and not status.empty and {"District_ID", "Month", "Sell_Through_Pct"} <= set(status.columns):
        s = status.copy()
        s["District_ID"] = s["District_ID"].astype(str).str.upper()
        s["Month"] = pd.to_datetime(s["Month"], errors="coerce")
        last = s[s["Month"] == s["Month"].max()].set_index("District_ID")
        base["sell_through_pct"] = pd.to_numeric(last["Sell_Through_Pct"], errors="coerce").reindex(base.index)
        base["sales_as_of"] = str(s["Month"].max().date())
        if "Units_Total_District" in last:
            units = pd.to_numeric(last["Units_Total_District"], errors="coerce").reindex(base.index)
            base["units_total"] = units
            base["respondents_per_100_units"] = (base["respondents"] / units * 100).round(2)
    # alternative explanations that exist in the data
    pm = projects.copy()
    pm["District_ID"] = pm["District_ID"].astype(str).str.upper()
    if {"Price_Multiplier", "Units_Planned"} <= set(pm.columns):
        w = pm.assign(w=pd.to_numeric(pm["Units_Planned"], errors="coerce"), pmv=pd.to_numeric(pm["Price_Multiplier"], errors="coerce"))
        base["price_multiplier_wavg"] = (w.assign(x=w["pmv"] * w["w"]).groupby("District_ID")["x"].sum() / w.groupby("District_ID")["w"].sum()).reindex(base.index).round(3)
    if funnel is not None and not funnel.empty and {"District_ID", "Month", "Leads", "Bookings"} <= set(funnel.columns):
        f = funnel.copy()
        f["District_ID"] = f["District_ID"].astype(str).str.upper()
        f["Month"] = pd.to_datetime(f["Month"], errors="coerce")
        f = f[f["Month"] > f["Month"].max() - pd.DateOffset(months=12)]
        g = f.groupby("District_ID")
        base["leads_12m"] = g["Leads"].sum().reindex(base.index)
        base["lead_to_booking_rate_12m"] = (g["Bookings"].sum() / g["Leads"].sum()).reindex(base.index).round(4)
        if "Average_Lead_Response_Hours" in f:
            base["avg_lead_response_hours_12m"] = g["Average_Lead_Response_Hours"].mean().reindex(base.index).round(2)
    return base.reset_index()


def commercial_association(area_rows: list[dict[str, Any]], ctx: pd.DataFrame, *, seed: int = 20260101) -> dict[str, Any]:
    """District-level association between survey dissatisfaction and sell-through. n = districts, not respondents."""
    out: dict[str, Any] = {"unit_of_analysis": "district", "outcome": "sell_through_pct (latest month, cumulative)", "target_data_available": False,
                           "note": "No sales target exists in the source data; target achievement cannot be computed."}
    if "sell_through_pct" not in ctx or ctx["sell_through_pct"].notna().sum() < 3:
        return {**out, "status": "insufficient_data", "missing_data": ["district sales outcome (sell-through) for >=3 districts"]}
    tbl = pd.DataFrame(area_rows)
    idx = tbl.groupby("district_id")["bottom_box"].mean().rename("dissatisfaction_index")
    d = ctx.set_index("District_ID").join(idx).dropna(subset=["dissatisfaction_index", "sell_through_pct"])
    n = len(d)
    if n < 6:
        return {**out, "status": "insufficient_data", "n_districts": n, "missing_data": ["at least 6 districts with survey and sales data"]}

    def rho_perm(x: pd.Series, y: pd.Series) -> dict[str, Any]:
        """Spearman rho with a two-sided permutation p-value (vectorised; rank-transform then Pearson)."""
        rx, ry = stats.rankdata(x.to_numpy()), stats.rankdata(y.to_numpy())
        if np.ptp(rx) == 0 or np.ptp(ry) == 0:  # a constant variable has no defined correlation: never report a p-value
            return {"rho": None, "p_permutation": None, "undefined_reason": "no variation in one variable"}
        rho = float(np.corrcoef(rx, ry)[0, 1])
        rng = np.random.default_rng(seed)
        perms = rng.permuted(np.tile(rx, (9999, 1)), axis=1)
        pc, yc = perms - perms.mean(1, keepdims=True), ry - ry.mean()
        null = (pc @ yc) / np.sqrt((pc ** 2).sum(1) * (yc ** 2).sum())
        p = (1 + int((np.abs(null) >= abs(rho) - 1e-12).sum())) / (9999 + 1)
        return {"rho": _r(rho), "p_permutation": _r(p, 4)}

    primary = rho_perm(d["dissatisfaction_index"], d["sell_through_pct"])
    alts = []
    for col, label in (("price_multiplier_wavg", "Price level (project price multiplier)"), ("lead_to_booking_rate_12m", "Lead-to-booking rate"),
                       ("avg_lead_response_hours_12m", "Lead response time (hours)"), ("leads_12m", "Lead volume")):
        if col in d and d[col].notna().sum() >= 6:
            alts.append({"factor": label, "column": col, **rho_perm(d[col], d["sell_through_pct"])})
    maturity = None
    if "Lifecycle_Archetype" in d:
        maturity = {k: _r(float(v), 1) for k, v in d.groupby("Lifecycle_Archetype")["sell_through_pct"].mean().items()}
    rho = primary["rho"]
    q = benjamini_hochberg([primary["p_permutation"]] + [x["p_permutation"] for x in alts])  # None p-values stay None
    primary["q_bh_over_5_correlations"] = _r(q[0], 4)
    for x, qq in zip(alts, q[1:]):
        x["q_bh_over_5_correlations"] = _r(qq, 4)
    return {**out, "status": "preliminary_association_only", "n_districts": int(n),
            "primary": {"x": "mean bottom-box across services (unweighted district dissatisfaction index)", "y": "sell_through_pct", **primary,
                        "direction": ("not computable (no variation in one variable)" if rho is None else "no meaningful association (|rho| < 0.3 or p >= 0.05)" if (abs(rho) < 0.3 or primary["p_permutation"] >= 0.05) else ("higher dissatisfaction with lower sell-through" if rho < 0 else "higher dissatisfaction with higher sell-through")),
                        "ci95": None, "ci_note": "No confidence interval: n districts is too small for a reliable interval; permutation p only."},
            "alternative_explanations_same_outcome": alts, "sell_through_by_lifecycle_archetype": maturity,
            "interpretation": "Association between area-level aggregates (n = districts). Not causal. Development maturity (lifecycle stage) is a likely confounder of sell-through and cannot be adjusted for with this few districts.",
            "excluded_factors_no_data": ["payment plans", "sales-team effectiveness", "customer affordability", "marketing reach beyond spend", "survey participation bias (response rate unknown)"]}


# --------------------------------------------------------------------------- recommendations
def _action_for(row: dict[str, Any]) -> tuple[str, str, str]:
    """(action, reason, follow-up) for one graded area x service finding, following the documented decision rules."""
    corr, grade = row["independent"]["status"], row["evidence_grade"]
    if grade == "weak_or_contradictory":
        return ("review_operational_quality", "Residents are dissatisfied although the independent availability indicator is relatively high: examine quality, reliability and communication before any capacity decision.",
                "Service-quality audit plus targeted resident follow-up; re-survey this service in 90 days.")
    if grade == "preliminary_association":
        return ("collect_additional_resident_feedback", "Signal is above the threshold but the sample is small or the interval is wide.",
                f"Raise valid answers to >= {PARAMS['min_n_reportable']} per district and service, then re-grade.")
    if corr == "documented_relatively_low":
        return ("investigate_service_deficiency", "High dissatisfaction coincides with a relatively low independent availability indicator.",
                "Diagnose frequency, coverage and reliability; define a service target; track bottom-box and the indicator.")
    if corr == "no_independent_data":
        return ("investigate_cause_before_intervention", "Dissatisfaction is high but service availability is unknown; no infrastructure recommendation is justified.",
                "Collect availability, capacity and accessibility data for this service (not collected today), then reassess.")
    return ("investigate_accessibility_or_quality", "Independent indicator is not relatively low, so the cause may be access, quality or reliability rather than absence.",
            "Investigate access and quality drivers; track bottom-box in a 90-day re-survey.")


def build_recommendations(area_rows: list[dict[str, Any]], ctx: pd.DataFrame, commercial: dict[str, Any], *, synthetic: bool) -> list[dict[str, Any]]:
    """One recommendation per (service, action), covering all affected districts, so managers see few, traceable items."""
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    meta: dict[tuple[str, str], tuple[str, str]] = {}
    for r in area_rows:
        if r["evidence_grade"] in ("no_concern_detected", "insufficient_data"):
            continue
        action, reason, follow = _action_for(r)
        groups.setdefault((r["service"], action), []).append(r)
        meta[(r["service"], action)] = (reason, follow)
    rank = {"investigate_service_deficiency": 0, "investigate_cause_before_intervention": 1, "investigate_accessibility_or_quality": 2, "review_operational_quality": 3, "collect_additional_resident_feedback": 4}
    recs: list[dict[str, Any]] = []
    limits = ["Ordinal self-reported ratings", "No sales target in data", "No prospective-buyer flag", "q9 has no 'don't know' option"] + (["Synthetic demonstration data"] if synthetic else [])
    for (service, action), rows in groups.items():
        k, n = sum(x["bottom_box_count"] for x in rows), sum(x["n_valid"] for x in rows)
        lo, hi = wilson_ci(k, n)
        reason, follow = meta[(service, action)]
        best = max(rows, key=lambda x: x["important_unmet"] or 0)
        recs.append({
            "recommendation_id": f"R-{service}-{action}", "supports_finding_ids": [x["finding_id"] for x in rows],
            "business_problem": f"{SERVICES[service]['en']} dissatisfaction may reduce the attractiveness of the affected areas.",
            "affected_areas": sorted(x["district_id"] for x in rows), "service": service,
            "observed_evidence": {"pooled_valid_n": n, "pooled_bottom_box": _r(k / n if n else None), "pooled_bottom_box_ci95": [_r(lo), _r(hi)],
                                  "pooled_note": "Pooled across listed districts; see per-finding rows for district detail.",
                                  "strongest_area": {"district_id": best["district_id"], "important_unmet": best["important_unmet"], "bottom_box": best["bottom_box"]}},
            "evidence_grade": min((x["evidence_grade"] for x in rows), key=lambda g: list(GRADE_CRITERIA).index(g)),
            "recommended_action": action, "reason": reason,
            "alternative_explanations": ["Development maturity / lifecycle stage", "Pricing and payment plans", "Lead quality and response time", "Survey participation bias"],
            "evidence_limitations": limits,
            "missing_data": ([f"independent availability indicator for {SERVICES[service]['en']}"] if action == "investigate_cause_before_intervention" else []),
            "follow_up_measurement": follow, "re_measure_after_days": 90,
            "priority_basis": "Ordered by action type, then by the largest share of respondents rating the service important (>=4) and poor (<=2).",
            "illustrative_only": synthetic})
    recs.sort(key=lambda r: (rank[r["recommended_action"]], -(r["observed_evidence"]["strongest_area"]["important_unmet"] or 0)))
    if "sell_through_pct" in ctx and ctx["sell_through_pct"].notna().any():
        med = ctx["sell_through_pct"].median()
        flagged = {x["district_id"] for r in area_rows for x in [r] if r["evidence_grade"] in ("strong_descriptive", "consistent_but_limited") and r["scope"] == "area_specific"}
        weak = ctx[(ctx["sell_through_pct"] < med) & ~ctx["District_ID"].isin(flagged)]
        if len(weak):
            recs.append({
                "recommendation_id": "R-commercial-review", "supports_finding_ids": [], "business_problem": "Districts with below-median sell-through that the survey does not distinguish from other districts on any service.",
                "affected_areas": sorted(weak["District_ID"].tolist()), "service": None,
                "observed_evidence": {"sell_through_pct": {d: _r(v, 1) for d, v in zip(weak["District_ID"], weak["sell_through_pct"])}, "median_sell_through_pct": _r(med, 1),
                                      "survey_commercial_association": commercial.get("primary")},
                "evidence_grade": "no_service_explanation_found", "recommended_action": "request_commercial_investigation",
                "reason": "Service ratings do not separate these districts from the rest. This does not implicate the sales team: ask the commercial module to examine lead volume, pricing, conversion and development maturity.",
                "alternative_explanations": ["Development maturity", "Pricing", "Lead volume/conversion"], "evidence_limitations": limits,
                "missing_data": ["sales targets", "payment-plan data"], "follow_up_measurement": "Commercial module: decomposition of sell-through by price, lead volume, conversion and stage.", "re_measure_after_days": 90,
                "priority_basis": "Below-median sell-through.", "illustrative_only": synthetic})
    return recs


# --------------------------------------------------------------------------- top level
def analyze(survey: pd.DataFrame, districts: pd.DataFrame, projects: pd.DataFrame, *, indicators: pd.DataFrame | None = None,
            status: pd.DataFrame | None = None, funnel: pd.DataFrame | None = None, area_id: str | None = None) -> dict[str, Any]:
    clean, dq = validate_survey(survey, districts, projects)
    base = {"module": MODULE, "schema_version": SCHEMA_VERSION, "area_id": area_id, "parameters": PARAMS, "metric_definitions": METRIC_DEFINITIONS,
            "grade_criteria": GRADE_CRITERIA, "data_quality": dq}
    if dq.get("status") in ("invalid_schema", "no_valid_rows"):
        return {**base, "analysis_status": "insufficient_data", "service_findings": [], "service_priority": None, "supporting_evidence": [], "commercial_context": None,
                "alternative_explanations": [], "recommended_actions": [], "missing_data": ["valid eligible survey responses"], "limitations": [], "data_provenance": []}
    synthetic = bool(dq["is_synthetic"])
    area_rows, portfolio = build_area_service_table(clean, indicators)
    priority = service_priority(portfolio)
    ctx = district_context(clean, districts, projects, status, funnel)
    commercial = commercial_association(area_rows, ctx)
    recs = build_recommendations(area_rows, ctx, commercial, synthetic=synthetic)
    aid = area_id.upper() if area_id else None
    findings = [r for r in area_rows if aid is None or r["district_id"] == aid]
    recs_out = [r for r in recs if aid is None or aid in r["affected_areas"]]
    missing = ["service availability/capacity data for: " + ", ".join(SERVICES[s]["en"] for s in SERVICES if s not in INDEPENDENT_INDICATORS),
               "service-level targets (needed for target-based gaps)", "geographic coordinates / travel times (geographic accessibility not measurable)",
               "sales targets (target achievement not computable)", "respondent flag distinguishing prospective buyers from residents",
               "'don't know' / 'service not used' options in q9", "survey response rate vs. population per district"]
    limitations = ["Ordinal self-reported ratings; bottom-box proportions used as primary measure.",
                   "District is the finest area key with adequate n; project-level n is 12-40.",
                   "Survey sample is not probability-based; participation bias unknown.",
                   "Commercial association uses district aggregates (n = districts), cannot establish causation and cannot be adjusted for confounders."]
    if synthetic:
        limitations.insert(0, "ALL SURVEY AND SALES DATA ARE SYNTHETIC DEMONSTRATION DATA. Results exercise the method and are not evidence about any real area.")
    provenance = [{"dataset": "synthetic_survey_responses_ar.csv / acud_survey_responses", "unit": "one eligible resident response", "fields": "q8_*, q9_*, q11, district_id, project_id"},
                  {"dataset": "Fact_District_Service_Indicators", "unit": "district x month", "fields": ", ".join(INDEPENDENT_INDICATORS.values()), "window": "last 12 months"},
                  {"dataset": "Fact_District_Monthly_Status", "unit": "district x month", "fields": "Sell_Through_Pct, Units_Total_District", "window": "latest month"},
                  {"dataset": "Fact_Project_Sales_Funnel + Dim_Project", "unit": "project x month -> district", "fields": "Leads, Bookings, Average_Lead_Response_Hours, Price_Multiplier"}]
    return {**base, "analysis_status": "demonstration_only" if synthetic else "valid", "data_mode": "synthetic_demonstration" if synthetic else "observed",
            "service_findings": findings, "service_priority": priority, "portfolio_service_metrics": portfolio,
            "district_context": ctx.astype(object).where(ctx.notna(), None).to_dict("records"),
            "supporting_evidence": [{"finding_id": r["finding_id"], "grade": r["evidence_grade"], "reasons": r["grade_reasons"]} for r in findings if r["evidence_grade"] not in ("no_concern_detected",)],
            "commercial_context": commercial,
            "alternative_explanations": commercial.get("alternative_explanations_same_outcome", []),
            "recommended_actions": recs_out, "missing_data": missing, "limitations": limitations, "data_provenance": provenance}
