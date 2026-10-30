"""Executive layer on top of service_gap: sales-position groups, pain points, hypothesis test, priority classes.

Everything here is computed from the data; no narrative text contains hard-coded numbers.

IMPORTANT assumptions (also returned in the output):
* The repository contains NO sales target and NO official weak/good-sales classification. The "sales position"
  used here is DERIVED by this module: districts with latest cumulative sell-through below the median of all districts
  are "weaker", the rest "stronger". It is a relative split for comparison, not a target miss.
* Group comparisons use the DISTRICT as the unit (respondents are nested in districts). With 10 districts the
  test has very little power and the groups also differ in development stage (a confounder).
"""
from __future__ import annotations

import itertools
import math
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from . import service_gap as sg

Q4_TO_SERVICE = {
    "المواصلات": "transport", "الخدمات الطبية": "healthcare", "التعليم": "education", "الأمن": "security", "التسوق": "shopping",
    "النظافة": "cleanliness", "الطرق أو الإنارة": "roads_lighting", "الإنترنت": "internet", "المياه أو الكهرباء": "utilities", "الترفيه": "recreation",
}
Q5_NO_PROBLEM, Q4_NO_PROBLEM = "لا توجد مشكلة مؤثرة", "لم أواجه مشكلة"
SEVERE = ("تأثير كبير", "تأثير كبير جدًا")
FREQUENT = ("بشكل شبه يومي", "كثيرًا")
RECOMMEND_YES = ("غالبًا نعم", "بالتأكيد نعم")

EXEC_PARAMS = {"severe_share_high": 0.5, "min_group_districts": 3, "min_districts_for_group_test": 6,
               "practical_diff_pp": 5.0, "sensitivity_thresholds": [0.20, 0.25, 0.30], "sensitivity_severe": [0.4, 0.5, 0.6]}

SALES_POSITION_RULE = ("Derived by this module (no sales target or official weak/good classification exists in the repository): "
                       "latest cumulative sell-through % below the median across districts = 'weaker', otherwise 'stronger'.")

METRIC_CONTRACT = [
    {"name": "Dissatisfied share (bottom-box)", "meaning": "Share of residents rating a service 1 or 2 on the 1-5 current-level scale", "formula": "count(q9 in {1,2}) / count(valid q9)",
     "numerator": "respondents rating 1-2", "denominator": "eligible respondents in the current filter with a valid q9 for that service", "source": "survey q9_current_level_<service>_1_5",
     "filter_behavior": "recomputed on the filtered respondents", "null_handling": "null excluded from denominator and reported as missing rate; never counted as dissatisfied", "type": "proportion 0-1", "format": "integer % with 95% Wilson CI"},
    {"name": "Satisfied share (top-box)", "meaning": "Share rating 4 or 5", "formula": "count(q9 in {4,5}) / count(valid q9)", "numerator": "respondents rating 4-5", "denominator": "as above", "source": "survey q9",
     "filter_behavior": "recomputed on the filtered respondents", "null_handling": "as above; neutral (3) is neither", "type": "proportion", "format": "integer %"},
    {"name": "Missing rate", "meaning": "Share of eligible respondents with no valid q9", "formula": "count(null q9) / count(eligible respondents)", "numerator": "null or invalid q9", "denominator": "eligible respondents in filter",
     "source": "survey", "filter_behavior": "recomputed", "null_handling": "n/a", "type": "proportion", "format": "1 decimal %"},
    {"name": "Not-applicable rate", "meaning": "Not measurable: q9 has no not-applicable or don't-know option", "formula": "n/a", "numerator": "n/a", "denominator": "n/a", "source": "questionnaire",
     "filter_behavior": "n/a", "null_handling": "reported as null, never 0", "type": "null", "format": "n/a"},
    {"name": "Problem prevalence", "meaning": "Share of residents who reported a problem with the service in the last 90 days (multi-select)", "formula": "count(q4 contains service) / count(valid q4)",
     "numerator": "respondents listing the service", "denominator": "eligible respondents in filter with a valid q4 (multi-select, so shares do not sum to 100%)", "source": "survey q4_recent_service_problems",
     "filter_behavior": "recomputed", "null_handling": "null q4 excluded", "type": "proportion", "format": "integer % with Wilson CI"},
    {"name": "Severe share", "meaning": "Among residents naming the service as the most impactful problem, share rating impact large or very large", "formula": "count(q5=service and q7 in severe) / count(q5=service and valid q7)",
     "numerator": "severe-impact respondents", "denominator": "respondents whose main problem is the service", "source": "survey q5, q7", "filter_behavior": "recomputed (small n possible: shown)", "null_handling": "null q7 excluded", "type": "proportion", "format": "integer %, n shown"},
    {"name": "Importance and gap", "meaning": "Mean importance (q8) and mean importance-minus-current (q9) among respondents answering both", "formula": "mean(q8), mean(q8-q9)", "numerator": "sum", "denominator": "respondents with both answers",
     "source": "survey q8, q9", "filter_behavior": "recomputed", "null_handling": "pairwise complete", "type": "number 1-5", "format": "2 decimals; ordinal scores treated as interval, use with bottom-box"},
    {"name": "Overall area quality", "meaning": "q13 attractiveness 1-5: mean, top-box (4-5), bottom-box (1-2); q14 recommend: share 'probably/definitely yes'", "formula": "count(q13>=4)/count(valid q13)", "numerator": "q13 4-5", "denominator": "valid q13 in filter",
     "source": "survey q13, q14", "filter_behavior": "recomputed", "null_handling": "null excluded", "type": "number / proportion", "format": "1-2 decimals / integer %"},
    {"name": "Sales position (derived)", "meaning": "Relative sales standing of a district", "formula": "sell_through_pct < median(sell_through_pct over districts) -> weaker else stronger", "numerator": "n/a", "denominator": "districts with sales data",
     "source": "Fact_District_Monthly_Status.Sell_Through_Pct (latest month)", "filter_behavior": "used as a filter; the group tests always use all districts", "null_handling": "districts without sales data are unclassified", "type": "category", "format": "label"},
    {"name": "Group difference (weaker minus stronger)", "meaning": "Difference in mean district dissatisfied share between groups, tested at district level", "formula": "mean_d(bottom_box | weaker) - mean_d(bottom_box | stronger); exact Mann-Whitney U, rank-biserial effect; BH across services",
     "numerator": "district bottom-box values", "denominator": "districts per group", "source": "derived", "filter_behavior": "not filter dependent (needs all districts)", "null_handling": "districts without n>=1 excluded", "type": "percentage points, p, effect size", "format": "1 decimal pp"},
    {"name": "Sell-through association", "meaning": "Spearman rho between district dissatisfaction index and sell-through", "formula": "see service_gap.commercial_association", "numerator": "n/a", "denominator": "districts",
     "source": "derived", "filter_behavior": "not filter dependent", "null_handling": "districts without both values excluded", "type": "rho, p", "format": "2 decimals"},
]


# --------------------------------------------------------------------------- helpers
def sales_positions(ctx: pd.DataFrame) -> tuple[dict[str, str], float | None]:
    if "sell_through_pct" not in ctx or ctx["sell_through_pct"].notna().sum() < EXEC_PARAMS["min_districts_for_group_test"]:
        return {}, None
    st = ctx.set_index("District_ID")["sell_through_pct"].dropna()
    med = float(st.median())
    return {d: ("weaker" if v < med else "stronger") for d, v in st.items()}, med


def _share(k: int, n: int) -> dict[str, Any]:
    lo, hi = sg.wilson_ci(k, n)
    return {"k": int(k), "n": int(n), "share": sg._r(k / n if n else None), "ci95": [sg._r(lo), sg._r(hi)]}


def pain_points(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Per service: widespread (prevalence q4), most impactful (q5) and severe (q7) kept as separate measures."""
    q4 = df["q4_recent_service_problems"].dropna().astype(str) if "q4_recent_service_problems" in df else pd.Series(dtype=str)
    lists = q4.str.split(r"\s*[;；]\s*") if len(q4) else pd.Series(dtype=object)
    q5 = df["q5_most_impactful_problem"].dropna().astype(str) if "q5_most_impactful_problem" in df else pd.Series(dtype=str)
    out = []
    for ar, key in Q4_TO_SERVICE.items():
        reporters = int(lists.apply(lambda xs: ar in xs).sum()) if len(lists) else 0
        sub = df[df.get("q5_most_impactful_problem") == ar] if "q5_most_impactful_problem" in df else df.iloc[0:0]
        q7 = sub["q7_problem_severity"].dropna() if "q7_problem_severity" in sub else pd.Series(dtype=str)
        q6 = sub["q6_problem_frequency"].dropna() if "q6_problem_frequency" in sub else pd.Series(dtype=str)
        out.append({"service": key, "label_en": sg.SERVICES[key]["en"], "label_ar": sg.SERVICES[key]["ar"],
                    "prevalence_q4": _share(reporters, len(lists)), "main_problem_q5": _share(int((q5 == ar).sum()), len(q5)),
                    "severe_among_main": _share(int(q7.isin(SEVERE).sum()), len(q7)), "frequent_among_main": _share(int(q6.isin(FREQUENT).sum()), len(q6)),
                    "affected_respondents": reporters})
    return out


def satisfaction_by_service(df: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []
    for key, names in sg.SERVICES.items():
        m = sg._cell_metrics(sg._service_cells(df, key))
        rows.append({"service": key, "label_en": names["en"], "label_ar": names["ar"], **m,
                     "missing_rate": sg._r(m["n_null_excluded"] / len(df) if len(df) else None), "not_applicable_rate": None,
                     "not_applicable_note": "q9 has no not-applicable or don't-know option, so this cannot be measured"})
    return rows


def overall_quality(df: pd.DataFrame) -> dict[str, Any]:
    q13 = pd.to_numeric(df.get("q13_area_attractiveness_1_5"), errors="coerce").dropna() if "q13_area_attractiveness_1_5" in df else pd.Series(dtype=float)
    q14 = df["q14_recommend_area"].dropna() if "q14_recommend_area" in df else pd.Series(dtype=str)
    return {"q13_n": int(len(q13)), "q13_mean": sg._r(float(q13.mean()) if len(q13) else None, 2), "q13_top_box": _share(int((q13 >= 4).sum()), len(q13)),
            "q13_bottom_box": _share(int((q13 <= 2).sum()), len(q13)), "q14_recommend_yes": _share(int(q14.isin(RECOMMEND_YES).sum()), len(q14))}


def district_service_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for key in sg.SERVICES:
        cells = sg._service_cells(df, key)
        for d, g in cells.groupby("district_id"):
            cur = g["cur"].dropna()
            rows.append({"district_id": d, "service": key, "n": int(len(cur)), "k": int((cur <= 2).sum())})
    t = pd.DataFrame(rows)
    t["bottom_box"] = t["k"] / t["n"].replace(0, np.nan)
    return t


def exact_mwu(a: np.ndarray, b: np.ndarray, max_combinations: int = 50000) -> tuple[float, float, str]:
    """Return (U for `a`, two-sided p, method). Exact permutation of group labels using average ranks (correct with ties);
    falls back to scipy's asymptotic test only when the number of label assignments is too large."""
    n1, n2 = len(a), len(b)
    u = float(stats.mannwhitneyu(a, b, alternative="two-sided", method="asymptotic").statistic)
    if math.comb(n1 + n2, n1) > max_combinations:
        return u, float(stats.mannwhitneyu(a, b, alternative="two-sided", method="asymptotic").pvalue), "asymptotic"
    ranks = stats.rankdata(np.concatenate([a, b]))
    centre, obs = n1 * n2 / 2, abs(u - n1 * n2 / 2)
    base = n1 * (n1 + 1) / 2
    hits = total = 0
    for idx in itertools.combinations(range(n1 + n2), n1):
        total += 1
        if abs(ranks[list(idx)].sum() - base - centre) >= obs - 1e-9:
            hits += 1
    return u, hits / total, "exact_permutation"


def group_comparison(df: pd.DataFrame, positions: dict[str, str]) -> dict[str, Any]:
    """Weaker vs stronger districts, per service, with the DISTRICT as the unit of analysis."""
    weak = sorted(d for d, p in positions.items() if p == "weaker")
    strong = sorted(d for d, p in positions.items() if p == "stronger")
    if min(len(weak), len(strong)) < EXEC_PARAMS["min_group_districts"]:
        return {"status": "insufficient_data", "reason": f"needs >= {EXEC_PARAMS['min_group_districts']} districts per group (have {len(weak)} / {len(strong)})", "services": []}
    t = district_service_table(df)
    out, pvals = [], []
    for key in sg.SERVICES:
        s = t[t["service"] == key].dropna(subset=["bottom_box"]).set_index("district_id")
        w, st = s.loc[s.index.intersection(weak), "bottom_box"].to_numpy(), s.loc[s.index.intersection(strong), "bottom_box"].to_numpy()
        row: dict[str, Any] = {"service": key, "label_en": sg.SERVICES[key]["en"], "weaker_districts_n": int(len(w)), "stronger_districts_n": int(len(st))}
        if len(w) >= 2 and len(st) >= 2:
            u, p, _method = exact_mwu(w, st)
            row.update({"weaker_mean": sg._r(float(w.mean())), "stronger_mean": sg._r(float(st.mean())), "diff_pp": sg._r(float(w.mean() - st.mean()) * 100, 1),
                        "rank_biserial": sg._r(2 * float(u) / (len(w) * len(st)) - 1, 2), "p_exact": sg._r(float(p), 4)})
            pvals.append(float(p))
        else:
            pvals.append(None)
        # descriptive pooled respondent shares (respondents are clustered in districts: no test is applied to these)
        for name, ids in (("weaker", weak), ("stronger", strong)):
            sub = s.loc[s.index.intersection(ids)]
            row[f"{name}_pooled"] = _share(int(sub["k"].sum()), int(sub["n"].sum()))
        out.append(row)
    adj = sg.benjamini_hochberg(pvals)
    for r, q in zip(out, adj):
        r["q_bh"] = sg._r(q, 4)
        r["supported"] = bool(q is not None and q < sg.PARAMS["alpha"] and (r.get("diff_pp") or 0) >= EXEC_PARAMS["practical_diff_pp"])
        r["direction"] = None if r.get("diff_pp") is None else ("weaker districts more dissatisfied" if r["diff_pp"] > 0 else "weaker districts less dissatisfied")
    return {"status": "ok", "unit": "district", "weaker_districts": weak, "stronger_districts": strong,
            "method": "two-sided Mann-Whitney U on district dissatisfied shares, p from exact permutation of group labels (tie-correct); rank-biserial effect size; Benjamini-Hochberg across services",
            "min_possible_p": sg._r(min(1.0, 2 / math.comb(len(weak) + len(strong), len(weak))), 4), "services": out}


def contradictory_districts(ctx: pd.DataFrame, area_rows: list[dict[str, Any]], df: pd.DataFrame, positions: dict[str, str]) -> dict[str, Any]:
    tbl = pd.DataFrame(area_rows).groupby("district_id")["bottom_box"].mean().rename("dissatisfaction_index")
    q13 = pd.to_numeric(df["q13_area_attractiveness_1_5"], errors="coerce").groupby(df["district_id"]).mean().rename("q13_mean")
    d = ctx.set_index("District_ID")[["sell_through_pct"]].join(tbl).join(q13).dropna()
    if len(d) < 6 or not positions:
        return {"status": "insufficient_data", "cases": []}
    med = float(d["dissatisfaction_index"].median())
    cases = []
    for did, r in d.iterrows():
        pos, high_dis = positions.get(did), r["dissatisfaction_index"] > med
        if pos == "weaker" and not high_dis:
            kind = "weaker sales position but below-median dissatisfaction"
        elif pos == "stronger" and high_dis:
            kind = "stronger sales position but above-median dissatisfaction"
        else:
            continue
        cases.append({"district_id": did, "kind": kind, "sell_through_pct": sg._r(r["sell_through_pct"], 1), "dissatisfaction_index": sg._r(r["dissatisfaction_index"]), "q13_mean": sg._r(r["q13_mean"], 2)})
    return {"status": "ok", "median_dissatisfaction_index": sg._r(med), "cases": cases,
            "reading": "These districts do not follow the 'services explain sales' pattern; they show services are not the only relevant factor."}


def hypothesis_conclusion(group: dict[str, Any], commercial: dict[str, Any], cases: dict[str, Any], lifecycle: dict[str, Any]) -> dict[str, Any]:
    """Deterministic evidence conclusion; wording matches what the numbers support."""
    if group.get("status") != "ok" or commercial.get("status") not in ("preliminary_association_only",):
        return {"label": "insufficient_data", "text": "Not enough districts with both survey and sales data to test the hypothesis.", "basis": []}
    sup = [s for s in group["services"] if s["supported"]]
    n_svc = len(group["services"])
    p = commercial["primary"]
    corr_ok = p["rho"] is not None and p["p_permutation"] is not None and p["p_permutation"] < sg.PARAMS["alpha"] and abs(p["rho"]) >= 0.3
    basis = [f"{len(sup)} of {n_svc} services differ between weaker and stronger districts after multiple-testing adjustment",
             (f"district dissatisfaction index vs sell-through: rho {p['rho']}, permutation p {p['p_permutation']}, n={commercial['n_districts']} districts" if p["rho"] is not None else "district dissatisfaction index vs sell-through: not computable (no variation)"),
             f"{len(cases.get('cases', []))} of {commercial['n_districts']} districts contradict a simple services-explain-sales pattern"]
    if lifecycle.get("confounded"):
        basis.append("weaker and stronger groups differ in development stage, which cannot be separated from services with this few districts")
    if not sup and not corr_ok:
        label, text = "no_credible_evidence", "These data provide no credible evidence that service dissatisfaction differs between weaker- and stronger-selling districts. The cause of weak sales is unresolved by the survey; commercial and development-stage factors need investigation."
    elif sup and corr_ok:
        label, text = "preliminary_association", "Weaker-selling districts show more dissatisfaction on some services and the district-level association is statistically detectable. This is a preliminary association between area aggregates, not evidence that services cause weak sales."
    else:
        label, text = "weak_or_mixed", "The two tests disagree (group comparison and association). Evidence is weak or mixed; do not act on services alone."
    return {"label": label, "text": text, "basis": basis, "causal_claim": False}


def lifecycle_comparability(ctx: pd.DataFrame, positions: dict[str, str]) -> dict[str, Any]:
    if "Lifecycle_Archetype" not in ctx or not positions:
        return {"confounded": None, "table": {}}
    t = ctx.assign(pos=ctx["District_ID"].map(positions)).dropna(subset=["pos"]).groupby(["Lifecycle_Archetype", "pos"]).size().unstack(fill_value=0)
    only_one = [a for a, r in t.iterrows() if (r > 0).sum() == 1]
    return {"confounded": bool(only_one), "archetypes_in_one_group_only": only_one, "table": {a: {k: int(v) for k, v in r.items()} for a, r in t.iterrows()},
            "note": "Development stage (lifecycle archetype) is a simulation label in the source data; groups that do not overlap on it are not like-for-like."}


# --------------------------------------------------------------------------- priority classes
def classify_services(portfolio: list[dict[str, Any]], pains: list[dict[str, Any]], area_rows: list[dict[str, Any]], group: dict[str, Any],
                      *, thr: float | None = None, severe_cut: float | None = None) -> list[dict[str, Any]]:
    thr = sg.PARAMS["bottom_box_threshold"] if thr is None else thr
    severe_cut = EXEC_PARAMS["severe_share_high"] if severe_cut is None else severe_cut
    pmap = {p["service"]: p for p in pains}
    gmap = {s["service"]: s for s in group.get("services", [])}
    out = []
    for p in portfolio:
        key = p["service"]
        lb = p["bottom_box_ci95"][0]
        widespread = lb is not None and lb >= thr
        important = (p["importance_mean"] or 0) >= sg.PARAMS["importance_high"]
        sev = pmap[key]["severe_among_main"]
        severe = sev["share"] is not None and sev["n"] >= 15 and sev["share"] >= severe_cut
        flagged = [r for r in area_rows if r["service"] == key and r["evidence_grade"] in ("strong_descriptive", "consistent_but_limited", "weak_or_contradictory", "preliminary_association")]
        contradicted = bool(flagged) and sum(r["independent"]["status"] == "relatively_high" for r in flagged) / len(flagged) >= 0.5
        has_independent = key in sg.INDEPENDENT_INDICATORS
        if widespread and important and severe and not contradicted:
            cls, why = "high_priority", "widespread (CI above threshold), important to residents, severe among those most affected, and not contradicted by the independent indicator"
        elif widespread and important and severe and contradicted:
            cls, why = "further_investigation", "survey signal is strong but the independent indicator mostly shows the service as relatively available: cause (quality, reliability, perception) unresolved"
        elif widespread or (p["bottom_box"] or 0) >= thr:
            cls, why = "medium_priority", "dissatisfaction is above the threshold but the widespread, importance and severity criteria are not all met"
        else:
            cls, why = "monitor", "dissatisfied share below the threshold; absence of a flag is not proof of adequacy"
        g = gmap.get(key, {})
        out.append({"service": key, "label_en": p["label_en"], "class": cls, "reason": why, "criteria": {"widespread": widespread, "important": important, "severe": severe, "contradicted_by_indicator": contradicted,
                    "independent_data_available": has_independent}, "bottom_box": p["bottom_box"], "bottom_box_ci95": p["bottom_box_ci95"], "importance_mean": p["importance_mean"],
                    "severe_among_main": sev, "prevalence_q4": pmap[key]["prevalence_q4"], "weak_sales_relevance": ("supported_preliminary" if g.get("supported") else ("not_supported" if g else "untested")),
                    "weaker_vs_stronger_diff_pp": g.get("diff_pp"), "weaker_vs_stronger_q": g.get("q_bh")})
    order = {"high_priority": 0, "further_investigation": 1, "medium_priority": 2, "monitor": 3}
    out.sort(key=lambda r: (order[r["class"]], -(r["bottom_box"] or 0)))
    return out


def classification_sensitivity(portfolio, pains, area_rows, group) -> dict[str, Any]:
    base = {r["service"]: r["class"] for r in classify_services(portfolio, pains, area_rows, group)}
    changes: dict[str, set[str]] = {}
    for thr in EXEC_PARAMS["sensitivity_thresholds"]:
        for sv in EXEC_PARAMS["sensitivity_severe"]:
            for r in classify_services(portfolio, pains, area_rows, group, thr=thr, severe_cut=sv):
                changes.setdefault(r["service"], set()).add(r["class"])
    unstable = {k: sorted(v) for k, v in changes.items() if len(v) > 1}
    return {"threshold_grid": EXEC_PARAMS["sensitivity_thresholds"], "severe_grid": EXEC_PARAMS["sensitivity_severe"], "base": base, "unstable_services": unstable,
            "stable_share": sg._r(1 - len(unstable) / len(base) if base else None, 2)}


# --------------------------------------------------------------------------- readiness & insights
def readiness(dq: dict[str, Any], ctx: pd.DataFrame, positions: dict[str, str], by_district_n: pd.Series, commercial: dict[str, Any]) -> list[dict[str, str]]:
    nd = int(ctx["sell_through_pct"].notna().sum()) if "sell_through_pct" in ctx else 0
    return [
        {"question": "Can we identify weak-sales areas?", "answer": "Limited" if positions else "No",
         "why": f"No sales target or official classification exists in the repository; a relative split on sell-through across {nd} districts is used instead." if positions else "No district sales outcome available."},
        {"question": "Can we measure resident satisfaction?", "answer": "Yes" if dq.get("eligible_clean_rows", 0) >= 200 else "Limited",
         "why": f"{dq.get('eligible_clean_rows')} eligible responses, ratings valid (invalid set to null: {dq.get('invalid_rating_values_set_to_null')}); synthetic data."},
        {"question": "Can we identify recurring issues?", "answer": "Yes", "why": "q4 (multi-select problems), q5 (main problem), q6 and q7 (frequency, severity) are present with low missingness."},
        {"question": "Can we compare areas?", "answer": "Yes (districts), Limited (projects)" if len(by_district_n) else "No",
         "why": f"District respondents range {int(by_district_n.min())}-{int(by_district_n.max())}; project samples are far smaller." if len(by_district_n) else "No area data."},
        {"question": "Can we test a service-sales relationship?", "answer": "Limited" if commercial.get("status") == "preliminary_association_only" else "No",
         "why": f"Only {commercial.get('n_districts', 0)} districts: low power, no confidence interval, confounding by development stage."},
        {"question": "Can we prioritise service improvements?", "answer": "Limited",
         "why": f"Rule-based classes only; independent availability data exists for {len(sg.INDEPENDENT_INDICATORS)} of {len(sg.SERVICES)} services and no service targets exist."}]


def insights(sat: list[dict[str, Any]], classes: list[dict[str, Any]], group: dict[str, Any], conclusion: dict[str, Any], cases: dict[str, Any], n: int, synthetic: bool) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    if not n:
        return [{"level": "observation", "text": "No respondents match the current filters."}]
    if synthetic:
        out.append({"level": "observation", "text": "All figures come from synthetic demonstration data; they illustrate the method and are not evidence about real residents."})
    top = [c for c in classes if c["class"] == "high_priority"][:3] or [c for c in classes if c["class"] in ("medium_priority", "further_investigation")][:3]
    if top:
        out.append({"level": "observation", "text": "Highest dissatisfaction: " + "; ".join(f"{c['label_en']} {round(c['bottom_box'] * 100)}% (95% CI {round(c['bottom_box_ci95'][0] * 100)}-{round(c['bottom_box_ci95'][1] * 100)}%)" for c in top) + f", n={n} respondents."})
    if group.get("status") == "ok":
        sup = [s for s in group["services"] if s["supported"]]
        out.append({"level": "observation", "text": f"{len(sup)} of {len(group['services'])} services show a statistically supported dissatisfaction difference between weaker-selling and stronger-selling districts ({len(group['weaker_districts'])} vs {len(group['stronger_districts'])} districts)."})
    out.append({"level": "interpretation", "text": conclusion["text"]})
    if cases.get("cases"):
        out.append({"level": "interpretation", "text": f"{len(cases['cases'])} districts contradict the services-explain-sales pattern ({', '.join(c['district_id'] for c in cases['cases'])}); services are not the only relevant factor."})
    if conclusion["label"] in ("no_credible_evidence", "weak_or_mixed", "insufficient_data"):
        out.append({"level": "decision", "text": "Do not justify service investment on sales grounds from this survey. Ask the commercial module to examine lead volume, pricing, conversion and development stage; validate service issues with availability data."})
    else:
        out.append({"level": "decision", "text": "Treat the association as a hypothesis: validate with availability data and a controlled comparison before any investment."})
    return out


# --------------------------------------------------------------------------- entry point
def executive_brief(survey, districts, projects, *, indicators=None, status=None, funnel=None, district_id: str | None = None,
                    sales_position: str | None = None) -> dict[str, Any]:
    base = sg.analyze(survey, districts, projects, indicators=indicators, status=status, funnel=funnel)
    meta = {"module": "resident_executive_intelligence", "schema_version": "1.0", "parameters": {**sg.PARAMS, **EXEC_PARAMS}, "metric_contract": METRIC_CONTRACT,
            "sales_position_rule": SALES_POSITION_RULE, "data_quality": base["data_quality"]}
    if base["analysis_status"] == "insufficient_data":
        return {**meta, "analysis_status": "insufficient_data", "missing_data": base["missing_data"]}
    clean, _ = sg.validate_survey(survey, districts, projects)
    ctx = pd.DataFrame(base["district_context"])
    positions, med = sales_positions(ctx)
    synthetic = base["data_mode"] == "synthetic_demonstration"
    area_rows = base["service_findings"]
    for r in area_rows:
        r["sales_position"] = positions.get(r["district_id"])

    sel = set(clean["district_id"].unique())
    if district_id:
        sel &= {district_id.upper()}
    if sales_position:
        sel &= {d for d, p in positions.items() if p == sales_position}
    sub = clean[clean["district_id"].isin(sel)]

    group = group_comparison(clean, positions) if positions else {"status": "insufficient_data", "reason": "no sales positions", "services": []}
    cases = contradictory_districts(ctx, area_rows, clean, positions) if positions else {"status": "insufficient_data", "cases": []}
    lifecycle = lifecycle_comparability(ctx, positions)
    conclusion = hypothesis_conclusion(group, base["commercial_context"], cases, lifecycle)
    pains_all = pain_points(clean)
    classes = classify_services(base["portfolio_service_metrics"], pains_all, area_rows, group)
    sens = classification_sensitivity(base["portfolio_service_metrics"], pains_all, area_rows, group)
    idx = pd.DataFrame(area_rows).groupby("district_id")["bottom_box"].mean()
    district_points = [{"district_id": r["District_ID"], "district_name": r.get("District_Name"), "lifecycle": r.get("Lifecycle_Archetype"), "respondents": int(r["respondents"]),
                        "dissatisfaction_index": sg._r(idx.get(r["District_ID"])), "sell_through_pct": sg._r(r.get("sell_through_pct"), 1), "sales_position": positions.get(r["District_ID"])}
                       for r in ctx.to_dict("records")]
    sat = satisfaction_by_service(sub)
    pains = pain_points(sub)
    per_district = {d: pain_points(clean[clean["district_id"] == d]) for d in sorted(sel)}
    by_n = clean.groupby("district_id").size()
    cls_by = {c["service"]: c for c in classes}
    actions = []
    for r in base["recommended_actions"]:
        if (district_id or sales_position) and not (set(r["affected_areas"]) & sel):
            continue
        c = cls_by.get(r["service"]) if r["service"] else None
        areas = [a for a in r["affected_areas"] if a in sel]
        k = sum(f["bottom_box_count"] for f in area_rows if f["finding_id"] in r["supports_finding_ids"] and f["district_id"] in sel)
        n = sum(f["n_valid"] for f in area_rows if f["finding_id"] in r["supports_finding_ids"] and f["district_id"] in sel)
        actions.append({"recommendation_id": r["recommendation_id"], "areas": areas, "service": r["service"], "problem": r["business_problem"],
                        "evidence": {"pooled_dissatisfied": _share(k, n) if r["service"] else None, "grade": r["evidence_grade"], "supports_finding_ids": r["supports_finding_ids"]},
                        "priority_class": c["class"] if c else "outside_service_scope", "confidence": r["evidence_grade"], "action": r["recommended_action"], "reason": r["reason"],
                        "next_step": r["follow_up_measurement"], "illustrative_only": r["illustrative_only"]})
    return {**meta, "analysis_status": base["analysis_status"], "data_mode": base["data_mode"], "filters": {"district_id": district_id, "sales_position": sales_position},
            "respondents_in_filter": int(len(sub)), "districts_in_filter": sorted(sel),
            "readiness": readiness(base["data_quality"], ctx, positions, by_n, base["commercial_context"]),
            "sales_position": {"rule": SALES_POSITION_RULE, "median_sell_through_pct": sg._r(med, 1) if med is not None else None, "by_district": positions, "official_classification_found": False,
                               "lifecycle_comparability": lifecycle},
            "satisfaction_by_service": sat, "overall_quality": overall_quality(sub), "pain_points": pains, "pain_points_by_district": per_district,
            "area_service_matrix": [r for r in area_rows if r["district_id"] in sel],
            "district_points": district_points, "group_comparison": group, "contradictory_districts": cases, "hypothesis_conclusion": conclusion, "commercial_context": base["commercial_context"],
            "service_priority": classes, "priority_sensitivity": sens, "recommended_actions": actions,
            "insights": insights(sat, classes, group, conclusion, cases, int(len(sub)), synthetic),
            "missing_data": base["missing_data"], "limitations": base["limitations"] + [
                "Resident feedback is indirect evidence about area attractiveness; it cannot explain why non-buyers declined to purchase.",
                "Sales position is derived by this module, not an official classification or target miss."], "data_provenance": base["data_provenance"]}
