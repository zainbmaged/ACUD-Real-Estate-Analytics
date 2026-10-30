"""End-to-end proof:  python verify.py   -> prints PASS/FAIL checklist and writes verify_report.json
Checks data integrity, every agent's output contract (evidence/assumptions/risks/missing_data), cross-source consistency,
and the API layer (via FastAPI TestClient if available)."""
import json
import math
import agents

res, D = [], agents._load()
def chk(name, ok, detail=""):
    res.append({"check": name, "ok": bool(ok), "detail": str(detail)}); print(("PASS " if ok else "FAIL ") + name, detail)

p = D["projects"]
inv = D["Fact_Inventory_Monthly"]
chk("unsold units: project table == unit table == inventory fact",
    p.unsold_units.sum() == (~D["avail_units"].is_sold).sum() == inv[inv.Month == inv.Month.max()].Available_End.sum(), int(p.unsold_units.sum()))
chk("collection rate capped at 100%", p.collection_rate.max() <= 1.0, f"raw max {p.collection_rate_raw.max():.3f}")
sold_out = set(p[p.unsold_units == 0].Project_ID)
recs = [x["Project_ID"] for b in (8e6, 15e6, 30e6, 60e6) for x in agents.investment_advisor(b)["results"]["top_projects"]]
chk("investment advisor only recommends projects with available units", len(recs) > 0 and not (set(recs) & sold_out),
    f"{len(recs)} recs checked, {len(sold_out)} sold-out projects excluded")

def contract(out, name):
    miss = [k for k in ("summary", "results", "evidence", "assumptions", "risks", "missing_data") if not out.get(k)]
    chk(f"{name}: has evidence/assumptions/risks/missing_data", not miss, f"missing: {miss}")
    try:
        json.dumps(out, allow_nan=False); chk(f"{name}: JSON-safe (no NaN)", True)
    except ValueError as e:
        chk(f"{name}: JSON-safe (no NaN)", False, e)

for name, kw in [("investment_advisor", dict(budget_egp=8e6, risk_tolerance="medium")),
                 ("investment_advisor", dict(budget_egp=30e6, risk_tolerance="low")),
                 ("developer_performance", {}), ("developer_performance", dict(developer_id="DEV5")),
                 ("unsold_inventory", {}), ("unsold_inventory", dict(project_id="P038")), ("unsold_inventory", dict(project_id="P030"))]:
    contract(agents.run_agent(name, **kw), f"{name}{kw}")

import re
en_txt = json.dumps([agents.run_agent(n, lang="en", **k) for n, k in [("investment_advisor", dict(budget_egp=8e6)), ("developer_performance", {}), ("unsold_inventory", dict(project_id="P038"))]], ensure_ascii=False)
chk("English mode: no Arabic text in any agent output", not re.search("[\u0600-\u06ff]", en_txt))
r = agents.unsold_inventory("P030")["results"]
chk("unsold agent never recommends a value-destroying alternative",
    not any(a.get("recommended") and a["delta_vs_baseline_egp"] <= 0 for a in r["alternatives"]))
chk("sensitivity sweep ran (12 scenarios)", len(r["sensitivity"]) == 12)

ia = agents.investment_advisor(10e6)["results"]
chk("advisor: rent, inflation and exit depth are in the returns and declared",
    all(x["rent_income_base_egp"] > 0 and x["irr_real_base_pct"] < x["irr_base_pct"] and x["exit_depth_months"] > 0 for x in ia["top_projects"]) and ia["return_assumptions"]["inflation_pct"] > 0,
    {x["Project_ID"]: (round(x["irr_base_pct"], 1), round(x["irr_real_base_pct"], 1)) for x in ia["top_projects"]})
cs = __import__("dashboard").smart_city()
chk("smart city: expected demand derived for populated districts", sum(1 for x in cs["districts"] if x["projected_utilization_pct"] is not None) >= 6)
su = __import__("dashboard").sustainability()
chk("sustainability: renewable share, tariffs, payback, water per resident all computed",
    all(su["kpis"][x] is not None for x in ("renewable_share_pct_12m", "electricity_cost_12m_egp", "portfolio_payback_years", "liters_per_resident_day")),
    {x: round(su["kpis"][x], 1) for x in ("renewable_share_pct_12m", "portfolio_payback_years", "liters_per_resident_day")})
html = open("dashboard.html", encoding="utf-8").read()
chk("dashboard: ACUD logo embedded, brand red used, synthetic banner removed", "data:image/png;base64," in html and "#ED1C24" in html and 'const synth = () => ""' in html)

try:
    from fastapi.testclient import TestClient
    import api
    c = TestClient(api.app)
    chk("API /health", c.get("/health").status_code == 200)
    o = c.post("/agents/unsold_inventory", json={"params": {"project_id": "P038"}}).json()
    chk("API /agents/unsold_inventory returns the P038 analysis", o.get("summary", "").startswith("P038"), o.get("summary", "")[:60])
    o = c.post("/agents/ask", json={"text": "عندي 8 مليون جنيه وعايز استثمار بمخاطرة منخفضة"}).json()
    chk("API /agents/ask parses budget and risk", o["parsed"] == {"budget_egp": 8000000.0, "risk_tolerance": "low"}, o["parsed"])
    o = c.post("/agents/unsold_inventory", json={"lang": "en", "params": {"project_id": "P038"}}).json()
    chk("API lang=en returns English", o["summary"].startswith("P038: 394 unsold"), o["summary"])
    chk("API /dashboard/overview", c.get("/dashboard/overview").status_code == 200)
except Exception as e:
    chk("API layer", False, e)

bad = [x for x in res if not x["ok"]]
json.dump(res, open("verify_report.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(f"\n{len(res) - len(bad)}/{len(res)} checks passed"); raise SystemExit(1 if bad else 0)
