# Agents API contract
POST /agents/{investment_advisor|developer_performance|unsold_inventory}  body: {"params": {...}}
POST /agents/ask  body: {"text": "free text"}      GET /kpis/portfolio | /kpis/districts | /kpis/projects
Every agent returns: agent, summary, results, evidence, assumptions, risks, missing_data, data_status.
Params
- investment_advisor: budget_egp*, horizon_years, risk_tolerance(low|medium|high), liquidity_need, unit_type, district_id, top_n, weights{return,risk,liquidity,services,construction,price_value},
  rental_yield_pct (one gross yield for all types; default by type), service_charge_pct (1.0), vacancy_pct (8), inflation_pct (13.9)  -> all optional scenario assumptions
- developer_performance: developer_id, stage(completed|under_construction), min_units, max_units, max_age_months, weights{sales,profitability,collection,delivery,inventory}
- unsold_inventory: project_id (omit for top risky), top_n, discount_rate_pct, response_assumptions{price_cut_pct, price_elasticity, extend_years, down_payment_cut_pts, uplift_per_dp_pt, limited_discount_pct}
See sample_*.json for real outputs.
