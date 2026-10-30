"""Unified API: 3 agents + free-text ask + dashboard endpoints + the web dashboard.
Run:  uvicorn api:app --reload --port 8000     then open http://localhost:8000
Docs: http://localhost:8000/docs
"""
import os
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

import agents
import dashboard

HERE = os.path.dirname(os.path.abspath(__file__))
app = FastAPI(title="ACUD Real Estate & Urban Intelligence - Agents + Dashboard API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class AgentRequest(BaseModel):
    params: Optional[Dict[str, Any]] = {}
    lang: str = "ar"   # "ar" | "en" - language of the generated texts


class AskRequest(BaseModel):
    text: str
    defaults: Optional[Dict[str, Any]] = {}
    lang: str = "ar"


@app.get("/health")
def health():
    return {"status": "ok", "agents": list(agents.AGENTS), "ml_predictions_loaded": agents._load()["ml"] is not None}


# ---- agents (unified entry) -------------------------------------------------
@app.post("/agents/ask")
def ask_agents(req: AskRequest):
    """Free-text request, e.g. 'قارن أفضل 3 مشروعات بميزانية 8 مليون لمدة 5 سنوات'."""
    try:
        return agents.ask(req.text, lang=req.lang, **(req.defaults or {}))
    except (TypeError, ValueError) as e:
        raise HTTPException(422, f"Bad request: {e}")


@app.post("/agents/{name}")
def call_agent(name: str, req: AgentRequest):
    # /agents/investment_advisor     {"params": {"budget_egp": 10000000, "horizon_years": 5, "risk_tolerance": "medium", "unit_type": "Apartment"}}
    # /agents/developer_performance  {"params": {"developer_id": "DEV5", "stage": "completed"}}
    # /agents/unsold_inventory       {"params": {"project_id": "P030", "discount_rate_pct": 12}}
    try:
        return agents.run_agent(name, lang=req.lang, **(req.params or {}))
    except ValueError as e:
        raise HTTPException(404, str(e))
    except TypeError as e:
        raise HTTPException(422, f"Bad params: {e}")


# ---- KPI endpoints (for the Power BI / dashboard teammates) -----------------------
@app.get("/kpis/portfolio")
def kpis_portfolio():
    return agents.portfolio_kpis()


@app.get("/kpis/districts")
def kpis_districts():
    return agents.district_kpis()


@app.get("/kpis/projects")
def kpis_projects():
    return agents.project_kpis()


# ---- dashboard pages --------------------------------------------------------
@app.get("/meta")
def meta():
    return dashboard.meta()


@app.get("/dashboard/overview")
def dash_overview():
    return dashboard.overview()


@app.get("/dashboard/sales")
def dash_sales(district_id: Optional[str] = None):
    return dashboard.sales_financial(district_id)


@app.get("/dashboard/inventory")
def dash_inventory(district_id: Optional[str] = None):
    return dashboard.inventory_risk(district_id)


@app.get("/dashboard/developers")
def dash_developers(stage: Optional[str] = None, min_units: Optional[int] = None,
                    max_units: Optional[int] = None, max_age_months: Optional[float] = None):
    return dashboard.developers(stage=stage, min_units=min_units, max_units=max_units, max_age_months=max_age_months)


@app.get("/dashboard/smart-city")
def dash_city():
    return dashboard.smart_city()


@app.get("/dashboard/sustainability")
def dash_sustain():
    return dashboard.sustainability()


@app.get("/dashboard/project/{project_id}")
def dash_project(project_id: str):
    r = dashboard.project_detail(project_id)
    if r is None:
        raise HTTPException(404, f"Unknown project {project_id}")
    return r


# ---- web UI -----------------------------------------------------------------
@app.get("/")
def ui():
    return FileResponse(os.path.join(HERE, "dashboard.html"))
