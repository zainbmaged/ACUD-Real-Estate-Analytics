
"""Unified ACUD API:
Operations + Sales + Survey + AI Agents + Dashboard.

Run from the backend directory:
    uvicorn app.main:app --reload --host 127.0.0.1 --port 8000

Docs:
    http://localhost:8000/docs

Agents dashboard:
    http://localhost:8000/agents-ui
"""

import sys
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from sqlalchemy import text

# ------------------------------------------------------------
# PROJECT PATH
# ------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ------------------------------------------------------------
# EXISTING BACKEND IMPORTS
# ------------------------------------------------------------

from app.categories import CATEGORIES
from app.config import get_settings
from app.database import Base, engine
from app import models  # noqa: F401
from app.routers import (
    analytics,
    chat,
    complaints,
    sales,
    survey_sales,
)

# ------------------------------------------------------------
# AGENTS + DASHBOARD IMPORTS (OUTSIDE BACKEND)
# ------------------------------------------------------------

import agents
import dashboard

settings = get_settings()

AGENTS_DASHBOARD = PROJECT_ROOT / "dashboard.html"


# ------------------------------------------------------------
# DATABASE LIFECYCLE
# ------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    if settings.auto_create_tables:
        Base.metadata.create_all(bind=engine)

    yield


# ------------------------------------------------------------
# FASTAPI APPLICATION
# ------------------------------------------------------------

app = FastAPI(
    title="ACUD Unified Real Estate & Urban Intelligence API",
    description=(
        "Unified platform for complaints, survey intelligence, "
        "sales analytics, AI agents, KPIs and dashboards."
    ),
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ------------------------------------------------------------
# EXISTING OPERATIONS ROUTERS
# ------------------------------------------------------------

app.include_router(complaints.router)
app.include_router(chat.router)
app.include_router(analytics.router)
app.include_router(sales.router)
app.include_router(survey_sales.router)


# ------------------------------------------------------------
# REQUEST MODELS
# ------------------------------------------------------------

class AgentRequest(BaseModel):
    params: Dict[str, Any] = Field(default_factory=dict)
    lang: str = "ar"


class AskRequest(BaseModel):
    text: str
    defaults: Dict[str, Any] = Field(default_factory=dict)
    lang: str = "ar"


# ------------------------------------------------------------
# ROOT
# ------------------------------------------------------------

@app.get("/")
def root():
    return {
        "service": "ACUD Unified Real Estate & Urban Intelligence API",
        "version": "2.0.0",
        "status": "running",
        "docs": "/docs",
        "health": "/health",
        "agents_dashboard": "/agents-ui",
        "modules": [
            "complaints",
            "chat",
            "analytics",
            "sales",
            "survey_sales",
            "agents",
            "dashboard",
        ],
    }


# ------------------------------------------------------------
# HEALTH CHECK
# ------------------------------------------------------------

@app.get("/health")
def health():
    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
        database_status = "ok"
    except Exception:
        database_status = "unavailable"

    try:
        ml_loaded = agents._load()["ml"] is not None
        agent_names = list(agents.AGENTS)
    except Exception:
        ml_loaded = False
        agent_names = []

    return {
        "status": "ok" if database_status == "ok" else "degraded",
        "database": database_status,
        "agents": agent_names,
        "ml_predictions_loaded": ml_loaded,
    }


# ------------------------------------------------------------
# CATEGORIES
# ------------------------------------------------------------

@app.get("/api/v1/categories")
def categories():
    return {"categories": CATEGORIES}


# ============================================================
# AI AGENTS
# ============================================================

@app.post("/agents/ask")
def ask_agents(req: AskRequest):
    """Free-text request to the ACUD AI agents."""

    try:
        return agents.ask(
            req.text,
            lang=req.lang,
            **(req.defaults or {}),
        )

    except (TypeError, ValueError) as exc:
        raise HTTPException(
            status_code=422,
            detail=f"Bad request: {exc}",
        ) from exc


@app.post("/agents/{name}")
def call_agent(name: str, req: AgentRequest):
    """Execute a specific ACUD agent."""

    try:
        return agents.run_agent(
            name,
            lang=req.lang,
            **(req.params or {}),
        )

    except ValueError as exc:
        raise HTTPException(
            status_code=404,
            detail=str(exc),
        ) from exc

    except TypeError as exc:
        raise HTTPException(
            status_code=422,
            detail=f"Bad params: {exc}",
        ) from exc


# ============================================================
# KPI ENDPOINTS
# ============================================================

@app.get("/kpis/portfolio")
def kpis_portfolio():
    return agents.portfolio_kpis()


@app.get("/kpis/districts")
def kpis_districts():
    return agents.district_kpis()


@app.get("/kpis/projects")
def kpis_projects():
    return agents.project_kpis()


# ============================================================
# AGENTS DASHBOARD DATA
# ============================================================

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
def dash_developers(
    stage: Optional[str] = None,
    min_units: Optional[int] = None,
    max_units: Optional[int] = None,
    max_age_months: Optional[float] = None,
):
    return dashboard.developers(
        stage=stage,
        min_units=min_units,
        max_units=max_units,
        max_age_months=max_age_months,
    )


@app.get("/dashboard/smart-city")
def dash_city():
    return dashboard.smart_city()


@app.get("/dashboard/sustainability")
def dash_sustain():
    return dashboard.sustainability()


@app.get("/dashboard/project/{project_id}")
def dash_project(project_id: str):
    result = dashboard.project_detail(project_id)

    if result is None:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown project {project_id}",
        )

    return result


# ============================================================
# AGENTS WEB DASHBOARD
# ============================================================

@app.get("/agents-ui")
def agents_ui():
    if not AGENTS_DASHBOARD.is_file():
        raise HTTPException(
            status_code=404,
            detail=f"Dashboard not found: {AGENTS_DASHBOARD}",
        )

    return FileResponse(str(AGENTS_DASHBOARD))
