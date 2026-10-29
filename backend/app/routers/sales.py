"""Database-backed ACUD real-estate analytics endpoints.

The prototype imports the supplied synthetic CSVs into tables prefixed with ``acud_``.
The dashboard reads current aggregates from those tables rather than trusting embedded HTML data.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import threading
import joblib
import re
from pathlib import Path
from typing import Literal
from datetime import datetime, timezone

import cohere
import pandas as pd
import numpy as np
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy import inspect, text

from app.config import get_settings
from app.database import engine

router = APIRouter(prefix="/api/v1/sales", tags=["ACUD real-estate analytics"])

DATA_DIR = Path(__file__).resolve().parents[3] / "data"
OUTPUT_DIR = Path(__file__).resolve().parents[3] / "outputs"
TARGETS = {
    "units_sold": "Units sold (new sales)",
    "sales_value_egp": "Contract value sold (EGP)",
    "collections_egp": "Cash collected (EGP)",
    "available_inventory": "Available (unsold) units, end of month",
    "construction_cost_egp": "Construction cost incurred (EGP)",
    "electricity_kwh": "Electricity consumption (kWh)",
}


def _table_exists(name: str) -> bool:
    return inspect(engine).has_table(name)


def _read(name: str) -> pd.DataFrame:
    table = f"acud_{re.sub(r'[^a-zA-Z0-9_]', '_', name).lower()}"
    if not _table_exists(table):
        raise HTTPException(status_code=503, detail=f"Dataset table '{table}' is missing. Run backend/seed_real_estate.py first.")
    with engine.connect() as conn:
        return pd.read_sql(text(f'SELECT * FROM "{table}"'), conn)


def _read_investment_inputs() -> pd.DataFrame:
    """Load investment assumptions reliably even when PostgreSQL has an older seed.

    The checked-in CSV is the canonical fallback. If the database table exists, its
    valid per-unit costs take precedence; blank/missing database values are filled
    from the CSV so an old seed cannot make every project appear to lack inputs.
    """
    required = {"Project_ID", "investment_cost_per_unit_egp"}
    csv_path = DATA_DIR / "Project_Investment_Inputs.csv"
    csv_df = pd.read_csv(csv_path, low_memory=False) if csv_path.exists() else pd.DataFrame()
    try:
        db_df = _read("Project_Investment_Inputs")
    except Exception:
        db_df = pd.DataFrame()

    def normalize(frame: pd.DataFrame) -> pd.DataFrame:
        if frame.empty:
            return pd.DataFrame(columns=["Project_ID", "investment_cost_per_unit_egp", "assumed_cost_ratio_pct", "sell_through_pct", "input_type"])
        # Tolerate case/spacing differences in older imported tables.
        rename = {c: str(c).strip() for c in frame.columns}
        frame = frame.rename(columns=rename).copy()
        lower_to_actual = {str(c).strip().lower(): c for c in frame.columns}
        aliases = {
            "project_id": "Project_ID",
            "investment_cost_per_unit_egp": "investment_cost_per_unit_egp",
            "assumed_cost_ratio_pct": "assumed_cost_ratio_pct",
            "sell_through_pct": "sell_through_pct",
            "input_type": "input_type",
        }
        frame = frame.rename(columns={actual: aliases[key] for key, actual in lower_to_actual.items() if key in aliases and actual != aliases[key]})
        for col in ("Project_ID", "investment_cost_per_unit_egp", "assumed_cost_ratio_pct", "sell_through_pct", "input_type"):
            if col not in frame.columns:
                frame[col] = np.nan
        frame["Project_ID"] = frame["Project_ID"].astype(str).str.strip()
        frame["investment_cost_per_unit_egp"] = pd.to_numeric(frame["investment_cost_per_unit_egp"], errors="coerce")
        return frame[["Project_ID", "investment_cost_per_unit_egp", "assumed_cost_ratio_pct", "sell_through_pct", "input_type"]].drop_duplicates("Project_ID", keep="last")

    csv_df = normalize(csv_df)
    db_df = normalize(db_df)
    if csv_df.empty:
        return db_df
    if db_df.empty:
        return csv_df
    # CSV establishes complete project coverage; use DB values only when valid.
    combined = csv_df.set_index("Project_ID")
    db = db_df.set_index("Project_ID")
    for col in ("investment_cost_per_unit_egp", "assumed_cost_ratio_pct", "sell_through_pct", "input_type"):
        if col in db.columns:
            db_values = db[col].reindex(combined.index)
            if col == "investment_cost_per_unit_egp":
                db_values = db_values.where(db_values > 0)
            combined[col] = db_values.combine_first(combined[col])
    return combined.reset_index()

def _monthly_dashboard() -> dict:
    projects = _read("Dim_Project")
    districts = _read("Dim_District")
    units = _read("Dim_Unit")
    full_monthly = _monthly_frame()      # gap-free full history, used for model fitting
    monthly = full_monthly.tail(24)      # display window
    months = monthly.index.strftime("%Y-%m").tolist()
    hist = {key: [float(v) for v in monthly[key].tolist()] for key in TARGETS}

    sales_window = int(monthly["units_sold"].sum())
    value_window = float(monthly["sales_value_egp"].sum())
    available_end = int(monthly["available_inventory"].iloc[-1]) if len(monthly) else 0
    summary = {
        "districts": int(districts["District_ID"].nunique()),
        "projects": int(projects["Project_ID"].nunique()),
        "units": int(units["Unit_ID"].nunique()),
        "sales_window": sales_window,
        "value_window": value_window,
        "collected": float(monthly["collections_egp"].sum()),
        "available_end": available_end,
        "unit_scored": available_end,
    }

    # Forecasts: every target is backtested against simple baselines and the winner (chosen on earlier
    # folds, reported on the latest folds) produces the forecast. Falls back to the heuristic
    # seasonal-naive rule only when there is too little history for a backtest.
    forecast, forecast_eval = [], {}
    if len(full_monthly) >= 12:
        last_month = full_monthly.index.max()
        for target in TARGETS:
            series = full_monthly[target].astype(float)
            try:
                ev = _evaluate_target_cached(series, TARGETS[target])
            except ValueError:
                ev = None
            if ev is not None:
                forecast_eval[target] = ev
                for horizon in (3, 6, 12):
                    for k, row in enumerate(ev["forecast"][:horizon]):
                        forecast.append({"target": target, "horizon_months": horizon, "Month": row["Month"],
                                         "yhat": row["yhat"], "lo": row["lo"], "hi": row["hi"],
                                         "baseline_yhat": row["baseline_yhat"],
                                         "model": f"selected:{ev['selected_model']}", "horizon_fold": f"{horizon}m",
                                         "mape_backtest": round(ev["after_metrics"]["WAPE_pct"], 2),
                                         "ml_beats_baseline": ev["beats_baseline"], "usable": True})
                continue
            scale = max(float(series.tail(12).mean()), 1.0)
            for horizon in (3, 6, 12):
                for step in range(1, horizon + 1):
                    future_month = last_month + pd.offsets.MonthBegin(step)
                    prior_month = future_month - pd.DateOffset(years=1)
                    base = float(series.loc[prior_month]) if prior_month in series.index else float(series.iloc[-1])
                    margin = scale * ({3: 0.15, 6: 0.22, 12: 0.30}[horizon])
                    forecast.append({"target": target, "horizon_months": horizon, "Month": future_month.strftime("%Y-%m"),
                                    "yhat": max(0.0, base), "lo": max(0.0, base - margin), "hi": max(0.0, base + margin),
                                    "baseline_yhat": max(0.0, base),
                                    "model": "database_seasonal_naive", "horizon_fold": f"{horizon}m", "mape_backtest": None,
                                    "ml_beats_baseline": False, "usable": True})

    result = {"source": "postgresql", "generated_at": pd.Timestamp.now(tz="UTC").isoformat(), "targets": TARGETS,
              "months": months, "hist": hist, "summary": summary, "forecast": forecast, "forecast_eval": forecast_eval,
              "data_status": "synthetic_prototype_data",
              "notes": ["Prototype data is synthetic; results are not evidence about actual ACUD market conditions.",
                        "Monthly values and forecasts are computed from database tables populated by the supplied CSV files.",
                        "Forecast models are selected by rolling-origin backtesting; bands are heuristic (holdout MAE widened by sqrt(step)) and are not calibrated confidence intervals."]}

    # Risk output was generated by the supplied model and is imported as a table too.
    try:
        risk = _read("unsold_risk_scores_asof_2025-12-31")
        if "Project_ID" in risk and "District_ID" in risk:
            pcol = next((c for c in risk.columns if c.lower() in {"p_unsold_90d", "prob_unsold_90d", "p_unsold", "predicted_probability"}), None)
            if pcol:
                risk[pcol] = pd.to_numeric(risk[pcol], errors="coerce").fillna(0).clip(0, 1)
                grp = risk.groupby(["Project_ID", "District_ID"]).agg(units_on_market=("Unit_ID", "count"), mean_p_unsold=(pcol, "mean")).reset_index()
                grp["expected_unsold_90d"] = grp.units_on_market * grp.mean_p_unsold
                grp["expected_sold_90d"] = grp.units_on_market - grp.expected_unsold_90d
                result["risk_proj"] = grp.sort_values("expected_unsold_90d", ascending=False).head(40).to_dict("records")
                dg = risk.groupby("District_ID").agg(units=("Unit_ID", "count"), mean_p=(pcol, "mean")).reset_index()
                dg["expected_unsold"] = dg.units * dg.mean_p
                dg["high"] = risk.assign(_high=risk[pcol] >= .8).groupby("District_ID")["_high"].sum().reindex(dg.District_ID).to_numpy()
                result["risk_dist"] = dg.to_dict("records")
    except HTTPException:
        pass
    return result


@router.get("/dashboard")
def sales_dashboard():
    """Live sales, collections, inventory, cost and utility aggregates from the database."""
    return _monthly_dashboard()


@router.get("/projects")
def project_performance():
    inv = _read("Fact_Inventory_Monthly")
    projects = _read("Dim_Project")
    developers = _read("Dim_Developer")
    latest = pd.to_datetime(inv["Month"]).max()
    recent = inv[pd.to_datetime(inv["Month"]) == latest]
    agg = recent.groupby("Project_ID", as_index=False).agg(units_sold=("Units_Sold", "sum"), available_end=("Available_End", "sum"), sales_value_egp=("Sales_Value_EGP", "sum"))
    out = agg.merge(projects, on="Project_ID", how="left")
    if "Developer_ID" in out and "Developer_ID" in developers:
        out = out.merge(developers, on="Developer_ID", how="left", suffixes=("", "_developer"))
    return {"source": "postgresql", "as_of": str(pd.Timestamp(latest).date()), "projects": out.replace({float("nan"): None}).to_dict("records")}


@router.get("/diagnostics")
def unsold_diagnostics():
    """Return project/district indicators to investigate unsold inventory; associations are not causal proof."""
    district_status = _read("Fact_District_Monthly_Status")
    services = _read("Fact_District_Service_Indicators")
    inventory = _read("Fact_Inventory_Monthly")
    projects = _read("Dim_Project")
    funnel = _read("Fact_Project_Sales_Funnel")
    for frame in (district_status, services, inventory, funnel):
        frame["Month"] = pd.to_datetime(frame["Month"])
    latest_status_month = district_status["Month"].max()
    latest_service_month = services["Month"].max()
    latest_inventory_month = inventory["Month"].max()
    latest_funnel_month = funnel["Month"].max()
    ds = district_status[district_status.Month == latest_status_month]
    sv = services[services.Month == latest_service_month]
    inv = inventory[inventory.Month == latest_inventory_month].groupby(["Project_ID", "District_ID"], as_index=False).agg(
        available_units=("Available_End", "sum"), units_sold=("Units_Sold", "sum"), sales_value_egp=("Sales_Value_EGP", "sum"))
    project_district = projects[[c for c in ["Project_ID", "Project_Name", "District_ID", "Developer_ID", "Price_Multiplier"] if c in projects]]
    district_view = ds.merge(sv, on="District_ID", how="left", suffixes=("_status", "_service"))
    district_inventory = inv.groupby("District_ID", as_index=False).agg(available_units=("available_units", "sum"), units_sold=("units_sold", "sum"))
    district_view = district_view.merge(district_inventory, on="District_ID", how="left")
    funnel_latest = funnel[funnel.Month == latest_funnel_month].copy()
    funnel_latest["viewing_rate"] = funnel_latest["Viewings"] / funnel_latest["Leads"].replace(0, pd.NA)
    funnel_latest["booking_rate"] = funnel_latest["Bookings"] / funnel_latest["Viewings"].replace(0, pd.NA)
    funnel_latest["cancellation_rate"] = funnel_latest["Booking_Cancellations"] / funnel_latest["Bookings"].replace(0, pd.NA)
    funnel_project = funnel_latest.groupby(["Project_ID", "District_ID"], as_index=False).agg(
        leads=("Leads", "sum"), viewings=("Viewings", "sum"), bookings=("Bookings", "sum"),
        cancellations=("Booking_Cancellations", "sum"), avg_lead_response_hours=("Average_Lead_Response_Hours", "mean"),
        marketing_spend_egp=("Marketing_Spend_EGP", "sum"))
    funnel_project["viewing_rate"] = funnel_project["viewings"] / funnel_project["leads"].replace(0, pd.NA)
    funnel_project["booking_rate"] = funnel_project["bookings"] / funnel_project["viewings"].replace(0, pd.NA)
    funnel_project["cancellation_rate"] = funnel_project["cancellations"] / funnel_project["bookings"].replace(0, pd.NA)
    project_view = inv.merge(project_district, on=["Project_ID", "District_ID"], how="left").merge(funnel_project, on=["Project_ID", "District_ID"], how="left")
    clean_districts = district_view.sort_values("available_units", ascending=False).astype(object).where(pd.notna(district_view.sort_values("available_units", ascending=False)), None)
    clean_projects = project_view.sort_values("available_units", ascending=False).head(50)
    clean_projects = clean_projects.astype(object).where(pd.notna(clean_projects), None)
    return {
        "source": "postgresql", "data_status": "synthetic_prototype_data",
        "latest_district_month": str(pd.Timestamp(latest_status_month).date()),
        "latest_funnel_month": str(pd.Timestamp(latest_funnel_month).date()),
        "district_indicators": clean_districts.to_dict("records"),
        "project_funnel_and_inventory": clean_projects.to_dict("records"),
        "interpretation_note": "Traffic, access, service, price multiplier, and funnel features are synthetic demonstrations. Associations are investigative signals, not proven causes."
    }


class SalesQuestion(BaseModel):
    question: str = Field(min_length=2, max_length=4000)
    language: Literal["en", "ar"] = "en"
    district_id: str | None = None
    project_id: str | None = None


@router.post("/ask")
def ask_sales(payload: SalesQuestion):
    """Answer sales questions from focused, time-relevant database evidence."""
    settings = get_settings()
    if not settings.cohere_api_key:
        raise HTTPException(status_code=503, detail="Cohere is not configured. Add COHERE_API_KEY to backend/.env.")

    # IDs are accepted from the optional filters, but also extracted from the question
    # so asking "what happened to P022?" works without filling in the filter controls.
    project_match = re.search(r"\bP\d{3}\b", payload.question.upper())
    district_match = re.search(r"\bD\d{2}\b", payload.question.upper())
    project_id = payload.project_id or (project_match.group(0) if project_match else None)
    district_id = payload.district_id or (district_match.group(0) if district_match else None)

    evidence = {
        "data_status": "synthetic_prototype_data",
        "question": payload.question,
        "selected_project_id": project_id,
        "selected_district_id": district_id,
        "data_granularity_warning": "Sales and operational indicators are monthly; they cannot establish an event on a specific day in mid-May.",
    }

    inventory = _read("Fact_Inventory_Monthly").copy()
    inventory["Month"] = pd.to_datetime(inventory["Month"], errors="coerce")
    projects_dim = _read("Dim_Project")
    district_id_for_project = None
    project_meta = pd.DataFrame()
    if project_id:
        inventory = inventory[inventory["Project_ID"].astype(str).str.upper() == project_id]
        project_meta = projects_dim[projects_dim["Project_ID"].astype(str).str.upper() == project_id]
        if not project_meta.empty:
            district_id_for_project = str(project_meta.iloc[0].get("District_ID", "")) or None
    if district_id:
        inventory = inventory[inventory["District_ID"].astype(str).str.upper() == district_id]
    if not district_id:
        district_id = district_id_for_project
    if district_id:
        evidence["selected_district_id"] = district_id

    # Preserve the monthly timeline instead of sending only a lifetime aggregate.
    evidence["project_monthly_sales_and_inventory"] = inventory.sort_values("Month").tail(36).assign(
        Month=lambda df: df["Month"].dt.strftime("%Y-%m-%d")
    ).where(pd.notna(inventory.sort_values("Month").tail(36)), None).to_dict("records")
    if not project_meta.empty:
        evidence["project_details"] = project_meta.replace({np.nan: None}).to_dict("records")

    # Attach flagged anomaly records, which contain observed-vs-expected metrics and dates.
    try:
        anomalies = _read("anomalies_flagged").copy()
        if project_id and "Entity_ID" in anomalies:
            anomalies = anomalies[anomalies["Entity_ID"].astype(str).str.upper() == project_id]
        if district_id and "Entity_ID" in anomalies and not project_id:
            anomalies = anomalies[anomalies["Entity_ID"].astype(str).str.upper() == district_id]
        evidence["flagged_sales_anomalies"] = anomalies.tail(30).replace({np.nan: None}).to_dict("records")
    except Exception as exc:
        evidence["flagged_sales_anomalies_note"] = f"Anomaly table unavailable: {type(exc).__name__}"

    # Add funnel metrics, including customer-budget segments and conversion indicators.
    try:
        funnel = _read("Fact_Project_Sales_Funnel").copy()
        funnel["Month"] = pd.to_datetime(funnel["Month"], errors="coerce")
        if project_id:
            funnel = funnel[funnel["Project_ID"].astype(str).str.upper() == project_id]
        if district_id:
            funnel = funnel[funnel["District_ID"].astype(str).str.upper() == district_id]
        funnel = funnel.sort_values("Month").tail(36).copy()
        if not funnel.empty:
            funnel["viewing_rate"] = (funnel["Viewings"] / funnel["Leads"].replace(0, np.nan)).round(4)
            funnel["booking_rate"] = (funnel["Bookings"] / funnel["Viewings"].replace(0, np.nan)).round(4)
            funnel["cancellation_rate"] = (funnel["Booking_Cancellations"] / funnel["Bookings"].replace(0, np.nan)).round(4)
            funnel["Month"] = funnel["Month"].dt.strftime("%Y-%m-%d")
        evidence["project_monthly_sales_funnel"] = funnel.replace({np.nan: None}).to_dict("records")
    except Exception as exc:
        evidence["project_monthly_sales_funnel_note"] = f"Sales funnel data unavailable: {type(exc).__name__}"

    # District service/access indicators and status history around the anomaly month.
    for dataset, key in [
        ("Fact_District_Service_Indicators", "district_service_history"),
        ("Fact_District_Monthly_Status", "district_status_history"),
    ]:
        try:
            frame = _read(dataset).copy()
            frame["Month"] = pd.to_datetime(frame["Month"], errors="coerce")
            if district_id and "District_ID" in frame:
                frame = frame[frame["District_ID"].astype(str).str.upper() == district_id]
            frame = frame.sort_values("Month").tail(36).copy()
            frame["Month"] = frame["Month"].dt.strftime("%Y-%m-%d")
            evidence[key] = frame.replace({np.nan: None}).to_dict("records")
        except Exception as exc:
            evidence[key + "_note"] = f"Dataset unavailable: {type(exc).__name__}"

    # Historical transaction dates allow a direct monthly count from sales records.
    try:
        sales = _read("Fact_Sales").copy()
        sales["Sale_Date"] = pd.to_datetime(sales["Sale_Date"], errors="coerce")
        if project_id:
            sales = sales[sales["Project_ID"].astype(str).str.upper() == project_id]
        if district_id:
            sales = sales[sales["District_ID"].astype(str).str.upper() == district_id]
        if not sales.empty:
            sales["Month"] = sales["Sale_Date"].dt.to_period("M").astype(str)
            monthly_sales = sales.groupby("Month", as_index=False).agg(
                transactions=("Sale_ID", "count"), contract_value_egp=("Contract_Value_EGP", "sum"))
            evidence["transaction_sales_by_month"] = monthly_sales.tail(36).replace({np.nan: None}).to_dict("records")
    except Exception as exc:
        evidence["transaction_sales_by_month_note"] = f"Transaction-level sales unavailable: {type(exc).__name__}"

    system_prompt = (
        "You are ACUD's evidence-grounded real-estate sales analyst. Answer in the requested language. "
        "You MUST analyze the supplied evidence rather than give generic advice. Start by identifying the relevant project and period. "
        "For a sales-drop question, compare the month(s) before, during, and after the flagged period; cite exact observed and expected values, "
        "and calculate percentage changes when possible. Use the anomaly table's Period_Start, observed_3m, expected_3m, score, and p_value when present. "
        "Then check whether funnel conversion, lead volume, price multiplier, district service/access indicators, or status indicators changed around that period. "
        "Only claim a factor changed if the provided records show it; distinguish correlation from causation. If data is monthly, explicitly say it cannot identify an event on a specific day. "
        "If evidence is missing, name the missing table/field instead of stopping at a generic disclaimer. "
        "This prototype data is synthetic; state that findings demonstrate the analysis method and are not actual ACUD market findings. "
        "Do not invent facts or metrics."
    )
    try:
        client = cohere.ClientV2(api_key=settings.cohere_api_key)
        response = client.chat(model=settings.cohere_model, messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "Database evidence (JSON):\n" + json.dumps(evidence, ensure_ascii=False, default=str) +
             "\n\nQuestion: " + payload.question + "\nLanguage: " + payload.language},
        ], temperature=0.1, max_tokens=1200)
        content = getattr(response.message, "content", None) or []
        answer = "\n".join(part.text for part in content if getattr(part, "text", None)).strip()
        if not answer:
            raise RuntimeError("Cohere returned an empty response")
    except Exception as exc:
        print(f"Cohere sales analytics error: {type(exc).__name__}: {exc}")
        raise HTTPException(status_code=502, detail="The sales assistant is temporarily unavailable.") from exc
    return {"reply": answer, "provider": "cohere", "model": settings.cohere_model,
            "evidence_source": "postgresql", "data_status": evidence["data_status"]}


# --------------------------- live retraining + forecast ROI ---------------------------
import threading
from datetime import datetime, timezone
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import TimeSeriesSplit

_TRAIN_LOCK = threading.Lock()
_TRAINING = {"status": "idle", "progress": 0, "message": "No retraining run yet", "started_at": None,
             "completed_at": None, "data_timestamp": None, "metrics": None, "before_metrics": None,
             "after_metrics": None, "anomaly_count": None, "error": None, "forecast": None,
             "target_results": None, "backtest_method": None}



# The unit-level 90-day risk model is a separate workflow from monthly forecasting.
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
from nac_ml import risk as risk_models  # noqa: E402

MODEL_DIR = OUTPUT_DIR / "retrained_models"
RISK_RESULTS_PATH = MODEL_DIR / "unsold_risk_latest.json"
_RISK_LOCK = threading.Lock()
_RISK_TRAINING = {"status": "idle", "progress": 0, "message": "No unsold-risk training run yet",
                  "started_at": None, "completed_at": None, "error": None, "results": None}

def _json_default(obj):
    """Make numpy / pandas scalars JSON-safe instead of silently stringifying them."""
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return None if not np.isfinite(obj) else float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, (pd.Timestamp, datetime)):
        return obj.isoformat()
    return str(obj)


# ------------------------------------------------------------------------------
# Monthly forecasting: candidate models, nested model selection, baseline comparison
# ------------------------------------------------------------------------------
MODEL_LABELS = {
    "last_value": "Last value",
    "ma3": "3-month average",
    "seasonal_naive": "Same month last year",
    "snaive_growth": "Same month last year × recent growth",
    "ridge": "Ridge trend + annual cycle",
    "ensemble": "Ensemble (3-mo avg + seasonal growth + Ridge)",
    "seasonal_drift": "Same month last year + recent level shift",
    "blend_seasonal_ma3": "Blend: same month last year + 3-mo average",
    "ets": "Exponential smoothing (level)",
    "median_ensemble": "Median of all candidates",
    "holt_damped": "Damped trend smoothing (Holt)",
    "local_trend": "Recent 9-month damped trend",
    "snaive_level3": "Same month last year × 3-month level change",
    "combo": "Top-3 combination",
}
SHORT_LABELS = {"last_value": "last value", "ma3": "3-mo avg", "seasonal_naive": "same month LY", "snaive_growth": "seasonal×growth",
                "ridge": "Ridge", "ensemble": "ensemble", "seasonal_drift": "seasonal+shift", "blend_seasonal_ma3": "seasonal/3-mo blend",
                "ets": "exp. smoothing", "median_ensemble": "median ensemble", "holt_damped": "Holt damped",
                "local_trend": "9-mo trend", "snaive_level3": "seasonal×level"}
SIMPLE_BASELINES = ("last_value", "ma3", "seasonal_naive")
CANDIDATES = ("last_value", "ma3", "seasonal_naive", "snaive_growth", "ridge", "ensemble",
              "seasonal_drift", "blend_seasonal_ma3", "ets", "median_ensemble",
              "holt_damped", "local_trend", "snaive_level3")
_EVAL_CACHE: dict = {}
_EVAL_CACHE_LOCK = threading.Lock()


def _monthly_frame() -> pd.DataFrame:
    """All six monthly target series on a gap-free month-start index (missing months = 0)."""
    inv = _read("Fact_Inventory_Monthly")
    col = _read("Fact_Collections_Monthly")
    con = _read("Fact_Construction_Monthly")
    util = _read("Fact_Utilities_Monthly")
    for frame in (inv, col, con, util):
        frame["Month"] = pd.to_datetime(frame["Month"], errors="coerce").dt.to_period("M").dt.to_timestamp()
    inv_month = inv.groupby("Month").agg(units_sold=("Units_Sold", "sum"), sales_value_egp=("Sales_Value_EGP", "sum"),
                                         available_inventory=("Available_End", "sum"))
    col_month = col.groupby("Month")["Amount_Collected_EGP"].sum().rename("collections_egp")
    con_month = con.groupby("Month")["Actual_Cost_EGP"].sum().rename("construction_cost_egp")
    util_month = util.groupby("Month")["Electricity_kWh"].sum().rename("electricity_kwh")
    monthly = inv_month.join(col_month, how="outer").join(con_month, how="outer").join(util_month, how="outer").sort_index()
    if monthly.empty:
        return monthly
    full = pd.date_range(monthly.index.min(), monthly.index.max(), freq="MS")
    return monthly.reindex(full).fillna(0)[list(TARGETS)]


def _pack_metrics(actual, predicted):
    """Return comparable regression metrics for a common evaluation window."""
    a = np.asarray(actual, dtype=float)
    p = np.asarray(predicted, dtype=float)
    return {
        "MAE": float(mean_absolute_error(a, p)),
        "RMSE": float(np.sqrt(mean_squared_error(a, p))),
        "WAPE_pct": float(100 * np.abs(a - p).sum() / max(np.abs(a).sum(), 1e-9)),
    }


def _ridge_forecast(values: np.ndarray, horizon: int, alpha: float = 1.0) -> np.ndarray:
    """Regularised Ridge on log1p(target): linear trend + annual sine/cosine terms.

    Extrapolated values are capped at 2x the historical maximum so a steep fitted trend
    cannot produce an absurd 12-month forecast.
    """
    n = len(values)

    def design(i):
        return np.column_stack([i, np.sin(2 * np.pi * i / 12), np.cos(2 * np.pi * i / 12)])

    model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
    model.fit(design(np.arange(n, dtype=float)), np.log1p(np.maximum(values, 0)))
    pred = np.expm1(model.predict(design(np.arange(n, n + horizon, dtype=float))))
    return np.clip(pred, 0, max(float(np.max(values)), 1.0) * 2.0)


def _seasonal_naive(values: np.ndarray, horizon: int) -> np.ndarray:
    n = len(values)
    if n < 12:
        return np.repeat(max(float(values[-1]), 0.0), horizon)
    return np.array([max(float(values[n - 12 + (k % 12)]), 0.0) for k in range(horizon)])


def _ses_level(v: np.ndarray) -> float:
    """Simple exponential smoothing; alpha picked by in-sample one-step squared error."""
    best = None
    for alpha in (0.1, 0.2, 0.3, 0.5, 0.7, 0.9):
        level, sse = float(v[0]), 0.0
        for x in v[1:]:
            sse += (float(x) - level) ** 2
            level = alpha * float(x) + (1 - alpha) * level
        if best is None or sse < best[0]:
            best = (sse, level)
    return max(best[1], 0.0)


def _holt_damped(v: np.ndarray, horizon: int, phi: float = 0.9) -> np.ndarray:
    """Holt's linear smoothing with a damped trend; (alpha, beta) picked by in-sample one-step SSE."""
    best = None
    for a in (0.2, 0.4, 0.6, 0.8):
        for b in (0.1, 0.2, 0.4):
            level, trend, sse = float(v[0]), (float(v[1] - v[0]) if len(v) > 1 else 0.0), 0.0
            for x in v[1:]:
                pred = level + phi * trend
                sse += (float(x) - pred) ** 2
                new = a * float(x) + (1 - a) * pred
                trend = b * (new - level) + (1 - b) * phi * trend
                level = new
            if best is None or sse < best[0]:
                best = (sse, level, trend)
    _, level, trend = best
    return np.maximum(level + np.cumsum(phi ** np.arange(1, horizon + 1)) * trend, 0.0)


def _local_trend(v: np.ndarray, horizon: int, window: int = 9, phi: float = 0.85) -> np.ndarray:
    """Straight line through the last `window` months, extrapolated with a damped slope."""
    w = v[-window:]
    k = len(w)
    if k < 3:
        return np.repeat(max(float(v[-1]), 0.0), horizon)
    slope, intercept = np.polyfit(np.arange(k, dtype=float), w, 1)
    end = intercept + slope * (k - 1)
    return np.maximum(end + slope * np.cumsum(phi ** np.arange(1, horizon + 1)), 0.0)


def _predict_all(values, horizon: int = 1) -> dict:
    """Forecast `horizon` steps with every candidate, fitted on `values` only (no look-ahead)."""
    v = np.asarray(values, dtype=float)
    out = {
        "last_value": np.repeat(max(float(v[-1]), 0.0), horizon),
        "ma3": np.repeat(max(float(v[-3:].mean()), 0.0), horizon),
        "seasonal_naive": _seasonal_naive(v, horizon),
    }
    growth = 1.0
    if len(v) >= 18:
        prior = float(v[-18:-12].sum())
        growth = float(np.clip(v[-6:].sum() / prior, 0.7, 1.4)) if prior > 0 else 1.0
    out["snaive_growth"] = out["seasonal_naive"] * growth
    out["ridge"] = _ridge_forecast(v, horizon)
    out["ensemble"] = np.mean([out["ma3"], out["snaive_growth"], out["ridge"]], axis=0)
    shift = float(v[-6:].mean() - v[-18:-12].mean()) if len(v) >= 18 else 0.0
    out["seasonal_drift"] = np.maximum(out["seasonal_naive"] + shift, 0.0)
    out["blend_seasonal_ma3"] = (out["seasonal_naive"] + out["ma3"]) / 2
    out["ets"] = np.repeat(_ses_level(v), horizon)
    out["holt_damped"] = _holt_damped(v, horizon)
    out["local_trend"] = _local_trend(v, horizon)
    level3 = 1.0
    if len(v) >= 15 and float(v[-15:-12].mean()) > 0:
        level3 = float(np.clip(v[-3:].mean() / v[-15:-12].mean(), 0.5, 2.0))
    out["snaive_level3"] = out["seasonal_naive"] * level3
    out["median_ensemble"] = np.median(np.vstack([out[k] for k in (
        "last_value", "ma3", "seasonal_naive", "snaive_growth", "seasonal_drift", "ridge", "ets", "holt_damped", "local_trend")]), axis=0)
    return out


def _fold_rows(y: np.ndarray, indices) -> list:
    rows = []
    for i in indices:
        preds = _predict_all(y[:i], 1)
        rows.append({"i": int(i), "actual": float(y[i]), "pred": {k: float(v[0]) for k, v in preds.items()}})
    return rows


def _candidate_metrics(rows) -> dict:
    actual = [r["actual"] for r in rows]
    return {m: _pack_metrics(actual, [r["pred"][m] for r in rows]) for m in CANDIDATES}


def _rank_key(metrics):
    return lambda name: (metrics[name]["WAPE_pct"], metrics[name]["MAE"])


def _evaluate_target(series: pd.Series, label: str, horizon: int = 12, max_folds: int = 6,
                     sel_folds: int = 12, combo_k: int = 3) -> dict:
    """Backtest every candidate and choose WITHOUT peeking at the reporting folds.

    * Selection folds  = up to `sel_folds` one-step-ahead months before the reporting window. The candidates are
      ranked here, the baseline is chosen here (in advance, exactly as a user would have to), and the
      forecast is the average of the `combo_k` best candidates (forecast combination).
    * Evaluation folds = the latest `max_folds` months. Reported metrics; nothing about them influenced the choice.
    * The best simple baseline *in hindsight* (chosen on the evaluation folds) is also reported as a stricter yardstick.
    """
    s = pd.to_numeric(series, errors="coerce").dropna().astype(float)
    if len(s) < 14:
        raise ValueError("At least 14 monthly observations are required for rolling backtesting.")
    y = s.to_numpy()
    n = len(y)
    eval_start = max(9, n - max_folds)
    eval_idx = list(range(eval_start, n))
    sel_idx = list(range(max(6, eval_start - sel_folds), eval_start))
    nested = len(sel_idx) >= 3

    eval_rows = _fold_rows(y, eval_idx)
    eval_m = _candidate_metrics(eval_rows)
    sel_m = _candidate_metrics(_fold_rows(y, sel_idx)) if nested else None
    rank_m = sel_m if nested else eval_m
    ranked = sorted(CANDIDATES, key=_rank_key(rank_m))
    members = ranked[:combo_k]
    best_single = ranked[0]

    # the combination is evaluated with members fixed in advance (from the selection folds)
    for r in eval_rows:
        r["pred"]["combo"] = float(np.mean([r["pred"][m] for m in members]))
    actual = [r["actual"] for r in eval_rows]
    eval_m["combo"] = _pack_metrics(actual, [r["pred"]["combo"] for r in eval_rows])
    selected = "combo"

    best_base = min(SIMPLE_BASELINES, key=_rank_key(rank_m))                 # chosen in advance
    hind_base = min(SIMPLE_BASELINES, key=_rank_key(eval_m))                 # best in hindsight (stricter)
    base_wape, sel_wape = eval_m[best_base]["WAPE_pct"], eval_m[selected]["WAPE_pct"]
    improvement = (base_wape - sel_wape) / base_wape * 100 if base_wape > 0 else None
    hind_wape = eval_m[hind_base]["WAPE_pct"]
    hind_improvement = (hind_wape - sel_wape) / hind_wape * 100 if hind_wape > 0 else None
    metric_wins = [k for k in ("MAE", "RMSE", "WAPE_pct") if eval_m[selected][k] < eval_m[best_base][k]]
    best_on_eval = min(CANDIDATES, key=_rank_key(eval_m))

    holdout = []
    for r in eval_rows:
        holdout.append({
            "Month": pd.Timestamp(s.index[r["i"]]).strftime("%Y-%m"), "actual": r["actual"],
            "baseline": r["pred"][best_base], "seasonal_naive": r["pred"]["seasonal_naive"],
            "retrained": r["pred"]["ridge"], "selected_prediction": r["pred"][selected],
            "predictions": r["pred"],
        })

    allp = _predict_all(y, horizon)
    allp["combo"] = np.mean([allp[m] for m in members], axis=0)
    band = eval_m[selected]["MAE"]
    forecast = []
    for k in range(horizon):
        month = pd.Timestamp(s.index[-1]) + pd.offsets.MonthBegin(k + 1)
        value = float(allp[selected][k])
        spread = band * float(np.sqrt(k + 1))
        forecast.append({"Month": month.strftime("%Y-%m"), "yhat": round(value, 2),
                         "lo": round(max(0.0, value - spread), 2), "hi": round(value + spread, 2),
                         "baseline_yhat": round(float(allp[best_base][k]), 2)})
    return {
        "label": label, "model": "Top-3 combination: " + " + ".join(SHORT_LABELS[m] for m in members),
        "selected_model": selected, "members": members, "best_single_model": best_single,
        "baseline_model": MODEL_LABELS[best_base], "baseline_key": best_base,
        "before_metrics": eval_m[best_base], "after_metrics": eval_m[selected],
        "hindsight_baseline_model": MODEL_LABELS[hind_base], "hindsight_metrics": eval_m[hind_base],
        "hindsight_improvement_pct": hind_improvement, "beats_hindsight_baseline": bool(hind_improvement is not None and hind_improvement > 0),
        "candidate_metrics": eval_m, "selection_metrics": sel_m,
        "selection_method": "nested" if nested else "same_folds", "selection_folds": len(sel_idx) if nested else 0,
        "wape_improvement_pct": improvement,
        "beats_baseline": bool(improvement is not None and improvement > 0),
        "metrics_won": metric_wins,
        "best_on_eval_model": best_on_eval, "model_labels": MODEL_LABELS,
        "observations": int(n), "backtest_folds": len(eval_idx), "holdout": holdout,
        "forecast": forecast, "band_mae": band,
        "last_observed_month": pd.Timestamp(s.index[-1]).strftime("%Y-%m"),
    }


def _evaluate_target_cached(series: pd.Series, label: str) -> dict:
    s = pd.to_numeric(series, errors="coerce").dropna().astype(float)
    key = (label, pd.Timestamp(s.index[-1]).isoformat() if len(s) else None, tuple(np.round(s.to_numpy(), 2)))
    with _EVAL_CACHE_LOCK:
        hit = _EVAL_CACHE.get(key)
    if hit is not None:
        return hit
    result = _evaluate_target(s, label)
    with _EVAL_CACHE_LOCK:
        if len(_EVAL_CACHE) > 64:
            _EVAL_CACHE.clear()
        _EVAL_CACHE[key] = result
    return result


def _portfolio_scales(horizon: int):
    """Ratio of the selected portfolio-level forecast to a flat latest-3-month run-rate.

    Used to calibrate the per-project run-rate scenario in /roi so it inherits the
    trend/seasonality learned by the backtested model instead of a plain 3-month mean.
    """
    try:
        frame = _monthly_frame()
        scales, models = {}, {}
        for key in ("sales_value_egp", "units_sold"):
            ev = _evaluate_target_cached(frame[key], TARGETS[key])
            recent = float(frame[key].tail(3).mean())
            fc = float(sum(r["yhat"] for r in ev["forecast"][:horizon]))
            scales[key] = float(np.clip(fc / (recent * horizon), 0.5, 2.0)) if recent > 0 else 1.0
            models[key] = ev["model"]
        return scales["sales_value_egp"], scales["units_sold"], models
    except Exception:
        return 1.0, 1.0, None


def _run_retraining():
    try:
        with _TRAIN_LOCK:
            _TRAINING.update(status="running", progress=5, message="Loading monthly series from PostgreSQL",
                             started_at=datetime.now(timezone.utc).isoformat(), completed_at=None, error=None,
                             target_results=None, forecast=None)
        with _EVAL_CACHE_LOCK:
            _EVAL_CACHE.clear()
        frame = _monthly_frame()
        latest_month = frame.index.max() if not frame.empty else None
        target_results, total = {}, len(TARGETS)
        for idx, (key, label) in enumerate(TARGETS.items(), start=1):
            with _TRAIN_LOCK:
                _TRAINING.update(progress=5 + int(85 * (idx - 1) / total), message=f"Backtesting candidates: {label}")
            try:
                res = _evaluate_target(frame[key], label)
            except ValueError as exc:
                target_results[key] = {"label": label, "error": str(exc), "observations": int(len(frame[key].dropna())),
                                       "selected_model": None}
                continue
            res["forecast"] = [{**row, "target": key} for row in res["forecast"]]
            target_results[key] = res
        units = target_results.get("units_sold", {})
        anomalies = 0
        if "units_sold" in frame and len(frame) >= 8:
            changes = frame["units_sold"].pct_change().replace([np.inf, -np.inf], np.nan).dropna()
            if len(changes):
                med = changes.median(); mad = (changes - med).abs().median()
                scale = 1.4826 * mad if mad > 0 else (changes.std(ddof=0) or 1.0)
                anomalies = int(((changes - med).abs() / scale >= 3.5).sum())
        with _TRAIN_LOCK:
            _TRAINING.update(status="completed", progress=100, message="Backtest and model selection completed",
                             completed_at=datetime.now(timezone.utc).isoformat(),
                             data_timestamp=pd.Timestamp(latest_month).isoformat() if latest_month is not None else None,
                             target_results=target_results,
                             before_metrics=units.get("before_metrics"), after_metrics=units.get("after_metrics"),
                             metrics=units.get("after_metrics"), forecast=units.get("forecast"),
                             model=units.get("model", "Per-target model selection"),
                             observations=units.get("observations"), anomaly_count=anomalies,
                             backtest_method="nested: select on 6 earlier one-step folds, report on the latest 6")
    except Exception as exc:
        with _TRAIN_LOCK:
            _TRAINING.update(status="error", progress=100, message="Retraining failed",
                             completed_at=datetime.now(timezone.utc).isoformat(), error=str(exc))


@router.post("/retrain")
def retrain_latest():
    """Start a background model fit on the latest database sales series."""
    with _TRAIN_LOCK:
        if _TRAINING["status"] == "running":
            return {"accepted": True, **_TRAINING}
        _TRAINING.update(status="queued", progress=0, message="Retraining queued", error=None)
    thread = threading.Thread(target=_run_retraining, daemon=True)
    thread.start()
    return {"accepted": True, "status": "queued", "progress": 0, "message": "Retraining queued"}


@router.get("/retrain/status")
def retraining_status():
    with _TRAIN_LOCK:
        return dict(_TRAINING)


@router.get("/roi")
def forecast_roi(horizon_months: int = Query(3, ge=1, le=12),
                 district_id: str | None = None, project_id: str | None = None):
    """Return a project-level investment-recovery scenario aligned with /equilibrium.

    The primary ROI proxy uses the same total-project investment threshold as the
    break-even chart: planned units × assumed investment cost per unit. It compares
    historical cumulative contract value plus forecast contract value against that
    total. It is not accounting profit or an audited ROI.
    """
    inv = _read("Fact_Inventory_Monthly")
    projects = _read("Dim_Project")
    districts = _read("Dim_District")
    investment_inputs = _read_investment_inputs()
    if inv.empty:
        return {"source": "postgresql", "horizon_months": horizon_months, "rows": [],
                "roi_available": False, "missing_inputs": ["forecast sales", "project investment assumptions"]}

    inv["Month"] = pd.to_datetime(inv["Month"], errors="coerce")
    inv["Sales_Value_EGP"] = pd.to_numeric(inv["Sales_Value_EGP"], errors="coerce").fillna(0)
    inv["Units_Sold"] = pd.to_numeric(inv["Units_Sold"], errors="coerce").fillna(0)
    inv = inv.dropna(subset=["Month"])
    if inv.empty:
        return {"source": "postgresql", "horizon_months": horizon_months, "rows": [],
                "roi_available": False, "missing_inputs": ["valid monthly sales data"]}

    last_month = inv["Month"].max()
    recent = inv[inv["Month"] >= last_month - pd.DateOffset(months=2)].copy()
    agg = recent.groupby(["District_ID", "Project_ID"], as_index=False).agg(
        avg_monthly_sales_egp=("Sales_Value_EGP", "mean"),
        avg_monthly_units_sold=("Units_Sold", "mean"))

    # Use the same project metadata and investment inputs as /equilibrium.
    pcols = [c for c in ["Project_ID", "Project_Name", "District_ID", "Units_Planned"] if c in projects.columns]
    meta = projects[pcols].drop_duplicates("Project_ID")
    if "Units_Planned" not in meta.columns:
        meta["Units_Planned"] = np.nan
    input_cols = [c for c in ["Project_ID", "investment_cost_per_unit_egp", "assumed_cost_ratio_pct", "sell_through_pct", "input_type"] if c in investment_inputs.columns]
    investment_inputs = investment_inputs[input_cols].drop_duplicates("Project_ID")
    meta = meta.merge(investment_inputs, on="Project_ID", how="left")
    agg = agg.merge(meta, on=["Project_ID", "District_ID"], how="left", suffixes=("", "_meta"))

    # Historical cumulative sales use the full recorded window, exactly as the
    # break-even endpoint does, rather than just the recent forecast window.
    historical = inv.groupby("Project_ID", as_index=False).agg(
        historical_cumulative_sales_egp=("Sales_Value_EGP", "sum"))
    agg = agg.merge(historical, on="Project_ID", how="left")
    dname = next((c for c in districts.columns if c.lower() in {"district_name", "name", "district"}), None)
    if dname and "District_ID" in districts.columns:
        agg = agg.merge(districts[["District_ID", dname]].drop_duplicates("District_ID"), on="District_ID", how="left")
    else:
        dname = "District_ID"

    if district_id:
        agg = agg[agg["District_ID"].astype(str) == str(district_id)]
    if project_id:
        agg = agg[agg["Project_ID"].astype(str) == str(project_id)]

    scale_sales, scale_units, scale_models = _portfolio_scales(horizon_months)
    agg["forecast_revenue_egp"] = pd.to_numeric(agg["avg_monthly_sales_egp"], errors="coerce").fillna(0) * horizon_months * scale_sales
    agg["forecast_units"] = pd.to_numeric(agg["avg_monthly_units_sold"], errors="coerce").fillna(0) * horizon_months * scale_units
    rows = []
    for r in agg.to_dict("records"):
        planned = pd.to_numeric(pd.Series([r.get("Units_Planned")]), errors="coerce").iloc[0]
        unit_cost = pd.to_numeric(pd.Series([r.get("investment_cost_per_unit_egp")]), errors="coerce").iloc[0]
        valid_cost = pd.notna(planned) and pd.notna(unit_cost) and float(planned) > 0 and float(unit_cost) > 0
        total_investment = float(planned) * float(unit_cost) if valid_cost else None
        forecast_revenue = float(r.get("forecast_revenue_egp") or 0)
        historical_sales = float(r.get("historical_cumulative_sales_egp") or 0)
        projected_cumulative_sales = historical_sales + forecast_revenue
        recovery_pct = (projected_cumulative_sales / total_investment * 100) if total_investment else None
        # A sales-vs-investment proxy, not profit: contract value is not net income.
        project_roi_proxy = ((projected_cumulative_sales - total_investment) / total_investment * 100) if total_investment else None
        gap = max(total_investment - projected_cumulative_sales, 0) if total_investment else None
        avg_monthly_sales = float(r.get("avg_monthly_sales_egp") or 0) * scale_sales
        months_to_cover = int(np.ceil(gap / avg_monthly_sales)) if gap is not None and gap > 0 and avg_monthly_sales > 0 else (0 if gap == 0 else None)
        projected_be = (last_month + pd.DateOffset(months=horizon_months + months_to_cover)).strftime("%Y-%m") if months_to_cover is not None else None
        rows.append({
            "district_id": r.get("District_ID"), "district": r.get(dname) or r.get("District_ID"),
            "project_id": r.get("Project_ID"), "project": r.get("Project_Name") or r.get("Project_ID"),
            "horizon_months": horizon_months,
            "units_sold": int(round(float(r.get("forecast_units") or 0))),
            "forecast_revenue_egp": round(forecast_revenue, 2),
            "historical_cumulative_sales_egp": round(historical_sales, 2),
            "projected_cumulative_sales_egp": round(projected_cumulative_sales, 2),
            "total_project_investment_egp": round(total_investment, 2) if total_investment is not None else None,
            # Keep this compatibility field, but it now represents total project
            # investment, matching the break-even chart (not cost of forecast units).
            "investment_cost_egp": round(total_investment, 2) if total_investment is not None else None,
            "investment_cost_per_unit_egp": round(float(unit_cost), 2) if pd.notna(unit_cost) else None,
            "assumed_cost_ratio_pct": round(float(r["assumed_cost_ratio_pct"]), 2) if pd.notna(r.get("assumed_cost_ratio_pct")) else None,
            "sell_through_pct": round(float(r["sell_through_pct"]), 2) if pd.notna(r.get("sell_through_pct")) else None,
            "projected_recovery_pct": round(recovery_pct, 2) if recovery_pct is not None else None,
            "projected_gap_egp": round(gap, 2) if gap is not None else None,
            "projected_break_even_month": projected_be,
            "roi_pct": round(project_roi_proxy, 2) if project_roi_proxy is not None else None,
            "roi_status": "scenario_estimate" if project_roi_proxy is not None else "missing_investment_input",
            "break_even_status": "projected_reached" if gap == 0 else ("projected_not_reached" if gap is not None else "missing_investment_input"),
        })
    return {"source": "postgresql", "as_of": last_month.strftime("%Y-%m-%d"), "horizon_months": horizon_months,
            "rows": rows, "roi_available": bool(rows) and any(r["roi_pct"] is not None for r in rows),
            "missing_inputs": [] if any(r["total_project_investment_egp"] is not None for r in rows) else ["planned units and investment_cost_per_unit_egp"],
            "method": "Aligned project-level scenario: total project investment = planned units × assumed investment cost per unit (same threshold as /equilibrium). Projected cumulative sales = all historical recorded contract value + forecast contract value for the selected horizon. Project ROI proxy (%) = (projected cumulative sales − total project investment) / total project investment × 100; recovery (%) = projected cumulative sales / total project investment × 100. This is a sales-versus-investment scenario, not accounting profit or audited ROI. Each project's forecast is its latest-three-month run-rate, scaled by the ratio of the backtest-selected portfolio forecast to the portfolio's flat three-month run-rate (clipped to 0.5x-2x), so it inherits the portfolio model's trend and seasonality. The projected break-even month extrapolates that scaled run-rate and is not guaranteed.",
            "data_status": "synthetic_scenario_assumptions",
            "forecast_calibration": {"sales_scale": round(scale_sales, 4), "units_scale": round(scale_units, 4), "models": scale_models}}


def _load_risk_tables() -> dict[str, pd.DataFrame]:
    """Read the source tables needed by the unit-level 90-day risk pipeline."""
    names = ["Dim_Date", "Dim_Price_Index", "Dim_District", "Dim_Developer", "Dim_Project", "Dim_Unit",
             "Fact_Sales", "Fact_Construction_Monthly", "Fact_District_Monthly_Status"]
    tables = {name: _read(name) for name in names}
    for col in ("Date",):
        tables["Dim_Date"][col] = pd.to_datetime(tables["Dim_Date"][col], errors="coerce")
    tables["Dim_Price_Index"]["Month"] = pd.to_datetime(tables["Dim_Price_Index"]["Month"], errors="coerce")
    for name in ("Fact_Construction_Monthly", "Fact_District_Monthly_Status"):
        tables[name]["Month"] = pd.to_datetime(tables[name]["Month"], errors="coerce")
    for col in ("Sale_Date", "Cancel_Date"):
        tables["Fact_Sales"][col] = pd.to_datetime(tables["Fact_Sales"][col], errors="coerce")
    tables["Dim_Unit"]["Release_Date"] = pd.to_datetime(tables["Dim_Unit"]["Release_Date"], errors="coerce")
    return tables


def _risk_metric_row(name: str, y_true: np.ndarray, probabilities: np.ndarray,
                     train: pd.DataFrame, test: pd.DataFrame) -> dict:
    p = np.clip(np.asarray(probabilities, dtype=float), 1e-4, 1 - 1e-4)
    y = np.asarray(y_true, dtype=int)
    top_precision = test.assign(_p=p).groupby("as_of").apply(
        lambda g: g.nlargest(max(int(len(g) * 0.2), 1), "_p").y_unsold.mean()
    ).mean()
    return {
        "model": name, "n_train": int(len(train)), "n_test": int(len(test)),
        "unsold_rate_train": round(float(train.y_unsold.mean()), 4),
        "unsold_rate_test": round(float(y.mean()), 4),
        "ROC_AUC": round(float(risk_models.roc_auc_score(y, p)) if len(np.unique(y)) > 1 and name != "prior_rate" else 0.5, 4),
        "PR_AUC": round(float(risk_models.average_precision_score(y, p)), 4),
        "Brier": round(float(risk_models.brier_score_loss(y, p)), 4),
        "LogLoss": round(float(risk_models.log_loss(y, p)), 4),
        "Precision@top20%": round(float(top_precision), 4),
    }


def _confusion_summary(y, p, share: float) -> dict:
    """Confusion matrix when the highest-risk `share` of listings is flagged 'will stay unsold'.

    The cut-off is rank-based (no labels used): `share` is the training unsold rate, so every
    model flags the same number of units and the comparison is fair.
    """
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    k = int(np.clip(round(len(p) * share), 1, len(p)))
    thr = float(np.sort(p)[::-1][k - 1])
    flag = p >= thr
    tp = int((flag & (y == 1)).sum()); fp = int((flag & (y == 0)).sum())
    fn = int((~flag & (y == 1)).sum()); tn = int((~flag & (y == 0)).sum())
    recall = tp / (tp + fn) if tp + fn else None
    specificity = tn / (tn + fp) if tn + fp else None
    top20 = y[p >= np.quantile(p, 0.8)]
    bottom20 = y[p <= np.quantile(p, 0.2)]
    return {
        "threshold": round(thr, 4), "flagged_share": round(float(flag.mean()), 4),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": round(tp / (tp + fp), 4) if tp + fp else None,
        "recall": round(recall, 4) if recall is not None else None,
        "specificity": round(specificity, 4) if specificity is not None else None,
        "balanced_accuracy": round((recall + specificity) / 2, 4) if recall is not None and specificity is not None else None,
        "top20_actual_unsold": round(float(top20.mean()), 4) if len(top20) else None,
        "bottom20_actual_unsold": round(float(bottom20.mean()), 4) if len(bottom20) else None,
        "base_rate": round(float(y.mean()), 4),
    }


def _risk_backtest_details(test_preds: pd.DataFrame, model_names: list, train_rate: float) -> dict:
    """Actual-vs-predicted evidence on the chronological hold-out (what the model said vs what happened)."""
    scored = [m for m in model_names if f"p_{m}" in test_preds.columns]
    y = test_preds["y_unsold"].astype(int).to_numpy()

    confusion = []
    for m in scored:
        if m == "prior_rate":
            continue  # constant probability: no ranking to evaluate
        confusion.append({"model": m, **_confusion_summary(y, test_preds[f"p_{m}"].to_numpy(), train_rate)})

    by_asof = []
    for as_of, g in test_preds.groupby("as_of"):
        rec = {"as_of": pd.Timestamp(as_of).strftime("%Y-%m-%d"), "n": int(len(g)),
               "actual": round(float(g.y_unsold.mean()), 4)}
        for m in scored:
            rec[m] = round(float(g[f"p_{m}"].mean()), 4)
        by_asof.append(rec)

    proj = test_preds.groupby(["Project_ID", "District_ID"])
    by_project = []
    for (pid, did), g in proj:
        rec = {"Project_ID": pid, "District_ID": did, "n": int(len(g)), "actual_unsold": int(g.y_unsold.sum()),
               "actual_rate": round(float(g.y_unsold.mean()), 4),
               "expected": {m: round(float(g[f"p_{m}"].sum()), 2) for m in scored},
               "predicted_rate": {m: round(float(g[f"p_{m}"].mean()), 4) for m in scored}}
        by_project.append(rec)
    by_project.sort(key=lambda r: -r["n"])

    totals = []
    actual_total = int(y.sum())
    for m in scored:
        expected_total = float(test_preds[f"p_{m}"].sum())
        pred_rates = np.array([r["predicted_rate"][m] for r in by_project])
        act_rates = np.array([r["actual_rate"] for r in by_project])
        corr = None
        if len(by_project) >= 3 and pred_rates.std() > 0 and act_rates.std() > 0:
            corr = round(float(np.corrcoef(pred_rates, act_rates)[0, 1]), 4)
        totals.append({"model": m, "expected_unsold": round(expected_total, 1), "actual_unsold": actual_total,
                       "error_pct": round((expected_total - actual_total) / max(actual_total, 1) * 100, 2),
                       "project_rate_corr": corr,
                       "project_mae_pp": round(float(np.abs(pred_rates - act_rates).mean() * 100), 2) if len(by_project) else None})
    return {"confusion": confusion, "by_asof": by_asof, "by_project": by_project[:120], "totals": totals,
            "n_projects": len(by_project), "n_test_obs": int(len(y)),
            "test_window": [by_asof[0]["as_of"], by_asof[-1]["as_of"]] if by_asof else None,
            "flag_share": round(float(train_rate), 4)}


def _train_unsold_risk() -> dict:
    """Retrain the 90-day risk baselines, logistic model and gradient boosting on latest DB data."""
    tables = _load_risk_tables()
    ctx = risk_models.Context(tables)
    sales = tables["Fact_Sales"]
    dates_dim = tables["Dim_Date"]["Date"].dropna()
    inv = _read("Fact_Inventory_Monthly")
    inv_months = pd.to_datetime(inv["Month"], errors="coerce").dropna()
    if sales["Sale_Date"].dropna().empty or dates_dim.empty or inv_months.empty:
        raise ValueError("Unsold-risk retraining requires dated sales, calendar, and monthly inventory data.")

    # Only as-of dates whose full 90-day outcome window has elapsed are eligible for supervised training.
    latest_observed = min(sales["Sale_Date"].max(), dates_dim.max())
    latest_label_asof = pd.Timestamp(latest_observed).normalize() - pd.Timedelta(days=risk_models.H)
    start = max(pd.Timestamp("2024-03-01"), pd.Timestamp(tables["Dim_Unit"]["Release_Date"].min()).normalize())
    dates = pd.date_range(start, latest_label_asof, freq="14D")
    if len(dates) < 24:
        raise ValueError(f"Not enough eligible as-of dates for 90-day risk training ({len(dates)} found; need at least 24).")

    # Chronological split with a 90-day purge gap to prevent overlapping outcome windows.
    cut_idx = max(int(len(dates) * 0.80), 12)
    test_dates = dates[cut_idx:]
    cutoff = test_dates.min()
    train_dates = dates[dates + pd.Timedelta(days=risk_models.H) < cutoff]
    if len(train_dates) < 10 or len(test_dates) < 3:
        raise ValueError("Not enough history for a purged temporal risk holdout; add more dated sales history.")

    train = risk_models.build(ctx, train_dates)
    test = risk_models.build(ctx, test_dates)
    if train.empty or test.empty or train.y_unsold.nunique() < 2:
        raise ValueError("Risk training data is empty or has only one outcome class.")
    train.loc[:, risk_models.FEATURES] = train[risk_models.FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0)
    test.loc[:, risk_models.FEATURES] = test[risk_models.FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0)

    models = risk_models._models()
    metrics, test_preds = [], test[["Unit_ID", "Project_ID", "District_ID", "as_of", "y_unsold"]].copy()
    fitted_final = {}
    train_rate = float(train.y_unsold.mean())
    for name, estimator in models.items():
        if name == "prior_rate":
            p_test = np.full(len(test), train_rate)
        elif name == "project_rate_rule":
            p_test = np.clip(risk_models._rule(test), 0.01, 0.99)
        else:
            estimator.fit(train[risk_models.FEATURES], train.y_unsold.astype(int))
            p_test = estimator.predict_proba(test[risk_models.FEATURES])[:, 1]
        p_test = np.clip(p_test, 1e-4, 1 - 1e-4)
        test_preds[f"p_{name}"] = p_test
        metrics.append(_risk_metric_row(name, test.y_unsold.to_numpy(), p_test, train, test))

    # Extra candidates: a probability blend of the two classifiers, and an isotonic-calibrated GBM
    # fitted on time-ordered folds (calibration never sees a later period than the model it corrects).
    try:
        test_preds["p_blend"] = np.clip((test_preds["p_logistic"] + test_preds["p_gbm"]) / 2, 1e-4, 1 - 1e-4)
        metrics.append(_risk_metric_row("blend", test.y_unsold.to_numpy(), test_preds["p_blend"].to_numpy(), train, test))
    except Exception as exc:
        print(f"Risk blend skipped: {type(exc).__name__}: {exc}")
    try:
        train_sorted = train.sort_values("as_of", kind="stable")
        cal_gbm = CalibratedClassifierCV(clone(models["gbm"]), method="isotonic", cv=TimeSeriesSplit(n_splits=3))
        cal_gbm.fit(train_sorted[risk_models.FEATURES], train_sorted.y_unsold.astype(int))
        test_preds["p_gbm_calibrated"] = np.clip(cal_gbm.predict_proba(test[risk_models.FEATURES])[:, 1], 1e-4, 1 - 1e-4)
        metrics.append(_risk_metric_row("gbm_calibrated", test.y_unsold.to_numpy(), test_preds["p_gbm_calibrated"].to_numpy(), train, test))
    except Exception as exc:
        print(f"Calibrated GBM skipped: {type(exc).__name__}: {exc}")

    # Skill versus the two baselines, and the primary scoring model chosen on hold-out Brier score.
    prior_row = next(m for m in metrics if m["model"] == "prior_rate")
    rule_row = next(m for m in metrics if m["model"] == "project_rate_rule")
    for m in metrics:
        m["Brier_skill_vs_prior"] = round(1 - m["Brier"] / prior_row["Brier"], 4) if prior_row["Brier"] > 0 else None
        m["ROC_AUC_lift_vs_rule"] = round(m["ROC_AUC"] - rule_row["ROC_AUC"], 4)
        is_model = m["model"] not in ("prior_rate", "project_rate_rule")
        m["metrics_won"] = ([k for k in ("ROC_AUC", "PR_AUC", "Precision@top20%") if m[k] > max(prior_row[k], rule_row[k])] +
                            [k for k in ("Brier", "LogLoss") if m[k] < min(prior_row[k], rule_row[k])]) if is_model else []
        m["beats_baselines"] = bool(is_model and m["ROC_AUC"] > rule_row["ROC_AUC"]
                                    and m["Brier"] < min(prior_row["Brier"], rule_row["Brier"]))
    by_name = {m["model"]: m for m in metrics}
    primary = min([n for n in ("logistic", "gbm", "blend", "gbm_calibrated") if n in by_name],
                  key=lambda nme: (by_name[nme]["Brier"], -by_name[nme]["ROC_AUC"]))
    details = _risk_backtest_details(test_preds, [m["model"] for m in metrics], train_rate)
    cal_by_model = {}
    for nme in ("logistic", "gbm", "blend", "gbm_calibrated", "project_rate_rule"):
        try:
            cal_by_model[nme] = risk_models.calibration(test_preds, f"p_{nme}").to_dict("records")
        except Exception:
            cal_by_model[nme] = []

    # Fit production classifiers on all label-complete dates, then score current listed units.
    # Final production fit uses every label-complete date, including the holdout and
    # the purge-gap dates after evaluation has been recorded.
    all_rows = risk_models.build(ctx, dates)
    all_rows.loc[:, risk_models.FEATURES] = all_rows[risk_models.FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0)
    for name in ("logistic", "gbm"):
        estimator = models[name]
        estimator.fit(all_rows[risk_models.FEATURES], all_rows.y_unsold.astype(int))
        fitted_final[name] = estimator
    if "gbm_calibrated" in by_name:
        try:
            ordered = all_rows.sort_values("as_of", kind="stable")
            final_cal = CalibratedClassifierCV(clone(models["gbm"]), method="isotonic", cv=TimeSeriesSplit(n_splits=3))
            final_cal.fit(ordered[risk_models.FEATURES], ordered.y_unsold.astype(int))
            fitted_final["gbm_calibrated"] = final_cal
        except Exception as exc:
            print(f"Final calibrated GBM skipped: {type(exc).__name__}: {exc}")

    calendar_limit = pd.Timestamp(dates_dim.max()).normalize() - pd.Timedelta(days=risk_models.H)
    inventory_asof = pd.to_datetime(inv_months.max()).to_period("M").end_time.normalize()
    as_of = min(inventory_asof, calendar_limit)
    if as_of < start:
        raise ValueError("Calendar does not extend far enough to score the latest 90-day unsold risk.")

    # Score using the logistic model as the dashboard's primary risk ranking; retain GBM too.
    current_features = risk_models.features(ctx, as_of, with_label=False)
    if current_features.empty:
        raise ValueError(f"No available units found to score as of {as_of.date()}.")
    current_features.loc[:, risk_models.FEATURES] = current_features[risk_models.FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0)
    for name, estimator in fitted_final.items():
        current_features[f"p_unsold_90d_{name}"] = estimator.predict_proba(current_features[risk_models.FEATURES])[:, 1]
    current_features["p_unsold_90d_blend"] = (current_features["p_unsold_90d_logistic"] + current_features["p_unsold_90d_gbm"]) / 2
    if f"p_unsold_90d_{primary}" not in current_features.columns:
        primary = "gbm" if primary == "gbm_calibrated" else "logistic"
    current_features["p_unsold_90d"] = current_features[f"p_unsold_90d_{primary}"].clip(0, 1)
    current_features["risk_band"] = pd.cut(current_features.p_unsold_90d, [0, 0.8, 0.92, 1.0],
                                             labels=["Lower", "Elevated", "High"], include_lowest=True).astype(str)

    model_dir = MODEL_DIR
    model_dir.mkdir(parents=True, exist_ok=True)
    for name, estimator in fitted_final.items():
        artifact = model_dir / f"unsold_risk_{name}.joblib"
        fd, temp_path = tempfile.mkstemp(prefix=artifact.stem, suffix=".tmp", dir=str(model_dir))
        os.close(fd)
        try:
            joblib.dump({"model": estimator, "features": risk_models.FEATURES, "horizon_days": risk_models.H,
                         "as_of": as_of.strftime("%Y-%m-%d"), "trained_through_as_of": latest_label_asof.strftime("%Y-%m-%d")}, temp_path)
            os.replace(temp_path, artifact)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    # Persist diagnostic CSVs for audit/reproducibility.
    pd.DataFrame(metrics).to_csv(model_dir / "unsold_risk_metrics.csv", index=False)
    for nme, rows_ in cal_by_model.items():
        pd.DataFrame(rows_).to_csv(model_dir / f"unsold_risk_calibration_{nme}.csv", index=False)
    pd.DataFrame(details["by_project"]).drop(columns=["expected", "predicted_rate"]).to_csv(
        model_dir / "unsold_risk_holdout_by_project.csv", index=False)
    current_features.to_csv(model_dir / "unsold_risk_scores_latest.csv", index=False)

    metric_frame = pd.DataFrame(metrics)
    proj = current_features.groupby(["Project_ID", "District_ID"]).agg(
        units_on_market=("Unit_ID", "size"), expected_unsold_90d=("p_unsold_90d", "sum"),
        mean_p_unsold=("p_unsold_90d", "mean")).reset_index()
    proj["expected_sold_90d"] = proj.units_on_market - proj.expected_unsold_90d
    dist = current_features.groupby("District_ID").agg(units=("Unit_ID", "size"), mean_p=("p_unsold_90d", "mean"),
        expected_unsold=("p_unsold_90d", "sum")).reset_index()
    high_counts = current_features.assign(_high=current_features.p_unsold_90d >= 0.92).groupby("District_ID")["_high"].sum()
    dist["high"] = dist.District_ID.map(high_counts).fillna(0).astype(int)
    top_cols = [c for c in ["Unit_ID", "Project_ID", "District_ID", "p_unsold_90d", "risk_band"] if c in current_features.columns]
    current_scores = {
        "band_counts": {str(k): int(v) for k, v in current_features["risk_band"].value_counts().items()},
        "top_units": current_features.sort_values("p_unsold_90d", ascending=False).head(25)[top_cols].round(4).to_dict("records"),
        "mean_p_unsold": round(float(current_features.p_unsold_90d.mean()), 4),
        "expected_unsold": round(float(current_features.p_unsold_90d.sum()), 1),
    }
    result = {
        "available": True,
        "as_of": as_of.strftime("%Y-%m-%d"), "trained_through_as_of": latest_label_asof.strftime("%Y-%m-%d"),
        "n_train": int(len(train)), "n_test": int(len(test)), "n_current_units": int(len(current_features)),
        "unsold_rate_test": round(float(test.y_unsold.mean()), 4), "metrics": metric_frame.to_dict("records"),
        "primary_model": primary, "calibration_model": primary,
        "calibration": cal_by_model.get(primary, []), "calibration_by_model": cal_by_model,
        "backtest": details, "current_scores": current_scores,
        "projects": proj.sort_values("expected_unsold_90d", ascending=False).round(4).to_dict("records"),
        "districts": dist.round(4).to_dict("records"),
        "artifacts": {name: str((model_dir / f"unsold_risk_{name}.joblib").relative_to(OUTPUT_DIR.parent)) for name in fitted_final},
                "model_directory": str(model_dir),
        "message": "90-day unsold-risk models evaluated on a chronological holdout with a 90-day purge gap.",
    }
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    temp_json = RISK_RESULTS_PATH.with_suffix(".json.tmp")
    temp_json.write_text(json.dumps(result, default=_json_default, indent=2), encoding="utf-8")
    os.replace(temp_json, RISK_RESULTS_PATH)
    return result


@router.get("/risk/latest")
def latest_unsold_risk_results():
    """Return the latest persisted 90-day risk evaluation; never invent fallback metrics."""
    try:
        if not RISK_RESULTS_PATH.exists():
            return {"available": False, "metrics": [], "calibration": [], "projects": [], "districts": [],
                    "as_of": None, "trained_through_as_of": None,
                    "message": "No persisted 90-day unsold-risk evaluation is available. Start risk retraining below."}
        saved = json.loads(RISK_RESULTS_PATH.read_text(encoding="utf-8"))
        if not isinstance(saved, dict) or not saved.get("available"):
            raise ValueError("Saved risk evaluation has an invalid format.")
        return saved
    except Exception as exc:
        return {"available": False, "metrics": [], "calibration": [], "projects": [], "districts": [],
                "as_of": None, "trained_through_as_of": None,
                "message": f"Saved risk evaluation could not be read: {type(exc).__name__}: {exc}"}


def _run_risk_retraining():
    with _RISK_LOCK:
        _RISK_TRAINING.update(status="running", progress=5, message="Loading dated unit, sales and market tables",
                               started_at=datetime.now(timezone.utc).isoformat(), completed_at=None, error=None)
    try:
        with _RISK_LOCK:
            _RISK_TRAINING.update(progress=20, message="Building 90-day outcomes and temporal holdout")
        result = _train_unsold_risk()
        with _RISK_LOCK:
            _RISK_TRAINING.update(status="completed", progress=100, message="Unsold-risk evaluation and current scores saved",
                                  completed_at=datetime.now(timezone.utc).isoformat(), results=result, error=None)
    except Exception as exc:
        with _RISK_LOCK:
            _RISK_TRAINING.update(status="error", progress=100, message="Unsold-risk retraining failed",
                                  completed_at=datetime.now(timezone.utc).isoformat(), error=f"{type(exc).__name__}: {exc}")


@router.post("/risk/retrain")
def retrain_unsold_risk():
    """Start the independent 90-day unsold-risk workflow."""
    with _RISK_LOCK:
        if _RISK_TRAINING["status"] == "running":
            return {"accepted": True, **_RISK_TRAINING}
        _RISK_TRAINING.update(status="queued", progress=0, message="Unsold-risk retraining queued", error=None)
    threading.Thread(target=_run_risk_retraining, daemon=True).start()
    return {"accepted": True, "status": "queued", "progress": 0, "message": "Unsold-risk retraining queued"}


@router.get("/risk/retrain/status")
def unsold_risk_retraining_status():
    with _RISK_LOCK:
        return dict(_RISK_TRAINING)


@router.get("/equilibrium")
def project_sales_equilibrium(district_id: str | None = None, project_id: str | None = None):
    """Monthly project sales and an explicitly estimated cumulative-revenue break-even point.

    Break-even threshold = Units_Planned × synthetic investment_cost_per_unit_egp.
    Break-even month is the first observed month cumulative contract value sold meets that
    threshold. Since investment inputs are synthetic, this is a scenario estimate, not an
    audited accounting break-even. Projects that never cross the threshold are reported as
    not reached within the observed period.
    """
    monthly = _read("Fact_Inventory_Monthly")
    projects = _read("Dim_Project")
    inputs = _read_investment_inputs()
    districts = _read("Dim_District")
    if monthly.empty:
        return {"source": "postgresql", "rows": [], "projects": [], "as_of": None,
                "method": "No monthly sales data available."}

    monthly["Month"] = pd.to_datetime(monthly["Month"], errors="coerce")
    for col in ("Sales_Value_EGP", "Units_Sold"):
        monthly[col] = pd.to_numeric(monthly[col], errors="coerce").fillna(0)
    monthly = monthly.dropna(subset=["Month"])
    group_cols = [c for c in ["Project_ID", "District_ID", "Month"] if c in monthly.columns]
    monthly = monthly.groupby(group_cols, as_index=False).agg(
        sales_value_egp=("Sales_Value_EGP", "sum"), units_sold=("Units_Sold", "sum"))
    pcols = [c for c in ["Project_ID", "Project_Name", "District_ID", "Units_Planned"] if c in projects.columns]
    meta = projects[pcols].drop_duplicates("Project_ID")
    if "Units_Planned" not in meta:
        meta["Units_Planned"] = np.nan
    inputs = inputs[[c for c in ["Project_ID", "investment_cost_per_unit_egp", "assumed_cost_ratio_pct", "input_type"] if c in inputs.columns]].drop_duplicates("Project_ID")
    meta = meta.merge(inputs, on="Project_ID", how="left")
    dname = next((c for c in districts.columns if c.lower() in {"district_name", "name", "district"}), None)
    if dname and "District_ID" in districts.columns:
        meta = meta.merge(districts[["District_ID", dname]].drop_duplicates("District_ID"), on="District_ID", how="left")
    else:
        dname = "District_ID"

    if district_id:
        meta = meta[meta["District_ID"].astype(str) == str(district_id)]
    if project_id:
        meta = meta[meta["Project_ID"].astype(str) == str(project_id)]
    monthly = monthly[monthly["Project_ID"].astype(str).isin(meta["Project_ID"].astype(str))]

    rows = []
    for project in meta.to_dict("records"):
        pid = str(project.get("Project_ID"))
        pm = monthly[monthly["Project_ID"].astype(str) == pid].sort_values("Month").copy()
        if pm.empty:
            continue
        pm["cumulative_sales_egp"] = pm["sales_value_egp"].cumsum()
        planned = pd.to_numeric(pd.Series([project.get("Units_Planned")]), errors="coerce").iloc[0]
        cost_unit = pd.to_numeric(pd.Series([project.get("investment_cost_per_unit_egp")]), errors="coerce").iloc[0]
        valid_cost = pd.notna(planned) and pd.notna(cost_unit) and planned > 0 and cost_unit > 0
        total_investment = float(planned * cost_unit) if valid_cost else None
        crossing = pm[pm["cumulative_sales_egp"] >= total_investment] if valid_cost else pm.iloc[0:0]
        be = crossing.iloc[0] if not crossing.empty else None
        history = [{"month": r.Month.strftime("%Y-%m"), "units_sold": int(round(float(r.units_sold))),
                    "sales_value_egp": round(float(r.sales_value_egp), 2),
                    "cumulative_sales_egp": round(float(r.cumulative_sales_egp), 2)} for r in pm.itertuples()]
        last_cum = float(pm["cumulative_sales_egp"].iloc[-1])
        rows.append({
            "project_id": pid, "project": project.get("Project_Name") or pid,
            "district_id": project.get("District_ID"), "district": project.get(dname) or project.get("District_ID"),
            "units_planned": int(planned) if pd.notna(planned) else None,
            "investment_cost_per_unit_egp": round(float(cost_unit), 2) if pd.notna(cost_unit) else None,
            "total_investment_cost_egp": round(total_investment, 2) if total_investment is not None else None,
            "total_sales_value_egp": round(float(pm["sales_value_egp"].sum()), 2),
            "latest_cumulative_sales_egp": round(last_cum, 2),
            "break_even_progress_pct": round(last_cum / total_investment * 100, 2) if total_investment else None,
            "break_even_month": be["Month"].strftime("%Y-%m") if be is not None else None,
            "break_even_cumulative_sales_egp": round(float(be["cumulative_sales_egp"]), 2) if be is not None else None,
            "break_even_status": "reached" if be is not None else ("not_reached" if total_investment else "missing_investment_input"),
            "history": history,
        })
    rows.sort(key=lambda x: (x["break_even_status"] != "reached", x["break_even_month"] or "9999-99", x["project_id"]))
    return {"source": "postgresql", "as_of": monthly["Month"].max().strftime("%Y-%m-%d") if not monthly.empty else None,
            "rows": rows,
            "projects": [{"project_id": r["project_id"], "project": r["project"], "district_id": r["district_id"], "district": r["district"]} for r in rows],
            "method": "Scenario break-even = first observed month cumulative contract value sold >= total assumed investment. Total assumed investment = Units_Planned × synthetic investment cost per unit. The exact crossing month is exact within the supplied monthly dataset, but the threshold is only an estimate because the investment inputs are synthetic.",
            "investment_inputs_are_synthetic": True}

