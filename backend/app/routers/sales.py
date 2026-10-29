"""Database-backed ACUD real-estate analytics endpoints.

The prototype imports the supplied synthetic CSVs into tables prefixed with ``acud_``.
The dashboard reads current aggregates from those tables rather than trusting embedded HTML data.
"""
from __future__ import annotations

import json
import joblib
import os
import tempfile
import re
import sys
from pathlib import Path
from typing import Literal

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
    # The checked-in CSV is the source of truth for scenario assumptions. This is
    # important when per-project cost assumptions are recalibrated: stale but valid
    # PostgreSQL values must not silently overwrite the newer CSV values. Use the
    # database only to fill fields that are missing in the canonical CSV.
    combined = csv_df.set_index("Project_ID")
    db = db_df.set_index("Project_ID")
    for col in ("investment_cost_per_unit_egp", "assumed_cost_ratio_pct", "sell_through_pct", "input_type"):
        if col in db.columns:
            db_values = db[col].reindex(combined.index)
            if col == "investment_cost_per_unit_egp":
                db_values = db_values.where(db_values > 0)
            combined[col] = combined[col].combine_first(db_values)
    return combined.reset_index()


def _monthly_dashboard() -> dict:
    inv = _read("Fact_Inventory_Monthly")
    col = _read("Fact_Collections_Monthly")
    con = _read("Fact_Construction_Monthly")
    util = _read("Fact_Utilities_Monthly")
    projects = _read("Dim_Project")
    districts = _read("Dim_District")
    units = _read("Dim_Unit")

    for frame in (inv, col, con, util):
        frame["Month"] = pd.to_datetime(frame["Month"]).dt.to_period("M").dt.to_timestamp()
    inv_month = inv.groupby("Month").agg(units_sold=("Units_Sold", "sum"), sales_value_egp=("Sales_Value_EGP", "sum"), available_inventory=("Available_End", "sum"))
    col_month = col.groupby("Month")["Amount_Collected_EGP"].sum().rename("collections_egp")
    con_month = con.groupby("Month")["Actual_Cost_EGP"].sum().rename("construction_cost_egp")
    util_month = util.groupby("Month")["Electricity_kWh"].sum().rename("electricity_kwh")
    monthly = inv_month.join(col_month, how="outer").join(con_month, how="outer").join(util_month, how="outer").sort_index().fillna(0)
    # Keep the same recent 24-month window used by the supplied dashboard when available.
    monthly = monthly.tail(24)
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

    # Recompute a transparent seasonal-naive forecast from the database series on each request.
    # This is a baseline, not a causal model; intervals are heuristic and must be back-tested on real data.
    forecast = []
    if len(monthly) >= 12:
        last_month = monthly.index.max()
        for target in TARGETS:
            series = monthly[target].astype(float)
            scale = max(float(series.tail(12).mean()), 1.0)
            for horizon in (3, 6, 12):
                for step in range(1, horizon + 1):
                    future_month = last_month + pd.offsets.MonthBegin(step)
                    prior_month = future_month - pd.DateOffset(years=1)
                    base = float(series.loc[prior_month]) if prior_month in series.index else float(series.iloc[-1])
                    # A wider heuristic band is used for longer horizons.
                    margin = scale * ({3: 0.15, 6: 0.22, 12: 0.30}[horizon])
                    forecast.append({"target": target, "horizon_months": horizon, "Month": future_month.strftime("%Y-%m"),
                                    "yhat": max(0.0, base), "lo": max(0.0, base - margin), "hi": max(0.0, base + margin),
                                    "model": "database_seasonal_naive", "horizon_fold": f"{horizon}m", "mape_backtest": None,
                                    "ml_beats_baseline": False, "usable": True})

    # When live retraining has produced persisted model forecasts, use them for the
    # three primary forecasting targets instead of the seasonal-naive fallback.
    forecast = _apply_persisted_forecasts(forecast)

    # Compute live anomaly scores from the current PostgreSQL monthly aggregates.
    # These are statistical flags, not labelled/ground-truth events.
    anomaly_scores = []
    for target in TARGETS:
        series = monthly[target].astype(float)
        if len(series) < 6:
            continue
        rolling_expected = series.shift(1).rolling(window=min(6, max(3, len(series) - 1)), min_periods=3).median()
        residual = series - rolling_expected
        baseline_residual = residual.dropna()
        med = float(baseline_residual.median()) if len(baseline_residual) else 0.0
        mad = float((baseline_residual - med).abs().median()) if len(baseline_residual) else 0.0
        scale = 1.4826 * mad
        if not np.isfinite(scale) or scale <= 0:
            scale = float(baseline_residual.std(ddof=0)) if len(baseline_residual) else 0.0
        scale = max(scale, abs(float(series.median())) * 0.01, 1.0)
        for month, actual, expected, resid in zip(monthly.index, series, rolling_expected, residual):
            if pd.isna(expected) or pd.isna(resid):
                continue
            score = abs(float(resid) - med) / scale
            anomaly_scores.append({"Month": month.strftime("%Y-%m"), "target": target,
                                   "actual": float(actual), "expected": float(expected),
                                   "residual": float(resid), "score": round(float(score), 3),
                                   "flagged": bool(score >= 3.5)})
    anomaly_scores.sort(key=lambda row: row["score"], reverse=True)

    # Basic live data-quality checks derived from the tables read for this response.
    validation_checks = [
        {"name": "Monthly inventory dates are valid", "pass": bool(inv["Month"].notna().all())},
        {"name": "Monthly collection dates are valid", "pass": bool(col["Month"].notna().all())},
        {"name": "Monthly construction dates are valid", "pass": bool(con["Month"].notna().all())},
        {"name": "Monthly utility dates are valid", "pass": bool(util["Month"].notna().all())},
        {"name": "Inventory sales values are non-negative", "pass": bool((pd.to_numeric(inv["Units_Sold"], errors="coerce").dropna() >= 0).all())},
        {"name": "Inventory available values are non-negative", "pass": bool((pd.to_numeric(inv["Available_End"], errors="coerce").dropna() >= 0).all())},
        {"name": "Project identifiers are unique", "pass": bool(projects["Project_ID"].astype(str).is_unique)},
        {"name": "Unit identifiers are unique", "pass": bool(units["Unit_ID"].astype(str).is_unique)},
    ]
    result = {"source": "postgresql", "generated_at": pd.Timestamp.now(tz="UTC").isoformat(), "targets": TARGETS,
              "months": months, "hist": hist, "summary": summary, "forecast": forecast,
              "anomaly_scores": anomaly_scores, "anomaly_method": "Rolling 6-month median residual; robust MAD score; flag threshold >= 3.5",
              "validation": {"pass": sum(1 for item in validation_checks if item["pass"]),
                             "total": len(validation_checks), "checks": validation_checks},
              "data_status": "synthetic_prototype_data",
              "notes": ["Prototype data is synthetic; results are not evidence about actual ACUD market conditions.",
                        "Monthly values, anomaly scores and baseline forecasts are computed from PostgreSQL tables.",
                        "Anomaly flags are statistical signals; no labelled event ground truth is available.",
                        "Forecast bands are heuristic and are not probabilistic confidence intervals."]}

    # Use the exact same latest successful PostgreSQL model run as /risk/latest.
    try:
        latest_run = _latest_model_run()
        if latest_run:
            risk = dict(latest_run.get("risk_results") or {})
            result["risk_metrics"] = risk.get("metrics", [])
            result["risk_cal"] = risk.get("calibration", [])
            result["risk_proj"] = risk.get("projects", [])
            result["risk_dist"] = risk.get("districts", [])
            result["risk_as_of"] = risk.get("as_of")
            result["risk_train_start"] = risk.get("training_start")
            result["risk_train_end"] = risk.get("training_end")
            result["risk_test_start"] = risk.get("test_start")
            result["risk_test_end"] = risk.get("test_end")
            result["risk_run_id"] = latest_run["run_id"]
            completed = latest_run.get("completed_at")
            result["risk_run_timestamp"] = completed.isoformat() if hasattr(completed, "isoformat") else str(completed)
            result["risk_data_timestamp"] = str(latest_run.get("data_timestamp"))
            result["risk_source"] = "postgresql"
        else:
            result["risk_metrics"] = []
            result["risk_cal"] = []
            result["risk_proj"] = []
            result["risk_dist"] = []
            result["risk_source"] = "no_completed_postgresql_run"
    except Exception as exc:
        # Do not fall back to the old fixed-date table: report missing DB run data.
        result["risk_metrics"] = []
        result["risk_cal"] = []
        result["risk_proj"] = []
        result["risk_dist"] = []
        result["risk_source"] = "postgresql_unavailable"
        result["risk_error"] = f"{type(exc).__name__}: {exc}"
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

# The shared risk-model module lives at the repository root beside backend/.
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
from nac_ml import risk as risk_models  # noqa: E402

_TRAIN_LOCK = threading.Lock()
MODEL_DIR = OUTPUT_DIR / "retrained_models"
TRAINING_RESULTS_PATH = MODEL_DIR / "training_results.json"
RETRAIN_TARGETS = {
    "units_sold": {"table": "Fact_Inventory_Monthly", "column": "Units_Sold"},
    "sales_value_egp": {"table": "Fact_Inventory_Monthly", "column": "Sales_Value_EGP"},
    "collections_egp": {"table": "Fact_Collections_Monthly", "column": "Amount_Collected_EGP"},
}
_TRAINING = {"status": "idle", "progress": 0, "message": "No retraining run yet", "started_at": None,
             "completed_at": None, "data_timestamp": None, "metrics": None, "before_metrics": None,
             "after_metrics": None, "target_results": None, "risk_results": None, "anomaly_count": None, "error": None, "forecast": None}


# PostgreSQL is the source of truth for completed model runs. JSON files remain
# optional audit exports only; API reads never fall back to them.
MODEL_RUNS_TABLE = "acud_sales_model_runs"

def _ensure_model_runs_table() -> None:
    with engine.begin() as conn:
        conn.execute(text(f"""
            CREATE TABLE IF NOT EXISTS {MODEL_RUNS_TABLE} (
                run_id BIGSERIAL PRIMARY KEY,
                status TEXT NOT NULL CHECK (status IN ('completed')),
                started_at TIMESTAMPTZ NOT NULL,
                completed_at TIMESTAMPTZ NOT NULL,
                data_timestamp TIMESTAMPTZ,
                risk_as_of DATE,
                risk_trained_through_as_of DATE,
                training_start DATE,
                training_end DATE,
                test_start DATE,
                test_end DATE,
                target_results JSONB NOT NULL,
                risk_results JSONB NOT NULL,
                run_payload JSONB NOT NULL
            )
        """))
        conn.execute(text(f"CREATE INDEX IF NOT EXISTS ix_{MODEL_RUNS_TABLE}_completed_at ON {MODEL_RUNS_TABLE} (completed_at DESC)"))

def _latest_model_run() -> dict | None:
    _ensure_model_runs_table()
    with engine.connect() as conn:
        row = conn.execute(text(f"""SELECT run_id, status, started_at, completed_at, data_timestamp,
            target_results, risk_results, run_payload FROM {MODEL_RUNS_TABLE}
            WHERE status='completed' ORDER BY completed_at DESC, run_id DESC LIMIT 1""")).mappings().first()
    if not row:
        return None
    # psycopg normally decodes JSONB into dictionaries; tolerate string values too.
    def decode(value):
        return json.loads(value) if isinstance(value, str) else value
    return {**dict(row), "target_results": decode(row["target_results"]),
            "risk_results": decode(row["risk_results"]), "run_payload": decode(row["run_payload"])}

def _save_model_run(payload: dict) -> int:
    _ensure_model_runs_table()
    risk = payload.get("risk_results") or {}
    def date_or_none(value):
        return value if value else None
    with engine.begin() as conn:
        run_id = conn.execute(text(f"""INSERT INTO {MODEL_RUNS_TABLE}
            (status, started_at, completed_at, data_timestamp, risk_as_of,
             risk_trained_through_as_of, training_start, training_end, test_start, test_end,
             target_results, risk_results, run_payload)
            VALUES ('completed', :started_at, :completed_at, :data_timestamp, :risk_as_of,
             :risk_trained_through_as_of, :training_start, :training_end, :test_start, :test_end,
             CAST(:target_results AS JSONB), CAST(:risk_results AS JSONB), CAST(:run_payload AS JSONB))
            RETURNING run_id"""), {
            "started_at": payload["started_at"], "completed_at": payload["completed_at"],
            "data_timestamp": date_or_none(payload.get("data_timestamp")),
            "risk_as_of": date_or_none(risk.get("as_of")),
            "risk_trained_through_as_of": date_or_none(risk.get("trained_through_as_of")),
            "training_start": date_or_none(risk.get("training_start")),
            "training_end": date_or_none(risk.get("training_end")),
            "test_start": date_or_none(risk.get("test_start")),
            "test_end": date_or_none(risk.get("test_end")),
            "target_results": json.dumps(payload.get("target_results") or {}, default=str),
            "risk_results": json.dumps(risk, default=str),
            "run_payload": json.dumps(payload, default=str),
        }).scalar_one()
    return int(run_id)


def _feature_matrix(indices):
    """Small, reproducible time-series feature set: trend plus annual Fourier terms."""
    idx = np.asarray(indices, dtype=float)
    return np.column_stack([idx, np.sin(2 * np.pi * idx / 12), np.cos(2 * np.pi * idx / 12)])


def _metric_pack(actual, predicted):
    a = np.asarray(actual, dtype=float)
    p = np.asarray(predicted, dtype=float)
    return {
        "MAE": float(mean_absolute_error(a, p)),
        "RMSE": float(np.sqrt(mean_squared_error(a, p))),
        "WAPE_pct": float(100 * np.abs(a - p).sum() / max(np.abs(a).sum(), 1e-9)),
    }


def _fit_target_metrics(series: pd.Series):
    """Evaluate with a chronological holdout; never randomly shuffle time series."""
    y = pd.to_numeric(series, errors="coerce").dropna().astype(float)
    if len(y) < 8:
        raise ValueError(f"At least 8 monthly observations are required; found {len(y)}.")
    test_n = min(3, max(1, len(y) // 5))
    train, actual = y.iloc[:-test_n], y.iloc[-test_n:]
    baseline = np.repeat(float(train.iloc[-1]), len(actual))
    model = make_pipeline(StandardScaler(), Ridge(alpha=2.0))
    model.fit(_feature_matrix(np.arange(len(train))), np.log1p(np.maximum(train.to_numpy(), 0)))
    pred = np.maximum(0, np.expm1(model.predict(_feature_matrix(np.arange(len(train), len(y))))) )
    holdout = [{"Month": pd.Timestamp(month).strftime("%Y-%m"), "actual": float(a),
                "baseline": float(b), "retrained": float(p)}
               for month, a, b, p in zip(actual.index, actual.to_numpy(), baseline, pred)]
    return _metric_pack(actual.to_numpy(), baseline), _metric_pack(actual.to_numpy(), pred), y, holdout


def _load_retraining_series() -> tuple[dict[str, pd.Series], str]:
    """Read fresh database records and aggregate each target into continuous monthly series."""
    inventory = _read("Fact_Inventory_Monthly")
    collections = _read("Fact_Collections_Monthly")
    if "Month" not in inventory or "Month" not in collections:
        raise ValueError("Sales and collections tables must both contain a Month column.")
    inventory["Month"] = pd.to_datetime(inventory["Month"], errors="coerce").dt.to_period("M").dt.to_timestamp()
    collections["Month"] = pd.to_datetime(collections["Month"], errors="coerce").dt.to_period("M").dt.to_timestamp()
    inventory = inventory.dropna(subset=["Month"])
    collections = collections.dropna(subset=["Month"])
    inv_month = inventory.groupby("Month").agg(
        units_sold=("Units_Sold", "sum"), sales_value_egp=("Sales_Value_EGP", "sum"))
    col_month = collections.groupby("Month")["Amount_Collected_EGP"].sum().rename("collections_egp")
    all_months = inv_month.index.union(col_month.index).sort_values()
    if len(all_months) == 0:
        raise ValueError("No monthly sales or collections records were found.")
    calendar = pd.date_range(all_months.min(), all_months.max(), freq="MS")
    combined = inv_month.join(col_month, how="outer").reindex(calendar).fillna(0).sort_index()
    series_map = {target: combined[target].astype(float) for target in RETRAIN_TARGETS}
    timestamp = pd.Timestamp(calendar.max()).isoformat()
    return series_map, timestamp


def _apply_persisted_forecasts(fallback_forecast: list[dict]) -> list[dict]:
    """Replace baseline forecast rows for retrained targets when persisted results exist."""
    try:
        latest_run = _latest_model_run()
        if not latest_run:
            return fallback_forecast
        saved = latest_run["run_payload"]
        target_results = latest_run.get("target_results") or {}
        out = [row for row in fallback_forecast if row.get("target") not in target_results]
        for target, details in target_results.items():
            future = details.get("forecast") or []
            after = details.get("after_metrics") or {}
            before = details.get("before_metrics") or {}
            beats = after.get("WAPE_pct") is not None and before.get("WAPE_pct") is not None and after["WAPE_pct"] < before["WAPE_pct"]
            for horizon in (3, 6, 12):
                for row in future[:horizon]:
                    out.append({"target": target, "horizon_months": horizon, "Month": row["Month"],
                                "yhat": row["yhat"], "lo": row["lo"], "hi": row["hi"],
                                "model": details.get("model", "ridge_trend_fourier_retrained"),
                                "horizon_fold": f"{horizon}m", "mape_backtest": after.get("WAPE_pct"),
                                "ml_beats_baseline": bool(beats), "usable": True,
                                "trained_at": saved.get("completed_at")})
        return out
    except Exception as exc:
        # A corrupt or stale artifact should not take the live dashboard down.
        print(f"Could not apply persisted forecasts: {type(exc).__name__}: {exc}")
        return fallback_forecast


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

    # Fit production classifiers on all label-complete dates, then score current listed units.
    # Final production fit uses every label-complete date, including the holdout and
    # the purge-gap dates after evaluation has been recorded.
    all_rows = risk_models.build(ctx, dates)
    all_rows.loc[:, risk_models.FEATURES] = all_rows[risk_models.FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0)
    for name in ("logistic", "gbm"):
        estimator = models[name]
        estimator.fit(all_rows[risk_models.FEATURES], all_rows.y_unsold.astype(int))
        fitted_final[name] = estimator

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
    current_features["p_unsold_90d"] = current_features["p_unsold_90d_logistic"].clip(0, 1)
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
    risk_models.calibration(test_preds.assign(p_logistic=test_preds["p_logistic"]), "p_logistic").to_csv(
        model_dir / "unsold_risk_calibration_logistic.csv", index=False)
    current_features.to_csv(model_dir / "unsold_risk_scores_latest.csv", index=False)

    metric_frame = pd.DataFrame(metrics)
    cal = risk_models.calibration(test_preds.assign(p_logistic=test_preds["p_logistic"]), "p_logistic")
    proj = current_features.groupby(["Project_ID", "District_ID"]).agg(
        units_on_market=("Unit_ID", "size"), expected_unsold_90d=("p_unsold_90d", "sum"),
        mean_p_unsold=("p_unsold_90d", "mean")).reset_index()
    proj["expected_sold_90d"] = proj.units_on_market - proj.expected_unsold_90d
    dist = current_features.groupby("District_ID").agg(units=("Unit_ID", "size"), mean_p=("p_unsold_90d", "mean"),
        expected_unsold=("p_unsold_90d", "sum")).reset_index()
    high_counts = current_features.assign(_high=current_features.p_unsold_90d >= 0.92).groupby("District_ID")["_high"].sum()
    dist["high"] = dist.District_ID.map(high_counts).fillna(0).astype(int)
    return {
        "as_of": as_of.strftime("%Y-%m-%d"), "trained_through_as_of": latest_label_asof.strftime("%Y-%m-%d"),
        "training_start": pd.Timestamp(train.as_of.min()).strftime("%Y-%m-%d"),
        "training_end": pd.Timestamp(train.as_of.max()).strftime("%Y-%m-%d"),
        "test_start": pd.Timestamp(test.as_of.min()).strftime("%Y-%m-%d"),
        "test_end": pd.Timestamp(test.as_of.max()).strftime("%Y-%m-%d"),
        "n_train": int(len(train)), "n_test": int(len(test)), "n_current_units": int(len(current_features)),
        "unsold_rate_test": round(float(test.y_unsold.mean()), 4), "metrics": metric_frame.to_dict("records"),
        "calibration": cal.to_dict("records"),
        "projects": proj.sort_values("expected_unsold_90d", ascending=False).round(4).to_dict("records"),
        "districts": dist.round(4).to_dict("records"),
        "artifacts": {name: str((model_dir / f"unsold_risk_{name}.joblib").relative_to(OUTPUT_DIR.parent)) for name in fitted_final},
        "model_directory": str(model_dir),
    }


def _run_retraining():
    try:
        with _TRAIN_LOCK:
            _TRAINING.update(status="running", progress=5, message="Loading latest sales and collections data",
                             started_at=datetime.now(timezone.utc).isoformat(), completed_at=None, error=None,
                             target_results=None, risk_results=None, forecast=None)
        series_map, timestamp = _load_retraining_series()
        with _TRAIN_LOCK:
            _TRAINING.update(progress=15, data_timestamp=timestamp, message="Evaluating 3 targets on chronological holdouts")

        target_results = {}
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        for i, (target, series) in enumerate(series_map.items()):
            before, after, y, holdout = _fit_target_metrics(series)
            # Refit on every available month for production forecasts and persist the fitted estimator.
            model = make_pipeline(StandardScaler(), Ridge(alpha=2.0))
            model.fit(_feature_matrix(np.arange(len(y))), np.log1p(np.maximum(y.to_numpy(), 0)))
            future_idx = np.arange(len(y), len(y) + 12)
            predictions = np.maximum(0, np.expm1(model.predict(_feature_matrix(future_idx))))
            residual = max(float(after["MAE"]), float(np.nanmedian(np.abs(y.to_numpy()))) * 0.03, 1.0)
            last = pd.Timestamp(y.index.max())
            forecast = [{"Month": (last + pd.offsets.MonthBegin(step + 1)).strftime("%Y-%m"),
                         "yhat": round(float(value), 2),
                         "lo": round(max(0.0, float(value) - residual), 2),
                         "hi": round(float(value) + residual, 2), "target": target}
                        for step, value in enumerate(predictions)]
            artifact_path = MODEL_DIR / f"forecast_{target}.joblib"
            fd, temp_path = tempfile.mkstemp(prefix=artifact_path.stem, suffix=".tmp", dir=str(MODEL_DIR))
            os.close(fd)
            try:
                joblib.dump({"model": model, "target": target, "observations": len(y),
                             "last_observed_month": last.strftime("%Y-%m"), "features": "trend + annual Fourier terms"}, temp_path)
                os.replace(temp_path, artifact_path)
            finally:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
            target_results[target] = {
                "label": TARGETS.get(target, target), "model": "Ridge trend + annual Fourier terms",
                "artifact": str(artifact_path.relative_to(MODEL_DIR.parent)),
                "observations": int(len(y)), "before_metrics": before, "after_metrics": after,
                "holdout": holdout, "forecast": forecast,
            }
            with _TRAIN_LOCK:
                _TRAINING.update(progress=15 + int(70 * (i + 1) / len(series_map)),
                                 message=f"Trained and evaluated {i + 1} of {len(series_map)} targets")

        with _TRAIN_LOCK:
            _TRAINING.update(progress=88, message="Retraining 90-day unsold-risk baselines, logistic regression, and gradient boosting")
        risk_results = _train_unsold_risk()
        with _TRAIN_LOCK:
            _TRAINING.update(progress=97, message="Saving forecasting and unsold-risk model results", risk_results=risk_results)

        units = series_map["units_sold"]
        changes = units.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
        anomalies = 0
        if len(changes):
            med = changes.median()
            mad = (changes - med).abs().median()
            scale = 1.4826 * mad if mad > 0 else (changes.std(ddof=0) or 1.0)
            anomalies = int(((changes - med).abs() / scale >= 3.5).sum())

        completed_at = datetime.now(timezone.utc).isoformat()
        saved = {"status": "completed", "started_at": _TRAINING.get("started_at"), "completed_at": completed_at,
                 "data_timestamp": timestamp, "target_results": target_results, "risk_results": risk_results,
                 "anomaly_count": anomalies, "model_directory": str(MODEL_DIR), "source": "postgresql"}
        # Commit the complete successful run to PostgreSQL first. A failed DB write
        # makes the run fail rather than presenting local JSON as the latest run.
        run_id = _save_model_run(saved)
        saved["run_id"] = run_id
        risk_results["run_id"] = run_id
        risk_results["run_timestamp"] = completed_at
        risk_results["data_timestamp"] = timestamp
        risk_results["source"] = "postgresql"
        saved["risk_results"] = risk_results
        # Optional local audit mirror; API endpoints do not use this file as source.
        fd, temp_path = tempfile.mkstemp(prefix="training_results", suffix=".tmp", dir=str(MODEL_DIR))
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(saved, handle, ensure_ascii=False, indent=2)
            os.replace(temp_path, TRAINING_RESULTS_PATH)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

        primary = target_results["units_sold"]
        with _TRAIN_LOCK:
            _TRAINING.update(status="completed", progress=100, message="All 3 forecasting targets and 90-day unsold-risk models trained and saved",
                             completed_at=completed_at, before_metrics=primary["before_metrics"],
                             after_metrics=primary["after_metrics"], metrics=primary["after_metrics"],
                             target_results=target_results, risk_results=risk_results, anomaly_count=anomalies,
                             forecast=primary["forecast"], model=primary["model"], observations=primary["observations"],
                             model_directory=str(MODEL_DIR), persisted=True)
    except Exception as exc:
        with _TRAIN_LOCK:
            _TRAINING.update(status="error", progress=100, message="Retraining failed",
                             completed_at=datetime.now(timezone.utc).isoformat(), error=f"{type(exc).__name__}: {exc}")


@router.get("/risk/latest")
def latest_unsold_risk_results():
    """Return risk results from the newest successful PostgreSQL model run."""
    try:
        latest = _latest_model_run()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Could not read model runs from PostgreSQL: {type(exc).__name__}: {exc}") from exc
    if not latest:
        raise HTTPException(status_code=404, detail="No successful PostgreSQL model run exists yet. Run POST /retrain first.")
    risk = dict(latest.get("risk_results") or {})
    risk.update({"run_id": latest["run_id"], "run_timestamp": latest["completed_at"].isoformat() if hasattr(latest["completed_at"], "isoformat") else str(latest["completed_at"]),
                 "data_timestamp": latest["data_timestamp"].isoformat() if hasattr(latest["data_timestamp"], "isoformat") else str(latest["data_timestamp"]),
                 "source": "postgresql", "status": "completed"})
    return risk


@router.post("/retrain")
def retrain_latest():
    """Retrain three monthly forecasts plus 90-day unsold-risk baselines/classifiers."""
    with _TRAIN_LOCK:
        if _TRAINING["status"] in {"queued", "running"}:
            return {"accepted": True, **_TRAINING}
        _TRAINING.update(status="queued", progress=0, message="Retraining queued", error=None)
    thread = threading.Thread(target=_run_retraining, daemon=True)
    thread.start()
    return {"accepted": True, "status": "queued", "progress": 0, "message": "Retraining queued"}


@router.get("/retrain/status")
def retraining_status():
    """Return active job progress or metadata for the newest completed PostgreSQL run."""
    with _TRAIN_LOCK:
        current = dict(_TRAINING)
    if current.get("status") in {"queued", "running"}:
        return current
    try:
        latest = _latest_model_run()
    except Exception as exc:
        current.update(status="error", error=f"PostgreSQL model-run lookup failed: {type(exc).__name__}: {exc}")
        return current
    if not latest:
        return current
    payload = latest.get("run_payload") or {}
    targets = latest.get("target_results") or {}
    units = targets.get("units_sold") or {}
    current.update(status="completed", progress=100, message="Loaded latest successful PostgreSQL model run",
                   run_id=latest["run_id"], started_at=payload.get("started_at"),
                   completed_at=payload.get("completed_at"), data_timestamp=payload.get("data_timestamp"),
                   target_results=targets, risk_results=latest.get("risk_results"),
                   anomaly_count=payload.get("anomaly_count"), forecast=units.get("forecast"),
                   before_metrics=units.get("before_metrics"), after_metrics=units.get("after_metrics"),
                   metrics=units.get("after_metrics"), model=units.get("model"),
                   observations=units.get("observations"), model_directory=payload.get("model_directory"),
                   persisted=True, source="postgresql")
    return current


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

    agg["forecast_revenue_egp"] = pd.to_numeric(agg["avg_monthly_sales_egp"], errors="coerce").fillna(0) * horizon_months
    agg["forecast_units"] = pd.to_numeric(agg["avg_monthly_units_sold"], errors="coerce").fillna(0) * horizon_months
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
        avg_monthly_sales = float(r.get("avg_monthly_sales_egp") or 0)
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
            "method": "Aligned project-level scenario: total project investment = planned units × assumed investment cost per unit (same threshold as /equilibrium). Projected cumulative sales = all historical recorded contract value + forecast contract value for the selected horizon. Project ROI proxy (%) = (projected cumulative sales − total project investment) / total project investment × 100; recovery (%) = projected cumulative sales / total project investment × 100. This is a sales-versus-investment scenario, not accounting profit or audited ROI. Forecast uses the mean of the latest three recorded months; projected break-even month extrapolates that average and is not guaranteed.",
            "data_status": "synthetic_scenario_assumptions"}


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
