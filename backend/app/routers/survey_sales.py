"""Database-backed survey and sales intelligence endpoints for the ACUD prototype."""
from __future__ import annotations
import json
from typing import Literal

import cohere
import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy import inspect, text

from app.config import get_settings
from app.database import engine
from app.routers.sales import _read
from app.analytics.executive import executive_brief

router = APIRouter(prefix="/api/v1/survey-sales", tags=["Survey + sales intelligence"])
TABLE = "acud_survey_responses"
SERVICE_NAMES = {
    "transport": "المواصلات", "healthcare": "الخدمات الطبية", "education": "التعليم",
    "security": "الأمن", "shopping": "التسوق", "cleanliness": "النظافة",
    "roads_lighting": "الطرق والإنارة", "internet": "الإنترنت", "utilities": "المرافق",
    "recreation": "الترفيه",
}


def _survey() -> pd.DataFrame:
    if not inspect(engine).has_table(TABLE):
        raise HTTPException(status_code=503, detail="Survey table is missing. Run `python seed_survey_responses.py` from the backend folder after seeding the sales tables.")
    with engine.connect() as conn:
        df = pd.read_sql(text(f'SELECT * FROM "{TABLE}"'), conn)
    if "is_eligible" in df:
        df["is_eligible"] = df["is_eligible"].astype(str).str.lower().isin(["true", "1", "yes"])
    for key in ["district_id", "project_id"]:
        if key in df:
            df[key] = df[key].astype(str).str.strip().str.upper()
    return df


def _records(df: pd.DataFrame) -> list[dict]:
    return df.astype(object).where(pd.notna(df), None).to_dict("records")


def _decision_signal(frame: pd.DataFrame) -> dict:
    """Summarize reported decision influence without treating it as observed conversion."""
    col = "q11_problem_changed_decision"
    if col not in frame:
        return {"valid_responses": 0, "affected_count": 0, "affected_pct": None, "outcome_changed_count": 0, "outcome_changed_pct": None}
    values = frame[col].dropna().astype(str).str.strip()
    valid = values[~values.str.contains("لا ينطبق", na=False)]
    affected = valid.str.startswith("نعم")
    outcome_changed = valid.str.contains("أجل القرار|منطقة أخرى", regex=True, na=False)
    return {
        "valid_responses": int(len(valid)),
        "affected_count": int(affected.sum()),
        "affected_pct": round(float(affected.mean() * 100), 1) if len(valid) else None,
        "outcome_changed_count": int(outcome_changed.sum()),
        "outcome_changed_pct": round(float(outcome_changed.mean() * 100), 1) if len(valid) else None,
    }


def _latest_sales(district_id: str | None = None, project_id: str | None = None):
    inv = _read("Fact_Inventory_Monthly").copy()
    inv["Month"] = pd.to_datetime(inv["Month"], errors="coerce")
    if district_id:
        inv = inv[inv["District_ID"].astype(str).str.upper() == district_id.upper()]
    if project_id:
        inv = inv[inv["Project_ID"].astype(str).str.upper() == project_id.upper()]
    latest = inv["Month"].max() if not inv.empty else pd.NaT
    current = inv[inv["Month"] == latest].copy() if pd.notna(latest) else inv.copy()
    proj = current.groupby(["Project_ID", "District_ID"], as_index=False).agg(
        units_sold=("Units_Sold", "sum"), sales_value_egp=("Sales_Value_EGP", "sum"), available_inventory=("Available_End", "sum")
    ) if not current.empty else pd.DataFrame(columns=["Project_ID", "District_ID", "units_sold", "sales_value_egp", "available_inventory"])
    try:
        dim_projects = _read("Dim_Project")
        proj = proj.merge(dim_projects[["Project_ID", "Project_Name", "District_ID"]].drop_duplicates("Project_ID"), on=["Project_ID", "District_ID"], how="left")
    except Exception:
        pass
    try:
        dim_districts = _read("Dim_District")
        if not proj.empty:
            proj = proj.merge(dim_districts[["District_ID", "District_Name"]].drop_duplicates("District_ID"), on="District_ID", how="left")
    except Exception:
        pass
    dist = proj.groupby("District_ID", as_index=False).agg(units_sold=("units_sold", "sum"), sales_value_egp=("sales_value_egp", "sum"), available_inventory=("available_inventory", "sum")) if not proj.empty else pd.DataFrame(columns=["District_ID", "units_sold", "sales_value_egp", "available_inventory"])
    try:
        dim_districts = _read("Dim_District")
        dist = dist.merge(dim_districts[["District_ID", "District_Name"]].drop_duplicates("District_ID"), on="District_ID", how="left")
    except Exception:
        pass
    return (str(latest.date()) if pd.notna(latest) else None, proj, dist)


@router.get("/options")
def options():
    """Return real district and project keys from the seeded sales dimensions."""
    try:
        districts = _read("Dim_District")
        projects = _read("Dim_Project")
    except HTTPException:
        raise
    return {
        "districts": _records(districts[["District_ID", "District_Name"]].drop_duplicates().sort_values("District_ID")),
        "projects": _records(projects[["Project_ID", "Project_Name", "District_ID"]].drop_duplicates().sort_values("Project_ID")),
        "region_note": "The supplied source dimensions contain District_ID/District_Name but no separate Region_ID/Region_Name dimension. No region names have been inferred.",
    }


@router.get("/dashboard")
def dashboard(
    district_id: str | None = Query(default=None, min_length=2, max_length=32),
    project_id: str | None = Query(default=None, min_length=2, max_length=32),
):
    df = _survey()
    required_link_keys = {"district_id", "project_id"}
    if not required_link_keys.issubset(df.columns):
        raise HTTPException(status_code=409, detail="Survey table is an older version without district_id/project_id. Run `python seed_survey_responses.py` to replace it with the linked 1,000-response demo dataset.")
    eligible = df[df["is_eligible"]].copy() if "is_eligible" in df else df.copy()
    survey = eligible.copy()
    if district_id:
        survey = survey[survey["district_id"] == district_id.upper()]
    if project_id:
        survey = survey[survey["project_id"] == project_id.upper()]

    def counts(column: str, split: bool = False):
        if column not in survey:
            return []
        values = survey[column].dropna().astype(str)
        if split:
            values = values[values.str.strip().ne("")].str.split(r"\s*[;；]\s*").explode()
        denominator = len(values)
        return [{"label": str(k), "count": int(v), "share_pct": round(float(v / denominator * 100), 1) if denominator else None} for k, v in values.value_counts().head(12).items()]

    service_gaps = []
    for key, label in SERVICE_NAMES.items():
        imp, level = f"q8_importance_{key}_1_5", f"q9_current_level_{key}_1_5"
        if imp in survey and level in survey:
            importance = pd.to_numeric(survey[imp], errors="coerce")
            current = pd.to_numeric(survey[level], errors="coerce")
            valid = importance.notna() & current.notna()
            if valid.any():
                service_gaps.append({"service": label, "importance": round(float(importance[valid].mean()), 2), "current_level": round(float(current[valid].mean()), 2), "gap": round(float((importance[valid] - current[valid]).mean()), 2), "valid_responses": int(valid.sum()), "coverage_pct": round(float(valid.mean() * 100), 1)})
    service_gaps.sort(key=lambda x: x["gap"], reverse=True)
    decision_signal = _decision_signal(survey)

    as_of, sales_projects, sales_districts = _latest_sales(district_id, project_id)
    # District-level complaint/sales comparison uses the same District_ID on both tables.
    linked_districts = []
    for did, group in survey.groupby("district_id", dropna=False):
        if pd.isna(did):
            continue
        top = group.get("q5_most_impactful_problem", pd.Series(dtype=str)).dropna().astype(str).value_counts()
        group_decision = _decision_signal(group)
        sales_row = sales_districts[sales_districts["District_ID"].astype(str).str.upper() == str(did).upper()]
        linked_districts.append({
            "district_id": str(did),
            "district_name": str(group["district_name"].dropna().iloc[0]) if "district_name" in group and group["district_name"].notna().any() else str(did),
            "survey_responses": int(len(group)),
            "top_complaint": str(top.index[0]) if len(top) else None,
            "top_complaint_count": int(top.iloc[0]) if len(top) else 0,
            "top_complaint_share_pct": round(float(top.iloc[0] / top.sum() * 100), 1) if len(top) and top.sum() else None,
            "average_attractiveness": round(float(pd.to_numeric(group.get("q13_area_attractiveness_1_5"), errors="coerce").mean()), 2) if "q13_area_attractiveness_1_5" in group else None,
            "decision_signal": group_decision,
            "sales": _records(sales_row.head(1))[0] if not sales_row.empty else None,
        })
    linked_districts.sort(key=lambda x: x["district_id"])

    linked_projects = []
    for pid, group in survey.groupby("project_id", dropna=False):
        if pd.isna(pid):
            continue
        top = group.get("q5_most_impactful_problem", pd.Series(dtype=str)).dropna().astype(str).value_counts()
        group_decision = _decision_signal(group)
        sales_row = sales_projects[sales_projects["Project_ID"].astype(str).str.upper() == str(pid).upper()]
        linked_projects.append({
            "project_id": str(pid),
            "project_name": str(group["project_name"].dropna().iloc[0]) if "project_name" in group and group["project_name"].notna().any() else str(pid),
            "district_id": str(group["district_id"].dropna().iloc[0]) if group["district_id"].notna().any() else None,
            "survey_responses": int(len(group)),
            "top_complaint": str(top.index[0]) if len(top) else None,
            "top_complaint_count": int(top.iloc[0]) if len(top) else 0,
            "top_complaint_share_pct": round(float(top.iloc[0] / top.sum() * 100), 1) if len(top) and top.sum() else None,
            "average_attractiveness": round(float(pd.to_numeric(group.get("q13_area_attractiveness_1_5"), errors="coerce").mean()), 2) if "q13_area_attractiveness_1_5" in group else None,
            "decision_signal": group_decision,
            "sales": _records(sales_row.head(1))[0] if not sales_row.empty else None,
        })
    linked_projects.sort(key=lambda x: x["project_id"])

    result = {
        "data_status": "synthetic_linked_demo",
        "source": "PostgreSQL tables acud_survey_responses + acud_dim_* + acud_fact_inventory_monthly",
        "survey": {
            "responses": int(len(survey)),
            "eligible_responses": int(len(survey)),
            "total_rows": int(len(df)),
            "screened_out": int(len(df) - len(eligible)),
            "districts": counts("district_name"),
            "relationship_to_area": counts("q1_relationship_to_area"),
            "location_label": counts("q2_location_label"),
            "relationship_duration": counts("q3_relationship_duration"),
            "top_problems": counts("q5_most_impactful_problem"),
            "recent_problems": counts("q4_recent_service_problems", True),
            "problem_frequency": counts("q6_problem_frequency"),
            "problem_severity": counts("q7_problem_severity"),
            "decision_factors": counts("q10_top_3_decision_factors", True),
            "priority_issues": counts("q15_top_improvement_priority"),
            "recommendation": counts("q14_recommend_area"),
            "decision_impact": counts("q11_problem_changed_decision"),
            "decision_signal": decision_signal,
            "attractiveness": round(float(pd.to_numeric(survey["q13_area_attractiveness_1_5"], errors="coerce").mean()), 2) if "q13_area_attractiveness_1_5" in survey and len(survey) else None,
            "attractiveness_distribution": counts("q13_area_attractiveness_1_5"),
            "service_gaps": service_gaps,
        },
        "sales": {"as_of": as_of, "districts": _records(sales_districts.sort_values("units_sold")) if not sales_districts.empty else [], "projects": _records(sales_projects.sort_values("units_sold").head(40)) if not sales_projects.empty else []},
        "filters": {"district_id": district_id, "project_id": project_id},
        "linked_comparison": {"districts": linked_districts, "projects": linked_projects},
        "region_note": "The source dataset has no separate Region_ID/Region_Name. Survey rows are linked directly to the existing District_ID and Project_ID. Add an authoritative region dimension if regional rollups are required.",
        "interpretation_note": "All 1,000 survey responses in this prototype are synthetic and assigned to existing synthetic project/district keys. Complaint/sales associations demonstrate the workflow only; they are not real customer evidence and do not prove causation.",
    }
    return result


class SurveySalesQuestion(BaseModel):
    question: str = Field(min_length=2, max_length=3000)
    language: Literal["en", "ar"] = "en"
    district_id: str | None = None
    project_id: str | None = None


@router.post("/ask")
def ask(payload: SurveySalesQuestion):
    settings = get_settings()
    if not settings.cohere_api_key:
        raise HTTPException(status_code=503, detail="Cohere is not configured. Add COHERE_API_KEY to backend/.env.")
    evidence = dashboard(payload.district_id, payload.project_id)
    system_prompt = (
        "You are ACUD's evidence-grounded survey and real-estate sales analyst. Answer in the requested language. "
        "Use only supplied dashboard data. Give 3 prioritized recommendations with owner, action, metric and review window. "
        "Separate observed facts from hypotheses; never claim causation from correlation. The dataset uses District_ID and Project_ID "
        "from the supplied sales dimensions; there is no separate Region_ID/Region_Name. All 1,000 survey responses are synthetic "
        "demonstration data, not real customer evidence. Do not invent metrics; state if data is insufficient."
    )
    try:
        client = cohere.ClientV2(api_key=settings.cohere_api_key)
        response = client.chat(
            model=settings.cohere_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": "Evidence JSON:\n" + json.dumps(evidence, ensure_ascii=False, default=str) + "\nQuestion: " + payload.question + "\nLanguage: " + payload.language},
            ],
            temperature=0.2,
            max_tokens=1400,
        )
        parts = getattr(response.message, "content", []) or []
        reply = "\n".join(part.text for part in parts if getattr(part, "text", None)).strip()
        if not reply:
            raise RuntimeError("Empty Cohere response")
    except Exception as exc:
        raise HTTPException(status_code=502, detail="Cohere assistant is temporarily unavailable.") from exc
    return {"reply": reply, "provider": "cohere", "model": settings.cohere_model, "data_status": evidence.get("data_status"), "evidence_source": "survey_and_sales_database"}


def _read_optional_table(name: str) -> pd.DataFrame:
    """Read an optional analytics source without breaking the executive endpoint."""
    try:
        result = _read(name)
        return result if isinstance(result, pd.DataFrame) else pd.DataFrame(result)
    except Exception:
        return pd.DataFrame()


def _executive_json_safe(value):
    """Convert pandas/numpy values into JSON-compatible Python values."""
    if isinstance(value, dict):
        return {str(k): _executive_json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_executive_json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if pd.isna(value):
        return None
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    return value


@router.get("/executive")
def executive(
    district_id: str | None = Query(default=None, min_length=2, max_length=32),
    project_id: str | None = Query(default=None, min_length=2, max_length=32),
    sales_position: Literal["weaker", "stronger"] | None = Query(default=None),
):
    """
    Executive resident-service analysis.

    Uses the service-gap analytics pipeline and PostgreSQL-backed source tables.
    The portfolio-wide sales-position comparison is calculated before applying
    the requested district/sales-position display filters.
    """
    # 1. Load source data.
    survey = _survey()

    try:
        districts = _read("Dim_District")
        projects = _read("Dim_Project")
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail="District/project dimensions could not be loaded from PostgreSQL.",
        ) from exc

    indicators = _read_optional_table("Fact_District_Service_Indicators")
    status = _read_optional_table("Fact_District_Monthly_Status")
    funnel = _read_optional_table("Fact_Project_Sales_Funnel")

    # 2. Validate the requested project filter against the actual dimension.
    if project_id:
        project_keys = (
            projects["Project_ID"].astype(str).str.strip().str.upper()
            if "Project_ID" in projects
            else pd.Series(dtype=str)
        )
        project_key = project_id.strip().upper()

        if project_key not in set(project_keys):
            raise HTTPException(status_code=404, detail="Unknown project_id.")

        # A project filter narrows the respondent-level analysis.
        survey = survey[
            survey["project_id"].astype(str).str.upper() == project_key
        ].copy()

        # Keep the selected project's district context available.
        selected_project = projects[
            projects["Project_ID"].astype(str).str.upper() == project_key
        ]
        if district_id and not selected_project.empty:
            actual_district = str(
                selected_project.iloc[0]["District_ID"]
            ).upper()
            if actual_district != district_id.strip().upper():
                raise HTTPException(
                    status_code=422,
                    detail="The selected project does not belong to the selected district.",
                )

    # 3. Run the executive analysis.
    # The executive module derives sales positions from the full district
    # context, not from a pre-filtered subset of districts.
    try:
        result = executive_brief(
            survey=survey,
            districts=districts,
            projects=projects,
            indicators=indicators,
            status=status,
            funnel=funnel,
            district_id=district_id,
            sales_position=sales_position,
        )
    except Exception as exc:
        # Log the original exception in the backend; do not expose internals.
        import logging
        logging.getLogger(__name__).exception(
            "Executive resident-service analysis failed"
        )
        raise HTTPException(
            status_code=500,
            detail="Executive analysis could not be completed. Check the backend logs.",
        ) from exc

    if not isinstance(result, dict):
        raise HTTPException(
            status_code=500,
            detail="Executive analysis returned an invalid response.",
        )

    # 4. Add source and filter metadata without replacing analytical results.
    result["filters"] = {
        "district_id": district_id.upper() if district_id else None,
        "project_id": project_id.upper() if project_id else None,
        "sales_position": sales_position,
    }
    result["source"] = (
        "PostgreSQL: acud_survey_responses, Dim_District, Dim_Project, "
        "Fact_District_Service_Indicators, Fact_District_Monthly_Status, "
        "Fact_Project_Sales_Funnel"
    )
    result["data_status"] = (
        "synthetic_linked_demo"
        if result.get("data_mode") == "synthetic_demonstration"
        else "observed"
    )

    return _executive_json_safe(result)
