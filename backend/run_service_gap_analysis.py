"""Run the service-gap analysis from the CSV files (no database needed) and write a JSON contract.

    python run_service_gap_analysis.py [--area D03] [--out ../outputs/service_gap_analysis.json]
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import pandas as pd

BACKEND = Path(__file__).resolve().parent
ROOT = BACKEND.parent
sys.path.insert(0, str(BACKEND))
from app.analytics.executive import executive_brief  # noqa: E402
from app.analytics.service_gap import analyze  # noqa: E402


def load(data_dir: Path) -> dict:
    rd = lambda n: pd.read_csv(data_dir / n, encoding="utf-8-sig", low_memory=False)
    return dict(survey=rd("synthetic_survey_responses_ar.csv"), districts=rd("Dim_District.csv"), projects=rd("Dim_Project.csv"),
                indicators=rd("Fact_District_Service_Indicators.csv"), status=rd("Fact_District_Monthly_Status.csv"), funnel=rd("Fact_Project_Sales_Funnel.csv"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--area", default=None)
    ap.add_argument("--data-dir", default=str(ROOT / "data"))
    ap.add_argument("--out", default=str(ROOT / "outputs" / "service_gap_analysis.json"))
    ap.add_argument("--executive-out", default=str(ROOT / "outputs" / "executive_brief.json"))
    a = ap.parse_args()
    result = analyze(**load(Path(a.data_dir)), area_id=a.area)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    if a.area is None:
        Path(a.executive_out).write_text(json.dumps(executive_brief(**load(Path(a.data_dir))), ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        print(f"wrote {a.executive_out}")
    print(f"wrote {a.out}: status={result['analysis_status']} findings={len(result['service_findings'])} recommendations={len(result['recommended_actions'])}")


if __name__ == "__main__":
    main()
