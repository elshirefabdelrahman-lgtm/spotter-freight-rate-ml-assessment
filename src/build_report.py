"""Build the final submission-review PDF using the official scorer chart."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[1]
CHART_PATH = ROOT / "scorer_results" / "candidate_december.png"
REPORT_PATH = ROOT / "output" / "pdf" / "freight_rate_assessment_report.pdf"

NAVY = "#17324D"
BLUE = "#2574A9"
LIGHT = "#EAF1F7"
TEXT = "#263645"


def _table(data, widths=None):
    table = Table(data, colWidths=widths, repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(NAVY)),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#F7F9FB")),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#C8D4DE")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 7),
                ("RIGHTPADDING", (0, 0), (-1, -1), 7),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    return table


def _page(canvas, doc):
    canvas.saveState()
    canvas.setStrokeColor(colors.HexColor("#D3DDE5"))
    canvas.line(0.65 * inch, 0.55 * inch, 7.85 * inch, 0.55 * inch)
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#607585"))
    canvas.drawString(0.65 * inch, 0.35 * inch, "Freight Rate ML Assessment")
    canvas.drawRightString(7.85 * inch, 0.35 * inch, f"Page {doc.page}")
    canvas.restoreState()


def build_report() -> None:
    manifest = json.loads((ROOT / "scorer_results" / "phase3_prediction_manifest.json").read_text())
    december = pd.read_csv(ROOT / "december_predictions.csv")
    if not CHART_PATH.is_file():
        raise FileNotFoundError(
            "Official scorer chart is missing. Run score.py before building the report."
        )

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="TitleCenter", parent=styles["Title"], alignment=TA_CENTER, textColor=colors.HexColor(NAVY), fontSize=25, leading=31, spaceAfter=15))
    styles.add(ParagraphStyle(name="Subtitle", parent=styles["Normal"], alignment=TA_CENTER, textColor=colors.HexColor("#587083"), fontSize=11, leading=16, spaceAfter=18))
    styles["Heading1"].textColor = colors.HexColor(NAVY)
    styles["Heading1"].fontSize = 17
    styles["Heading1"].spaceBefore = 10
    styles["Heading1"].spaceAfter = 8
    styles["Heading2"].textColor = colors.HexColor(BLUE)
    styles["Heading2"].fontSize = 13
    styles["BodyText"].textColor = colors.HexColor(TEXT)
    styles["BodyText"].fontSize = 9.5
    styles["BodyText"].leading = 14
    body = styles["BodyText"]

    doc = SimpleDocTemplate(str(REPORT_PATH), pagesize=letter, rightMargin=0.65*inch, leftMargin=0.65*inch, topMargin=0.62*inch, bottomMargin=0.72*inch, title="Freight Rate ML Assessment Report", author="Candidate submission package")
    story = [
        Spacer(1, 0.55 * inch),
        Paragraph("Freight Rate Prediction", styles["TitleCenter"]),
        Paragraph("Machine Learning Assessment Report", styles["TitleCenter"]),
        Paragraph("Local submission package for human review - no external submission performed", styles["Subtitle"]),
        Spacer(1, 0.25 * inch),
        _table(
            [
                ["Selected model", "Development rows", "October MAE", "October RMSE"],
                ["Histogram gradient boosting", "48,000", "121.891275", "652.345044"],
            ],
            [2.3*inch, 1.4*inch, 1.4*inch, 1.4*inch],
        ),
        Spacer(1, 0.35 * inch),
        Paragraph("Executive Summary", styles["Heading1"]),
        Paragraph(
            "A deterministic histogram gradient-boosting pipeline was selected using forward temporal backtests. The final model was trained on all 48,000 labeled development loads from January 1 through October 31, 2025. It produced 12,000 validation predictions and 31 December scenario predictions. All output predictions are finite and positive. The October results in this report are internal development/holdout measurements, not Spotter's post-submission evaluation.", body),
        Paragraph("Problem Definition", styles["Heading1"]),
        Paragraph("The task is continuous regression of posted_rate from load, geography, equipment, market, and quote attributes. The external validation file has no target and was used only after final training for inference.", body),
        Paragraph("Data Understanding and Quality", styles["Heading1"]),
        Paragraph("The development file contains 48,000 labeled rows and 14 columns. The submission validation file contains 12,000 rows and the same 13 predictor-side columns. Development has 300 missing weights and 374 missing market_index values. It also contains 292 invalid negative weights. Negative weights are converted to missing values and handled by training-fitted imputation. IDs are unique and excluded from modeling.", body),
        Paragraph("Leakage Prevention", styles["Heading1"]),
        Paragraph("posted_rate and load_id are removed before preprocessing. All learned medians and category vocabularies live inside the pipeline. validation.csv and the December scenarios do not enter fitting, tuning, or feature selection. No target encoding, external data, date-derived feature, or future aggregate is used.", body),
        PageBreak(),
        Paragraph("Feature Engineering", styles["Heading1"]),
        Paragraph("The retained numeric inputs are pickup/delivery latitude and longitude, distance, cleaned weight, market_index, and quote_signal. The retained categorical inputs are pickup, delivery, and equipment. Calendar features, route identifiers, logarithms, geographic ratios, and interactions were tested and rejected after worsening temporal backtest performance.", body),
        Paragraph("Validation Strategy", styles["Heading1"]),
        Paragraph("Model/configuration decisions used expanding one-month validations for July, August, and September with all earlier months as training. October was the designated final internal period. A methodological limitation must be disclosed: the initial broad benchmark artifact computed October diagnostics for model families before later tuning and feature selection. The later decisions used July-September only, and October targets never entered fitting or preprocessing, but October was not a pristine once-only holdout.", body),
        Paragraph("Model Comparison", styles["Heading1"]),
        _table(
            [
                ["Model", "Train MAE", "October MAE", "Train RMSE", "October RMSE"],
                ["Median baseline", "1127.486", "1146.794", "1521.013", "1567.970"],
                ["Ridge + route", "132.274", "167.646", "585.540", "653.741"],
                ["Random forest", "90.040", "148.883", "521.885", "660.831"],
                ["HGB reference", "113.911", "154.854", "557.267", "655.206"],
            ],
            [1.75*inch, 1.15*inch, 1.25*inch, 1.15*inch, 1.25*inch],
        ),
        Paragraph("Overfitting Analysis", styles["Heading1"]),
        Paragraph("Random forest showed the largest initial MAE and RMSE gaps. The selected compact model reduced complexity to at most 15 leaves per tree and a minimum of 100 observations per leaf. Its final October MAE gap was 8.512133; its RMSE gap was 78.134576. The remaining RMSE gap reflects sensitivity to relatively rare large errors.", body),
        Paragraph("Final Model and Configuration", styles["Heading1"]),
        Paragraph(f"HistGradientBoostingRegressor uses learning_rate=0.05, max_iter=220, max_leaf_nodes=15, min_samples_leaf=100, l2_regularization=10, and random_state=42. Seeded early stopping is enabled. The full-development fit stopped after {manifest['actual_iterations']} iterations. One-hot encoding uses min_frequency=10 and unknown-category handling.", body),
        Paragraph("October Evaluation", styles["Heading1"]),
        _table(
            [["Partition", "MAE", "RMSE"], ["Training through September", "113.379142", "574.210468"], ["October holdout", "121.891275", "652.345044"], ["Gap", "8.512133", "78.134576"]],
            [3.2*inch, 1.6*inch, 1.6*inch],
        ),
        PageBreak(),
        Paragraph("Final Validation Predictions", styles["Heading1"]),
        _table(
            [["Count", "Minimum", "Maximum", "Mean", "Median"], [f"{manifest['validation_statistics']['count']:,}", f"{manifest['validation_statistics']['minimum']:,.2f}", f"{manifest['validation_statistics']['maximum']:,.2f}", f"{manifest['validation_statistics']['mean']:,.2f}", f"{manifest['validation_statistics']['median']:,.2f}"]],
            [1.25*inch]*5,
        ),
        Paragraph("The output preserves template order and exact load IDs, with no duplicate IDs, missing values, non-finite values, or nonpositive rates. These predictions have no known targets locally and therefore no claimed performance metric.", body),
        Paragraph("December Predictions", styles["Heading1"]),
        Paragraph(f"All 31 scenarios share identical modeled inputs except date, which the selected model excludes. Their predicted rate is therefore {manifest['december_statistics']['median']:,.6f}. Coordinates were recovered from unique development-only city mappings. market_index and quote_signal are absent from the scenario file and are handled as missing by the final pipeline's development-fitted median imputer.", body),
        Image(str(CHART_PATH), width=5.8*inch, height=3.26*inch),
        Paragraph("Figure 1. December scenario chart generated by the official supplied score.py after both prediction files passed its validation checks.", ParagraphStyle(name="Caption", parent=body, fontSize=8, textColor=colors.HexColor("#607585"), spaceBefore=4)),
        Paragraph("Limitations", styles["Heading1"]),
        Paragraph("market_index and quote_signal improve measured backtests, but their real quote-time provenance is not documented. Eight cities are unseen in development, although unknown-category handling and supplied coordinates prevent pipeline failure. December signal imputation removes day-to-day variation. The official scorer validates file structure and creates the chart but does not report Spotter's private model-quality metric. The October exposure described above limits the strength of a strictly untouched-holdout claim.", body),
        KeepTogether([
            Paragraph("Conclusion", styles["Heading1"]),
            Paragraph("The local package contains deterministic, validated prediction files generated by the fixed Phase 2 pipeline. The selected model materially improves internal MAE over the median baseline while retaining controlled complexity. Final quality remains subject to Spotter's private post-submission evaluation; no submission or publication has been performed.", body),
        ]),
    ]
    doc.build(story, onFirstPage=_page, onLaterPages=_page)


if __name__ == "__main__":
    build_report()
    print(CHART_PATH)
    print(REPORT_PATH)
