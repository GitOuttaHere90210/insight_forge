# export_to_pdf.py
# Usage: run this script from the project root with the project venv active.

from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak, Preformatted
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

from bi.data_loader import load_data
from bi.metrics import sales_by_month, sales_by_region

OUTPUT_PDF = "insightforge_report.pdf"
ASSETS_DIR = Path(".pdf_assets")
ASSETS_DIR.mkdir(exist_ok=True)

README_PATH = Path("insightforge_readme.md")
README_TEXT = README_PATH.read_text(encoding="utf-8") if README_PATH.exists() else "InsightForge"

# Load data (uses your bi.data_loader)
try:
    df = load_data()
except Exception as e:
    print("Could not load data:", e)
    df = pd.DataFrame()

def save_sales_by_month_chart(df, out_path):
    try:
        series = sales_by_month(df)
        if getattr(series, "empty", True):
            return None
        fig, ax = plt.subplots(figsize=(8, 3.5))
        series.plot(kind="bar", ax=ax)
        ax.set_title("Sales by Month")
        ax.set_ylabel("Sales")
        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        return out_path
    except Exception as e:
        print("chart error:", e)
        return None

def save_sales_by_region_chart(df, out_path):
    try:
        series = sales_by_region(df)
        if getattr(series, "empty", True):
            return None
        fig, ax = plt.subplots(figsize=(8, 3.5))
        series.plot(kind="bar", ax=ax, color="tab:orange")
        ax.set_title("Sales by Region")
        ax.set_ylabel("Sales")
        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        return out_path
    except Exception as e:
        print("chart error:", e)
        return None

# render charts
month_chart = save_sales_by_month_chart(df, ASSETS_DIR / "sales_by_month.png")
region_chart = save_sales_by_region_chart(df, ASSETS_DIR / "sales_by_region.png")

# Build PDF
styles = getSampleStyleSheet()
title_style = styles["Title"]
normal = styles["BodyText"]
# Code style for monospaced blocks
try:
    code_style = ParagraphStyle("Code", parent=styles["Code"], fontName="Courier", fontSize=8)
except Exception:
    code_style = ParagraphStyle("Code", fontName="Courier", fontSize=8, leading=9)

doc = SimpleDocTemplate(OUTPUT_PDF, pagesize=letter, rightMargin=72, leftMargin=72, topMargin=72, bottomMargin=72)
story = []

# Title page
story.append(Paragraph("InsightForge Report", title_style))
story.append(Spacer(1, 12))
story.append(Paragraph("Overview:", styles["Heading2"]))
for line in README_TEXT.splitlines():
    story.append(Paragraph(line, normal))
story.append(PageBreak())

# Charts
story.append(Paragraph("Key Charts", styles["Heading2"]))
story.append(Spacer(1, 12))
if month_chart:
    story.append(Image(str(month_chart), width=6.5*inch, height=3*inch))
    story.append(Spacer(1, 12))
if region_chart:
    story.append(Image(str(region_chart), width=6.5*inch, height=3*inch))
    story.append(Spacer(1, 12))

# Top metrics table
story.append(PageBreak())
story.append(Paragraph("Top Metrics", styles["Heading2"]))
if not df.empty:
    try:
        top_region = sales_by_region(df).sort_values(ascending=False)
        rows = [["Metric", "Value"]]
        if not top_region.empty:
            rows.append(["Top Region", top_region.index[0]])
            rows.append(["Top Region Sales", f"${top_region.iloc[0]:,.2f}"])
        month = sales_by_month(df)
        if not getattr(month, "empty", True):
            rows.append(["Peak Month", month.idxmax()])
            rows.append(["Peak Month Sales", f"${month.max():,.2f}"])
        table = Table(rows, hAlign="LEFT")
        table.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (-1,0), colors.grey),
            ("TEXTCOLOR",(0,0),(-1,0),colors.whitesmoke),
            ("ALIGN",(0,0),(-1,-1),"LEFT"),
            ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
            ("BOTTOMPADDING", (0,0), (-1,0), 12),
        ]))
        story.append(table)
    except Exception as e:
        story.append(Paragraph("Could not compute top metrics: " + str(e), normal))
else:
    story.append(Paragraph("No data available to compute metrics.", normal))

# Full code appendix: app.py and bi/metrics.py
story.append(PageBreak())
story.append(Paragraph("Code Appendix", styles["Heading2"]))
story.append(Spacer(1, 12))

def append_code_file(path, title=None):
    story.append(Paragraph(title or str(path), styles["Heading3"]))
    try:
        with open(path, "r", encoding="utf-8") as f:
            code = f.read()
        # Escape nothing here; Preformatted will render monospace and preserve whitespace
        story.append(Spacer(1, 6))
        story.append(Preformatted(code, code_style))
    except Exception as e:
        story.append(Paragraph(f"Could not read {path}: {e}", normal))
    story.append(PageBreak())

# Append the full app.py
append_code_file("app.py", title="app.py - Full Source")

# Append the metrics module if present
metrics_path = Path("bi") / "metrics.py"
if metrics_path.exists():
    append_code_file(str(metrics_path), title="bi/metrics.py - Full Source")
else:
    story.append(Paragraph("bi/metrics.py not found", normal))

doc.build(story)
print("PDF written to", OUTPUT_PDF)