from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
                                 ListFlowable, ListItem)
from reportlab.lib.enums import TA_LEFT

styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name="H1c", parent=styles["Heading1"], fontSize=18,
                           textColor=colors.HexColor("#1a1a2e"), spaceAfter=4))
styles.add(ParagraphStyle(name="H2c", parent=styles["Heading2"], fontSize=12.5,
                           textColor=colors.HexColor("#2e4057"), spaceBefore=10, spaceAfter=4))
styles.add(ParagraphStyle(name="Body", parent=styles["Normal"], fontSize=9.3, leading=12.5))
styles.add(ParagraphStyle(name="Small", parent=styles["Normal"], fontSize=8, textColor=colors.grey))
styles.add(ParagraphStyle(name="Sub", parent=styles["Normal"], fontSize=10, textColor=colors.HexColor("#555555")))

doc = SimpleDocTemplate("/home/claude/seller-churn-analysis/reports/seller_churn_recommendations.pdf",
                         pagesize=letter, topMargin=0.55*inch, bottomMargin=0.55*inch,
                         leftMargin=0.6*inch, rightMargin=0.6*inch)

story = []

story.append(Paragraph("Seller Churn: Findings &amp; Intervention Plan", styles["H1c"]))
story.append(Paragraph("Marketplace Seller Health Program &middot; Analysis of 6,000 simulated sellers", styles["Sub"]))
story.append(Spacer(1, 8))

# KPI strip
kpi_style_num = ParagraphStyle(name="KpiNum", fontSize=16, fontName="Helvetica-Bold",
                                textColor=colors.HexColor("#c0392b"), alignment=1, leading=18)
kpi_style_cap = ParagraphStyle(name="KpiCap", fontSize=7.3, textColor=colors.HexColor("#555555"),
                                alignment=1, leading=9)
kpi_data = [
    [Paragraph("41.1%", kpi_style_num), Paragraph("27.2%", kpi_style_num),
     Paragraph("0.80 AUC", kpi_style_num), Paragraph("~390%", kpi_style_num)],
    [Paragraph("Sellers churn within 12 months", kpi_style_cap),
     Paragraph("of sellers flagged High Risk today", kpi_style_cap),
     Paragraph("early-warning model accuracy", kpi_style_cap),
     Paragraph("estimated ROI on targeted intervention", kpi_style_cap)],
]
kpi_table = Table(kpi_data, colWidths=[1.55*inch]*4)
kpi_table.setStyle(TableStyle([
    ("ALIGN", (0,0), (-1,-1), "CENTER"),
    ("VALIGN", (0,0), (-1,-1), "TOP"),
    ("BOTTOMPADDING", (0,0), (-1,0), 3),
    ("LINEBELOW", (0,0), (-1,0), 0.5, colors.HexColor("#dddddd")),
    ("TOPPADDING", (0,0), (-1,-1), 4),
]))
story.append(kpi_table)
story.append(Spacer(1, 8))

story.append(Paragraph("What's Driving Churn", styles["H2c"]))
story.append(Paragraph(
    "Churn is not random — it's concentrated in sellers with declining order momentum, high "
    "return rates, and repeated fulfillment (SLA) failures. The strongest signals, ranked by "
    "predictive strength in the model:", styles["Body"]))

bullets = [
    "<b>Order velocity decline</b> — a drop in 30-day order volume is the clearest early warning sign; sellers show slowing activity well before they formally stop selling.",
    "<b>Return rate</b> — sellers above the 75th percentile return rate churn at roughly double the base rate, concentrated in Fashion and Mobiles &amp; Accessories.",
    "<b>SLA / fulfillment breaches</b> — logistics failures compound with returns, and are most severe for Tier-3 town sellers (49.4% churn vs. 36.7% for Tier-1 metro sellers).",
    "<b>Rating decline &amp; support ticket volume</b> — a downward 90-day rating trend and rising support load add incremental signal on top of the operational metrics above.",
    "<b>Category effect</b> — Fashion has the highest churn rate (51.8%), Grocery the lowest (34.0%), reflecting structural differences in return burden and repeat-purchase behavior.",
]
story.append(ListFlowable([ListItem(Paragraph(b, styles["Body"]), leftIndent=8) for b in bullets],
                            bulletType="bullet", start="•", leftIndent=14))

story.append(Paragraph("Seller Archetypes", styles["H2c"]))
story.append(Paragraph(
    "Behavioral clustering (KMeans on operational metrics, independent of the churn label) surfaces "
    "four recurring seller archetypes:", styles["Body"]))

cell = ParagraphStyle(name="Cell", fontSize=7.6, leading=9.2)
cell_b = ParagraphStyle(name="CellB", fontSize=7.6, leading=9.2, fontName="Helvetica-Bold")
hdr = ParagraphStyle(name="Hdr", fontSize=7.8, leading=9.5, fontName="Helvetica-Bold", textColor=colors.white)

seg_rows = [
    ["Struggling / High-Return Risk", "1,335", "77.8%", "Rs.33,316", "Urgent — highest priority for intervention"],
    ["Declining Momentum", "1,397", "44.3%", "Rs.34,295", "Early warning stage — velocity/rating slipping"],
    ["Steady Mid-Tier", "1,372", "38.0%", "Rs.36,010", "Stable but not growing — coaching upside"],
    ["Star Performers", "1,896", "15.0%", "Rs.57,705", "Healthy — retain via recognition, not coaching"],
]
seg_data = [[Paragraph(h, hdr) for h in ["Archetype", "Sellers", "Churn Rate", "Avg. Mo. Revenue", "Read"]]]
for r in seg_rows:
    seg_data.append([Paragraph(r[0], cell_b), Paragraph(r[1], cell), Paragraph(r[2], cell),
                      Paragraph(r[3], cell), Paragraph(r[4], cell)])

seg_table = Table(seg_data, colWidths=[1.55*inch, 0.55*inch, 0.65*inch, 0.95*inch, 2.2*inch])
seg_table.setStyle(TableStyle([
    ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#2e4057")),
    ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#f4f4f8")]),
    ("GRID", (0,0), (-1,-1), 0.4, colors.HexColor("#cccccc")),
    ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
    ("TOPPADDING", (0,0), (-1,-1), 4),
    ("BOTTOMPADDING", (0,0), (-1,-1), 4),
    ("LEFTPADDING", (0,0), (-1,-1), 4),
    ("RIGHTPADDING", (0,0), (-1,-1), 4),
]))
story.append(seg_table)
story.append(Spacer(1, 8))

story.append(Paragraph("Recommended Interventions", styles["H2c"]))
interv_rows = [
    ["High return rate", "Inventory/QC coaching: listing accuracy, packaging audit, sizing guides (Fashion)", "Seller Success"],
    ["SLA breaches", "Logistics support: fulfillment partner enrollment, dispatch SLA training", "Logistics Ops"],
    ["Weak pricing competitiveness", "Automated pricing coaching / repricing tool recommendation", "Growth/Pricing"],
    ["Negative order velocity", "Demand boost: ad credits, category placement, promo eligibility", "Marketing"],
    ["Declining ratings / rising tickets", "Proactive CX outreach + root-cause review call", "Support"],
]
interv_data = [[Paragraph(h, hdr) for h in ["Driver Signal", "Intervention", "Owner"]]]
for r in interv_rows:
    interv_data.append([Paragraph(r[0], cell_b), Paragraph(r[1], cell), Paragraph(r[2], cell)])

interv_table = Table(interv_data, colWidths=[1.55*inch, 3.75*inch, 0.9*inch])
interv_table.setStyle(TableStyle([
    ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#2e4057")),
    ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#f4f4f8")]),
    ("GRID", (0,0), (-1,-1), 0.4, colors.HexColor("#cccccc")),
    ("VALIGN", (0,0), (-1,-1), "TOP"),
    ("TOPPADDING", (0,0), (-1,-1), 4),
    ("BOTTOMPADDING", (0,0), (-1,-1), 4),
    ("LEFTPADDING", (0,0), (-1,-1), 4),
    ("RIGHTPADDING", (0,0), (-1,-1), 4),
]))
story.append(interv_table)
story.append(Spacer(1, 8))

story.append(Paragraph("Prioritization &amp; Expected Impact", styles["H2c"]))
story.append(Paragraph(
    "1,631 sellers (27.2% of the base) are flagged <b>High Risk</b> by the churn model today. Targeting "
    "this group with the interventions above, assuming a 25% save rate, a 4-month expected lifetime "
    "without intervention vs. 14 months if retained, and Rs.2,000 average coaching cost per seller:",
    styles["Body"]))

impact_data = [
    ["Sellers targeted", "1,631"],
    ["Expected sellers saved (@25% success)", "~407"],
    ["Avg. incremental LTV per saved seller", "Rs.39,313"],
    ["Total incremental LTV", "Rs.1.60 crore (Rs.16.0M)"],
    ["Total program cost", "Rs.32.6 lakh (Rs.3.26M)"],
    ["Estimated ROI", "~390%"],
]
impact_table = Table(impact_data, colWidths=[3.2*inch, 2.9*inch])
impact_table.setStyle(TableStyle([
    ("FONTSIZE", (0,0), (-1,-1), 8.3),
    ("ROWBACKGROUNDS", (0,0), (-1,-1), [colors.white, colors.HexColor("#f4f4f8")]),
    ("GRID", (0,0), (-1,-1), 0.4, colors.HexColor("#cccccc")),
    ("FONTNAME", (0,0), (0,-1), "Helvetica-Bold"),
    ("TOPPADDING", (0,0), (-1,-1), 3),
    ("BOTTOMPADDING", (0,0), (-1,-1), 3),
]))
story.append(impact_table)
story.append(Spacer(1, 6))

story.append(Paragraph(
    "<b>Next steps:</b> (1) Pilot interventions on the top 200 highest-risk sellers for one quarter to "
    "validate the assumed 25% save rate before scaling. (2) Route the weekly at-risk feed from the "
    "model directly to Seller Success/Logistics/Pricing owners per the table above. (3) Re-run the "
    "model monthly and track precision/recall against actual outcomes to recalibrate thresholds.",
    styles["Body"]))

story.append(Spacer(1, 10))
story.append(Paragraph(
    "Source: analysis.ipynb (EDA, driver analysis, KMeans segmentation, Logistic Regression / "
    "Random Forest / Gradient Boosting churn models). Data is simulated for portfolio purposes; "
    "recalibrate commission rate, seller lifetime, and coaching cost assumptions against real "
    "platform economics before use.", styles["Small"]))

doc.build(story)
print("PDF built.")
