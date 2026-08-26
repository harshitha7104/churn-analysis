# Seller Churn Analysis — Flipkart/Amazon-style Marketplace
## Live Dashboard

[Churn Analysis — Live Dashboard](https://churn-analysis-oml9a6sqd6fupgzwcrpwy8.streamlit.app/)

## GitHub Repository

[Churn Analysis — Source Code](https://github.com/harshitha7104/churn-analysis)


**Business problem:** ~40% of marketplace sellers churn (stop selling) within their first year.
This project identifies churn drivers, segments sellers into behavioral archetypes, builds a
churn-prediction model, and recommends targeted interventions with an estimated ROI.

> Data is **simulated** (see `data/simulate_data.py`) with a realistic generative process —
> return rate, order velocity, ratings, SLA breaches, and category/city-tier effects all feed
> into churn probability — so the analysis has genuine signal to recover, without using real
> seller data.

## Live Results (this run)

- 6,000 simulated sellers, **41.1% churn rate**
- Best early-warning model: Logistic Regression, **AUC 0.80** (leakage-free feature set)
- **27.2%** of sellers flagged High Risk today
- 4 behavioral archetypes, from "Star Performers" (15% churn) to "Struggling / High-Return
  Risk" (78% churn)
- Estimated **~390% ROI** on a targeted intervention program (see PDF for assumptions)

## Repo Structure

```
seller-churn-analysis/
├── data/
│   ├── simulate_data.py        # generates sellers.csv (raw simulated data)
│   ├── sellers.csv              # raw simulated seller metrics
│   └── sellers_scored.csv       # output of the notebook: + segments + churn_risk_score
├── notebooks/
│   ├── build_notebook.py        # generates analysis.ipynb programmatically
│   └── analysis.ipynb           # EDA, driver analysis, segmentation, churn model, LTV calc
├── dashboard/
│   └── app.py                   # Streamlit app: segmentation + at-risk alerts
├── reports/
│   ├── build_pdf.py              # generates the PDF recommendation report
│   └── seller_churn_recommendations.pdf   # 2-page business summary
├── requirements.txt
└── README.md
```

## Setup

```bash
python3 -m venv venv && source venv/bin/activate   # optional but recommended
pip install -r requirements.txt
```

## Run the analysis

```bash
# 1. (Re)generate the simulated dataset
python3 data/simulate_data.py

# 2. Run the notebook (regenerates it from build_notebook.py, then executes it)
python3 notebooks/build_notebook.py
jupyter nbconvert --to notebook --execute notebooks/analysis.ipynb --output analysis.ipynb
# or just open notebooks/analysis.ipynb in Jupyter and "Run All"
# This writes data/sellers_scored.csv, which the dashboard reads.

# 3. Launch the dashboard
streamlit run dashboard/app.py

# 4. Rebuild the PDF (if you change the underlying numbers)
python3 reports/build_pdf.py
```

## Dashboard

Four tabs:
- **Segmentation** — archetype sizes, churn rate by archetype, radar profile of behavioral fingerprints
- **At-Risk Alerts** — sortable High-Risk seller list, revenue-at-risk KPI, rule-based suggested
  intervention per seller, CSV export
- **Churn Drivers** — correlation ranking, active-vs-churned box plots, category × city-tier
  churn heatmap
- **Seller Explorer** — look up any individual seller's full profile and risk score

## Methodology notes

- **Leakage handling:** `days_since_last_order` is near-definitional of churn at high values, so
  the primary ("early-warning") churn model excludes it. A secondary "trigger" model that
  includes it is also trained for comparison — it scores higher (AUC ~0.95) but mainly confirms
  churn already in progress rather than predicting it early. See Section 5 of the notebook.
- **Segmentation** is unsupervised (KMeans on behavioral/operational features), fit *without* the
  churn label, then profiled against churn rate afterward — so archetypes reflect genuine
  behavioral clusters, not a reverse-engineered churn split.
- **LTV/ROI assumptions** (commission rate, seller lifetime, coaching cost, save rate) are stated
  explicitly in the notebook and PDF and should be recalibrated against real platform economics
  before use in a real decision.

## Deploying

### GitHub
```bash
git init
git add .
git commit -m "Seller churn analysis: EDA, segmentation, churn model, dashboard, PDF report"
git branch -M main
git remote add origin https://github.com/<your-username>/seller-churn-analysis.git
git push -u origin main
```

### Streamlit Community Cloud (free live deployment)
1. Push this repo to GitHub (above).
2. Go to [share.streamlit.io](https://share.streamlit.io), sign in with GitHub.
3. "New app" → select this repo → set **Main file path** to `dashboard/app.py`.
4. Deploy. Since `data/sellers_scored.csv` is committed to the repo, no extra setup is needed —
   the app reads it directly. (If you want fresh data on every deploy, add a startup step that
   runs `data/simulate_data.py` and the notebook, or precompute and commit the CSV as done here.)

## Tech stack

pandas, NumPy, scikit-learn (Logistic Regression, Random Forest, Gradient Boosting, KMeans),
matplotlib/seaborn (static EDA), Plotly (interactive dashboard charts), Streamlit (dashboard),
ReportLab (PDF report).
