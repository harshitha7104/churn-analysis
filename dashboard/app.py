import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

st.set_page_config(page_title="Seller Churn & Risk Dashboard", layout="wide",
                    initial_sidebar_state="expanded")

# ---------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------
DATA_PATH_CANDIDATES = [
    Path(__file__).parent.parent / "data" / "sellers_scored.csv",
    Path(__file__).parent / "sellers_scored.csv",
]

@st.cache_data
def load_data():
    for p in DATA_PATH_CANDIDATES:
        if p.exists():
            df = pd.read_csv(p)
            return df
    st.error("Could not find sellers_scored.csv. Run notebooks/analysis.ipynb first "
             "(it exports the scored dataset), or place the file next to app.py.")
    st.stop()

df = load_data()

# ---------------------------------------------------------------
# Sidebar filters
# ---------------------------------------------------------------
st.sidebar.title("🔎 Filters")
categories = st.sidebar.multiselect("Category", sorted(df["category"].unique()),
                                     default=sorted(df["category"].unique()))
tiers = st.sidebar.multiselect("City Tier", sorted(df["seller_tier_city"].unique()),
                                default=sorted(df["seller_tier_city"].unique()))
risk_filter = st.sidebar.multiselect("Risk Tier", ["Low", "Medium", "High"],
                                      default=["Low", "Medium", "High"])
min_rev, max_rev = float(df["avg_monthly_revenue"].min()), float(df["avg_monthly_revenue"].max())
rev_range = st.sidebar.slider("Avg Monthly Revenue (Rs.)", min_value=0.0,
                               max_value=round(max_rev, -2), value=(0.0, round(max_rev, -2)))

fdf = df[
    df["category"].isin(categories) &
    df["seller_tier_city"].isin(tiers) &
    df["risk_tier"].isin(risk_filter) &
    df["avg_monthly_revenue"].between(rev_range[0], rev_range[1])
]

st.sidebar.markdown("---")
st.sidebar.caption(f"Showing **{len(fdf):,}** of {len(df):,} sellers")

# ---------------------------------------------------------------
# Header + KPIs
# ---------------------------------------------------------------
st.title("📉 Seller Churn & At-Risk Dashboard")
st.caption("Simulated Flipkart/Amazon-style marketplace seller base — segmentation, churn drivers, and at-risk alerts.")

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Total Sellers", f"{len(fdf):,}")
k2.metric("Churn Rate", f"{fdf['churned'].mean()*100:.1f}%")
k3.metric("High Risk Sellers", f"{(fdf['risk_tier']=='High').sum():,}",
          f"{(fdf['risk_tier']=='High').mean()*100:.1f}% of filtered")
k4.metric("Avg Rating", f"{fdf['avg_rating'].mean():.2f} ★")
k5.metric("Avg Monthly Revenue", f"Rs.{fdf['avg_monthly_revenue'].mean():,.0f}")

st.markdown("---")

tab1, tab2, tab3, tab4 = st.tabs(["🧭 Segmentation", "⚠️ At-Risk Alerts",
                                    "📊 Churn Drivers", "🧾 Seller Explorer"])

# ---------------------------------------------------------------
# TAB 1 — Segmentation
# ---------------------------------------------------------------
with tab1:
    st.subheader("Seller Archetypes")
    col1, col2 = st.columns([1, 1])

    seg_summary = fdf.groupby("segment_name").agg(
        n_sellers=("seller_id", "count"),
        churn_rate=("churned", "mean"),
        avg_revenue=("avg_monthly_revenue", "mean"),
        avg_rating=("avg_rating", "mean"),
        avg_return_rate=("return_rate", "mean"),
    ).reset_index().sort_values("churn_rate", ascending=False)

    with col1:
        fig = px.bar(seg_summary, x="segment_name", y="n_sellers",
                      color="churn_rate", color_continuous_scale="RdYlGn_r",
                      title="Sellers per Archetype (colored by churn rate)",
                      labels={"segment_name": "Archetype", "n_sellers": "# Sellers", "churn_rate": "Churn Rate"})
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig2 = px.bar(seg_summary, x="segment_name", y="churn_rate",
                       title="Churn Rate by Archetype",
                       labels={"segment_name": "Archetype", "churn_rate": "Churn Rate"},
                       color="churn_rate", color_continuous_scale="RdYlGn_r")
        fig2.update_layout(yaxis_tickformat=".0%")
        st.plotly_chart(fig2, use_container_width=True)

    st.dataframe(
        seg_summary.style.format({
            "churn_rate": "{:.1%}", "avg_revenue": "Rs.{:,.0f}",
            "avg_rating": "{:.2f}", "avg_return_rate": "{:.1%}"
        }),
        use_container_width=True
    )

    st.subheader("Archetype Profile — Behavioral Fingerprint")
    radar_features = ["order_count", "avg_rating", "return_rate",
                       "order_velocity_30d_change", "pricing_competitiveness", "sla_breach_rate"]
    radar_df = fdf.groupby("segment_name")[radar_features].mean()
    radar_norm = (radar_df - df[radar_features].min()) / (df[radar_features].max() - df[radar_features].min())

    fig3 = go.Figure()
    for seg in radar_norm.index:
        fig3.add_trace(go.Scatterpolar(r=radar_norm.loc[seg].values, theta=radar_features,
                                        fill='toself', name=seg))
    fig3.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                        showlegend=True, height=500)
    st.plotly_chart(fig3, use_container_width=True)

# ---------------------------------------------------------------
# TAB 2 — At-Risk Alerts
# ---------------------------------------------------------------
with tab2:
    st.subheader("⚠️ High-Risk Seller Alerts")
    st.caption("Sellers flagged by the churn model as High Risk (churn probability > 0.6, tier boundaries from the notebook).")

    high_risk = fdf[fdf["risk_tier"] == "High"].sort_values("churn_risk_score", ascending=False)

    c1, c2, c3 = st.columns(3)
    c1.metric("High-Risk Sellers (filtered)", f"{len(high_risk):,}")
    c2.metric("Revenue at Risk (monthly)", f"Rs.{high_risk['avg_monthly_revenue'].sum():,.0f}")
    c3.metric("Avg Days Since Last Order", f"{high_risk['days_since_last_order'].mean():.0f} days")

    st.markdown("#### Top 25 Sellers to Intervene On (by risk score)")
    display_cols = ["seller_id", "category", "seller_tier_city", "segment_name",
                     "avg_monthly_revenue", "return_rate", "order_velocity_30d_change",
                     "rating_trend_90d", "sla_breach_rate", "days_since_last_order",
                     "churn_risk_score"]
    st.dataframe(
        high_risk[display_cols].head(25).style.format({
            "avg_monthly_revenue": "Rs.{:,.0f}", "return_rate": "{:.1%}",
            "order_velocity_30d_change": "{:+.1%}", "rating_trend_90d": "{:+.2f}",
            "sla_breach_rate": "{:.1%}", "churn_risk_score": "{:.2f}"
        }).background_gradient(subset=["churn_risk_score"], cmap="Reds"),
        use_container_width=True
    )

    st.markdown("#### Suggested Intervention by Primary Risk Driver")
    def suggest_intervention(row):
        reasons = []
        if row["return_rate"] > df["return_rate"].quantile(0.75):
            reasons.append("📦 Inventory/QC coaching (high returns)")
        if row["sla_breach_rate"] > df["sla_breach_rate"].quantile(0.75):
            reasons.append("🚚 Logistics/fulfillment support (SLA breaches)")
        if row["pricing_competitiveness"] < df["pricing_competitiveness"].quantile(0.25):
            reasons.append("💰 Pricing coaching (uncompetitive pricing)")
        if row["order_velocity_30d_change"] < -0.15:
            reasons.append("📈 Demand/marketing boost (order velocity dropping)")
        if row["rating_trend_90d"] < -0.15:
            reasons.append("⭐ Customer experience review (rating declining)")
        return "; ".join(reasons) if reasons else "🔍 General account health check"

    high_risk_display = high_risk.head(25).copy()
    high_risk_display["suggested_intervention"] = high_risk_display.apply(suggest_intervention, axis=1)
    st.dataframe(high_risk_display[["seller_id", "segment_name", "suggested_intervention"]],
                 use_container_width=True)

    csv = high_risk[display_cols].to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download Full At-Risk List (CSV)", csv, "at_risk_sellers.csv", "text/csv")

# ---------------------------------------------------------------
# TAB 3 — Churn Drivers
# ---------------------------------------------------------------
with tab3:
    st.subheader("What Drives Churn?")
    driver_cols = ["return_rate", "order_velocity_30d_change", "rating_trend_90d",
                   "sla_breach_rate", "support_tickets_total", "pricing_competitiveness",
                   "days_since_last_order", "tenure_days"]
    corr = fdf[driver_cols + ["churned"]].corr()["churned"].drop("churned").sort_values(key=abs, ascending=False)

    fig4 = px.bar(corr, orientation="h", title="Correlation with Churn",
                   labels={"value": "Correlation", "index": "Metric"},
                   color=corr.values, color_continuous_scale="RdBu_r")
    fig4.update_layout(showlegend=False)
    st.plotly_chart(fig4, use_container_width=True)

    colA, colB = st.columns(2)
    with colA:
        metric = st.selectbox("Compare metric: Active vs Churned", driver_cols, index=0)
        fig5 = px.box(fdf, x="churned", y=metric, color="churned",
                       labels={"churned": "Churned (0=Active, 1=Churned)"},
                       title=f"{metric} — Active vs Churned")
        st.plotly_chart(fig5, use_container_width=True)

    with colB:
        fig6 = px.histogram(fdf, x="churn_risk_score", color="churned", nbins=40, barmode="overlay",
                             title="Churn Risk Score Distribution", opacity=0.7)
        st.plotly_chart(fig6, use_container_width=True)

    st.subheader("Churn Rate by Category & City Tier")
    heat = fdf.pivot_table(index="category", columns="seller_tier_city", values="churned", aggfunc="mean")
    fig7 = px.imshow(heat, text_auto=".0%", color_continuous_scale="RdYlGn_r",
                      title="Churn Rate Heatmap")
    st.plotly_chart(fig7, use_container_width=True)

# ---------------------------------------------------------------
# TAB 4 — Seller Explorer
# ---------------------------------------------------------------
with tab4:
    st.subheader("🧾 Individual Seller Explorer")
    seller_id = st.selectbox("Select a seller", fdf["seller_id"].sort_values().tolist())
    row = fdf[fdf["seller_id"] == seller_id].iloc[0]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Risk Tier", row["risk_tier"])
    c2.metric("Churn Probability", f"{row['churn_risk_score']:.1%}")
    c3.metric("Segment", row["segment_name"])
    c4.metric("Status", "Churned" if row["churned"] == 1 else "Active")

    st.markdown("#### Profile")
    profile_cols = ["category", "seller_tier_city", "tenure_days", "order_count",
                     "avg_monthly_revenue", "avg_rating", "return_rate",
                     "order_velocity_30d_change", "rating_trend_90d",
                     "pricing_competitiveness", "sla_breach_rate",
                     "support_tickets_total", "days_since_last_order"]
    st.table(row[profile_cols])

    st.markdown("#### Full Seller Table (filtered)")
    st.dataframe(fdf, use_container_width=True, height=400)

st.markdown("---")
st.caption("Built with Streamlit · Data is simulated for portfolio/demo purposes · "
           "Model + segmentation logic in notebooks/analysis.ipynb")
