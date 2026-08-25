"""
Simulates seller-level data for an e-commerce marketplace (Flipkart/Amazon-style).
Churn is NOT random - it's driven by a latent 'seller health' process so the
downstream analysis has real signal to recover (return rate, order velocity
decline, rating decline, support ticket load, category, tenure).
"""
import numpy as np
import pandas as pd

np.random.seed(42)
N = 6000

CATEGORIES = ["Fashion", "Electronics", "Home & Kitchen", "Beauty & Personal Care",
              "Mobiles & Accessories", "Grocery", "Toys & Baby", "Sports & Fitness"]
CATEGORY_RETURN_BASE = {  # categories vary structurally in return rate
    "Fashion": 0.18, "Electronics": 0.09, "Home & Kitchen": 0.07,
    "Beauty & Personal Care": 0.05, "Mobiles & Accessories": 0.11,
    "Grocery": 0.02, "Toys & Baby": 0.06, "Sports & Fitness": 0.06
}
TIERS = ["Tier-1 Metro", "Tier-2 City", "Tier-3 Town"]
TIER_LOGISTICS_RISK = {"Tier-1 Metro": 0.03, "Tier-2 City": 0.07, "Tier-3 Town": 0.13}

rows = []
today_day = 730  # simulate a 2-year platform window, "today" = day 730

for sid in range(1, N + 1):
    tenure_days = int(np.random.gamma(shape=2.2, scale=170))
    tenure_days = min(tenure_days, today_day)
    join_day = today_day - tenure_days
    category = np.random.choice(CATEGORIES)
    tier = np.random.choice(TIERS, p=[0.45, 0.35, 0.20])

    # Latent seller "quality" (skill/capital/ops maturity) drives almost everything
    quality = np.clip(np.random.normal(0.55, 0.20), 0.02, 0.98)

    # Onboarding cohort effect - later cohorts benefit from better platform tooling
    cohort_bonus = 0.05 if join_day > 365 else 0.0

    base_monthly_orders = np.clip(np.random.gamma(2.0, 25) * (0.4 + quality), 1, None)
    order_count = int(base_monthly_orders * (tenure_days / 30.0))
    order_count = max(order_count, 1)

    # Return rate: category base + inverse quality + logistics risk + noise
    return_rate = (CATEGORY_RETURN_BASE[category]
                   + (1 - quality) * 0.12
                   + TIER_LOGISTICS_RISK[tier]
                   + np.random.normal(0, 0.02))
    return_rate = float(np.clip(return_rate, 0.005, 0.65))

    # Rating: quality-driven, dragged down by returns/complaints
    avg_rating = np.clip(np.random.normal(3.3 + quality * 1.5 - return_rate * 2, 0.35), 1.0, 5.0)

    # Support tickets: inverse quality, plus return-driven complaints
    support_tickets = np.random.poisson(lam=max(0.5, (1 - quality) * 8 + return_rate * 15))

    # Order velocity trend: % change in orders, last 30d vs prior 30d
    # Struggling sellers show negative velocity; this is a leading churn indicator
    velocity_shift = np.random.normal((quality - 0.5) * 0.6 - return_rate * 0.8 + cohort_bonus, 0.25)
    order_velocity_30d_change = float(np.clip(velocity_shift, -1.0, 1.5))

    # Rating trend over last 90 days (decline signal)
    rating_trend_90d = float(np.clip(np.random.normal((quality - 0.5) * 0.4 - return_rate * 0.5, 0.15), -1.5, 1.0))

    # Pricing competitiveness index (0=overpriced vs category, 1=competitive)
    pricing_competitiveness = float(np.clip(np.random.normal(0.4 + quality * 0.4, 0.15), 0, 1))

    # Fulfillment SLA breach rate
    sla_breach_rate = float(np.clip(np.random.normal((1 - quality) * 0.25 + TIER_LOGISTICS_RISK[tier], 0.05), 0, 0.9))

    # Latent churn propensity - the "true" generative process
    churn_score = (
        -2.55
        + 3.2 * (1 - quality)
        + 2.0 * return_rate
        + 1.6 * max(0, -order_velocity_30d_change)
        + 1.3 * max(0, -rating_trend_90d)
        + 0.05 * support_tickets
        + 1.0 * (1 - pricing_competitiveness)
        + 1.5 * sla_breach_rate
        - 0.9 * (avg_rating - 3)
        - 0.002 * min(tenure_days, 400)  # tenure = slight protective effect (survivorship)
        + np.random.normal(0, 0.6)
    )
    churn_prob = 1 / (1 + np.exp(-churn_score))
    churned = int(np.random.rand() < churn_prob)

    # Days since last order - churned sellers have gone quiet; active ones ordered recently
    if churned:
        days_since_last_order = int(np.clip(np.random.gamma(2.2, 20) + np.random.normal(0, 15), 5, tenure_days))
    else:
        days_since_last_order = int(np.clip(np.random.exponential(9) + np.random.normal(0, 8), 0, min(45, tenure_days)))
    days_since_last_order = max(days_since_last_order, 0)

    monthly_revenue = order_count / max(tenure_days / 30.0, 1) * np.random.uniform(350, 1400)  # INR proxy per order avg

    rows.append(dict(
        seller_id=f"SLR{sid:06d}",
        category=category,
        seller_tier_city=tier,
        join_day=join_day,
        tenure_days=tenure_days,
        order_count=order_count,
        avg_monthly_revenue=round(monthly_revenue, 2),
        avg_rating=round(avg_rating, 2),
        return_rate=round(return_rate, 4),
        support_tickets_total=int(support_tickets),
        order_velocity_30d_change=round(order_velocity_30d_change, 4),
        rating_trend_90d=round(rating_trend_90d, 4),
        pricing_competitiveness=round(pricing_competitiveness, 4),
        sla_breach_rate=round(sla_breach_rate, 4),
        days_since_last_order=days_since_last_order,
        churned=churned,
    ))

df = pd.DataFrame(rows)
df.to_csv("/home/claude/seller-churn-analysis/data/sellers.csv", index=False)
print(df.shape)
print(df["churned"].mean())
print(df.head())
