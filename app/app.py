import os
import sys
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

st.set_page_config(page_title="Parcl | Buyer Segmentation", layout="wide", page_icon="🏢")

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLEANED_PATH = os.path.join(BASE, "outputs", "cleaned_data", "buyer_profile_cleaned.csv")
PROFILE_PATH = os.path.join(BASE, "outputs", "tables", "cluster_profile.csv")


@st.cache_data
def load_data():
    if os.path.exists(CLEANED_PATH) and os.path.exists(PROFILE_PATH):
        df = pd.read_csv(CLEANED_PATH, parse_dates=["date_of_birth", "first_purchase", "last_purchase"])
        profile = pd.read_csv(PROFILE_PATH)
        return df, profile

    # Fallback: rebuild from scratch (mirrors src/run_pipeline.py)
    sys.path.append(os.path.join(BASE, "src"))
    import subprocess
    subprocess.run([sys.executable, os.path.join(BASE, "src", "run_pipeline.py")], check=True)
    df = pd.read_csv(CLEANED_PATH, parse_dates=["date_of_birth", "first_purchase", "last_purchase"])
    profile = pd.read_csv(PROFILE_PATH)
    return df, profile


df, profile = load_data()

SEGMENT_COLORS = {
    "Global Investors": "#8e44ad", "Luxury Investors": "#c0392b",
    "Corporate Buyers": "#2980b9", "First-Time Buyers": "#27ae60",
}

# ---------------------------------------------------------------------------
# SIDEBAR FILTERS
# ---------------------------------------------------------------------------
st.sidebar.title("🏢 Parcl Co. Limited")
st.sidebar.caption("Buyer Segmentation & Investment Profiling")
st.sidebar.divider()

countries = sorted(df["country"].dropna().unique().tolist())
sel_countries = st.sidebar.multiselect("Country", countries, default=countries)

regions = sorted(df.loc[df["country"].isin(sel_countries), "region"].dropna().unique().tolist())
sel_regions = st.sidebar.multiselect("Region", regions, default=[])

purposes = sorted(df["acquisition_purpose"].dropna().unique().tolist())
sel_purposes = st.sidebar.multiselect("Acquisition Purpose", purposes, default=purposes)

client_types = sorted(df["client_type"].dropna().unique().tolist())
sel_types = st.sidebar.multiselect("Client Type", client_types, default=client_types)

segments = sorted(df["segment_label"].dropna().unique().tolist())
sel_segments = st.sidebar.multiselect("Segment", segments, default=segments)

st.sidebar.divider()
st.sidebar.info(
    "ℹ️ **Note on segment names:** labels (Global Investors, Luxury Investors, "
    "Corporate Buyers, First-Time Buyers) were assigned to the statistically "
    "closest-matching cluster. Silhouette scores (~0.08–0.12) indicate "
    "moderate, not sharp, separation — see the Methodology Notes tab."
)

mask = (
    df["country"].isin(sel_countries)
    & df["acquisition_purpose"].isin(sel_purposes)
    & df["client_type"].isin(sel_types)
    & df["segment_label"].isin(sel_segments)
)
if sel_regions:
    mask &= df["region"].isin(sel_regions)

fdf = df[mask].copy()

# ---------------------------------------------------------------------------
# HEADER + KPIs
# ---------------------------------------------------------------------------
st.title("Buyer Segmentation & Investment Profiling")
st.caption("Parcl Co. Limited — ML-based clustering of 2,000 buyers across 10,000 property transactions")

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Clients (filtered)", f"{len(fdf):,}")
k2.metric("Total Investment", f"${fdf['total_investment'].sum():,.0f}")
k3.metric("Avg Properties / Client", f"{fdf['total_properties'].mean():.1f}" if len(fdf) else "—")
k4.metric("Avg Satisfaction", f"{fdf['satisfaction_score'].mean():.1f}/5" if len(fdf) else "—")
k5.metric("% Loan Applied", f"{(fdf['loan_applied']=='Yes').mean()*100:.0f}%" if len(fdf) else "—")

st.divider()

tab_overview, tab_investor, tab_geo, tab_insights, tab_notes = st.tabs(
    ["📊 Segmentation Overview", "💰 Investor Behavior", "🗺️ Geographic Analysis", "🔎 Segment Insights", "📄 Methodology Notes"]
)

# ---------------------------------------------------------------------------
# SEGMENTATION OVERVIEW
# ---------------------------------------------------------------------------
with tab_overview:
    st.subheader("Buyer Segment Distribution")
    c1, c2 = st.columns([1, 1])
    with c1:
        counts = fdf["segment_label"].value_counts().reset_index()
        counts.columns = ["Segment", "Clients"]
        fig = px.pie(counts, names="Segment", values="Clients", color="Segment",
                     color_discrete_map=SEGMENT_COLORS, title="Share of Clients by Segment")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = px.bar(counts.sort_values("Clients", ascending=False), x="Segment", y="Clients",
                      color="Segment", color_discrete_map=SEGMENT_COLORS, title="Client Count by Segment")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Cluster Validation (K-Means vs. Hierarchical)")
    st.caption(
        "Both algorithms were run independently on the same K=4 solution; a cross-tab shows "
        "how closely they agree. High overlap validates that K-Means found genuine structure."
    )
    cross = pd.crosstab(df["kmeans_cluster"], df["hierarchical_cluster"])
    st.dataframe(cross, use_container_width=True)

# ---------------------------------------------------------------------------
# INVESTOR BEHAVIOR
# ---------------------------------------------------------------------------
with tab_investor:
    st.subheader("Investment Patterns by Segment")
    agg = fdf.groupby("segment_label").agg(
        avg_total_investment=("total_investment", "mean"),
        avg_sale_price=("avg_sale_price", "mean"),
        avg_properties=("total_properties", "mean"),
        avg_floor_area=("avg_floor_area_sqft", "mean"),
        pct_loan=("loan_applied", lambda s: (s == "Yes").mean() * 100),
        pct_office=("pct_office", "mean"),
    ).reset_index() if len(fdf) else pd.DataFrame()

    if len(agg):
        c1, c2 = st.columns(2)
        with c1:
            fig = px.bar(agg.sort_values("avg_total_investment", ascending=False), x="segment_label", y="avg_total_investment",
                          color="segment_label", color_discrete_map=SEGMENT_COLORS, title="Avg Total Investment by Segment")
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            fig = px.bar(agg.sort_values("avg_sale_price", ascending=False), x="segment_label", y="avg_sale_price",
                          color="segment_label", color_discrete_map=SEGMENT_COLORS, title="Avg Sale Price by Segment")
            st.plotly_chart(fig, use_container_width=True)

        c3, c4 = st.columns(2)
        with c3:
            fig = px.bar(agg.sort_values("pct_loan", ascending=False), x="segment_label", y="pct_loan",
                          color="segment_label", color_discrete_map=SEGMENT_COLORS, title="% Loan Applied by Segment")
            st.plotly_chart(fig, use_container_width=True)
        with c4:
            fig = px.bar(agg.sort_values("pct_office", ascending=False), x="segment_label", y="pct_office",
                          color="segment_label", color_discrete_map=SEGMENT_COLORS, title="% Office Purchases by Segment")
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("Age vs. Total Investment (bubble = portfolio size)")
        fig = px.scatter(fdf, x="age", y="total_investment", size="total_properties", color="segment_label",
                          color_discrete_map=SEGMENT_COLORS, hover_data=["client_id", "country"],
                          title="Client-Level View: Age vs. Total Investment")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No clients match the current filters.")

# ---------------------------------------------------------------------------
# GEOGRAPHIC ANALYSIS
# ---------------------------------------------------------------------------
with tab_geo:
    st.subheader("Buyer Segments by Country")
    if len(fdf):
        country_seg = fdf.groupby(["country", "segment_label"]).size().reset_index(name="clients")
        fig = px.bar(country_seg, x="country", y="clients", color="segment_label",
                      color_discrete_map=SEGMENT_COLORS, title="Client Count by Country and Segment", barmode="stack")
        fig.update_layout(xaxis={'categoryorder': 'total descending'})
        st.plotly_chart(fig, use_container_width=True)

        ISO3 = {
            "USA": "USA", "Canada": "CAN", "Germany": "DEU", "Belgium": "BEL", "Mexico": "MEX",
            "Russia": "RUS", "UK": "GBR", "Denmark": "DNK", "France": "FRA", "Australia": "AUS",
        }
        geo_counts = fdf.groupby("country").size().reset_index(name="clients")
        geo_counts["iso3"] = geo_counts["country"].map(ISO3)
        fig = px.choropleth(geo_counts.dropna(subset=["iso3"]), locations="iso3", color="clients",
                             hover_name="country", color_continuous_scale="Blues",
                             title="Global Client Distribution (all segments)")
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Top Regions by Client Count")
        top_regions = fdf["region"].value_counts().head(15).reset_index()
        top_regions.columns = ["Region", "Clients"]
        fig = px.bar(top_regions.sort_values("Clients"), x="Clients", y="Region", orientation="h",
                      title="Top 15 Regions")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No clients match the current filters.")

# ---------------------------------------------------------------------------
# SEGMENT INSIGHTS PANEL
# ---------------------------------------------------------------------------
with tab_insights:
    st.subheader("Descriptive Statistics per Segment")
    sel_seg_detail = st.selectbox("Choose a segment to inspect", profile["segment_label"].tolist())
    row = profile[profile["segment_label"] == sel_seg_detail].iloc[0]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Clients", f"{row['n_clients']:,.0f}")
    c2.metric("Avg Total Investment", f"${row['avg_total_investment']:,.0f}")
    c3.metric("Avg Sale Price", f"${row['avg_sale_price']:,.0f}")
    c4.metric("Avg Satisfaction", f"{row['avg_satisfaction']:.1f}/5")

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Avg Age", f"{row['avg_age']:.1f}")
    c6.metric("% Loan Applied", f"{row['pct_loan_applied']:.1f}%")
    c7.metric("% Company-Type", f"{row['pct_company']:.1f}%")
    c8.metric("Avg Properties Owned", f"{row['avg_total_properties']:.1f}")

    st.caption(f"Top country: {row['top_country']} · Top referral channel: {row['top_referral']} · "
               f"% non-USA clients: {row['pct_non_usa']:.1f}% across {int(row['n_countries'])} countries")

    st.divider()
    st.subheader("Full Cluster Profile Table")
    st.dataframe(profile, use_container_width=True)

    st.subheader("Individual Clients in this Segment")
    seg_clients = fdf[fdf["segment_label"] == sel_seg_detail][
        ["client_id", "client_type", "country", "region", "age", "total_properties",
         "total_investment", "avg_sale_price", "satisfaction_score", "loan_applied"]
    ]
    st.dataframe(seg_clients, use_container_width=True, height=350)

# ---------------------------------------------------------------------------
# METHODOLOGY NOTES
# ---------------------------------------------------------------------------
with tab_notes:
    st.subheader("How these segments were built")
    st.markdown(
        """
**Pipeline:** `clients.csv` (2,000 records) was merged with aggregated purchase
history from `properties.csv` (10,000 listings, only `Sold` units) to build a
per-client investment profile — total properties owned, total and average
spend, floor area, portfolio breadth (distinct towers), and purchase span.

**Encoding:** One-hot encoding for `client_type`, `acquisition_purpose`,
`referral_channel`, `gender`; label encoding for the higher-cardinality
`region` (57 values) and `country` (10 values) fields.

**Scaling:** All numeric features were standardized (`StandardScaler`) before
clustering, since K-Means is distance-based and sensitive to feature scale.

**Model selection:** K-Means and Hierarchical (Ward linkage) clustering were
both run for K = 2–10. The Elbow Method and Silhouette Score were used to
evaluate quality. **The silhouette-optimal K is 3** (score ≈ 0.124); **K=4
was used anyway** to match Parcl's four-persona business framework, scoring
only slightly lower (≈ 0.107). K-Means and Hierarchical clustering agreed on
cluster membership for about 80% of clients at K=4, which supports K-Means'
result without masking the fact that these are moderate, not sharp, clusters.

**Segment naming (relative, not absolute):**
1. *Global Investors* = largest average property portfolio.
2. *Luxury Investors* = highest average sale price among the rest.
3. *Corporate Buyers* = highest Company-type share among the rest.
4. *First-Time Buyers* = the remaining cluster.

**Known limitation:** the *First-Time Buyers* segment does not actually skew
young or loan-dependent in this data (avg. age 53.8, lowest loan-applied rate
of the four segments at 35.4%) — it is better understood as Parcl's baseline/
mainstream buyer group than a literal first-time-buyer persona. This is
reported transparently rather than relabeled to fit the assumption.

**Also note:** `transaction_date` in the source data is a monthly snapshot
(always the 1st of the month), not a daily timestamp — any time-based feature
here reflects month-level granularity only.
"""
    )
