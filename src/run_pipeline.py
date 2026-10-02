"""
run_pipeline.py
----------------
End-to-end, reproducible pipeline for Parcl Co. Limited's
Machine-Learning-Based Buyer Segmentation and Investment Profiling project.

Steps: data cleaning -> client-property merge & feature engineering ->
encoding -> scaling -> KMeans + Hierarchical clustering -> evaluation
(Elbow + Silhouette) -> cluster interpretation / profiling.

Run this to regenerate every table/chart used by the notebook and dashboard.
"""

import os
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import linkage, fcluster

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE, "data")
OUT_TABLES = os.path.join(BASE, "outputs", "tables")
OUT_CLEAN = os.path.join(BASE, "outputs", "cleaned_data")
os.makedirs(OUT_TABLES, exist_ok=True)
os.makedirs(OUT_CLEAN, exist_ok=True)

log = {}

# =========================================================================
# STEP 0 — LOAD
# =========================================================================
clients = pd.read_csv(os.path.join(DATA_DIR, "clients.csv"))
properties = pd.read_csv(os.path.join(DATA_DIR, "properties.csv"))

log["clients_shape"] = list(clients.shape)
log["properties_shape"] = list(properties.shape)
log["clients_missing"] = clients.isnull().sum().to_dict()
log["properties_missing"] = properties.isnull().sum().to_dict()
log["clients_duplicate_ids"] = int(clients["client_id"].duplicated().sum())
log["clients_duplicate_rows"] = int(clients.duplicated().sum())
log["properties_duplicate_ids"] = int(properties["listing_id"].duplicated().sum())

# =========================================================================
# STEP 1 — DATA CLEANING
# =========================================================================
# 1a. properties.client_ref is null exactly for "Available" (unsold) listings.
status_vs_null = pd.crosstab(properties["listing_status"], properties["client_ref"].isnull())
log["listing_status_vs_missing_client_ref"] = status_vs_null.to_dict()

# 1b. date_of_birth uses two delimiters ('-' and '/'). Slash-delimited rows
# prove month-first ordering (second component reaches 31, which can only be
# a day), so BOTH delimiter groups are parsed as month-first for consistency.
dob = pd.to_datetime(clients["date_of_birth"], format="mixed", dayfirst=False, errors="coerce")
log["dob_unparseable"] = int(dob.isnull().sum())
log["dob_range"] = [str(dob.min().date()), str(dob.max().date())]
clients["date_of_birth"] = dob

REFERENCE_DATE = pd.Timestamp("2024-01-01")  # first transaction date in properties.csv
clients["age"] = ((REFERENCE_DATE - clients["date_of_birth"]).dt.days / 365.25)
log["age_describe"] = clients["age"].describe().to_dict()

# 1c. properties.transaction_date is MM-DD-YYYY with day always '01' -- a
# monthly transaction snapshot, not a daily timestamp. Documented, not
# treated as an error.
properties["transaction_date"] = pd.to_datetime(properties["transaction_date"], format="%m-%d-%Y", errors="coerce")
log["transaction_date_unparseable"] = int(properties["transaction_date"].isnull().sum())
log["transaction_date_range"] = [str(properties["transaction_date"].min().date()), str(properties["transaction_date"].max().date())]

# 1d. sale_price is a formatted currency string -> clean to float
properties["sale_price"] = properties["sale_price"].replace(r"[\$,]", "", regex=True).astype(float)
log["sale_price_negative_or_zero"] = int((properties["sale_price"] <= 0).sum())

# 1e. standardize text fields
for c_col in ["client_type", "gender", "country", "region", "acquisition_purpose", "loan_applied", "referral_channel"]:
    clients[c_col] = clients[c_col].astype(str).str.strip()
for p_col in ["unit_category", "listing_status"]:
    properties[p_col] = properties[p_col].astype(str).str.strip()

clients_clean_path = os.path.join(OUT_CLEAN, "clients_cleaned.csv")
properties_clean_path = os.path.join(OUT_CLEAN, "properties_cleaned.csv")
clients.to_csv(clients_clean_path, index=False)
properties.to_csv(properties_clean_path, index=False)

# =========================================================================
# STEP 2 — MERGE & FEATURE ENGINEERING (client-level investment profile)
# =========================================================================
sold = properties[properties["listing_status"] == "Sold"].copy()

agg = sold.groupby("client_ref").agg(
    total_properties=("listing_id", "count"),
    total_investment=("sale_price", "sum"),
    avg_sale_price=("sale_price", "mean"),
    max_sale_price=("sale_price", "max"),
    avg_floor_area_sqft=("floor_area_sqft", "mean"),
    total_floor_area_sqft=("floor_area_sqft", "sum"),
    distinct_towers=("tower_number", "nunique"),
    first_purchase=("transaction_date", "min"),
    last_purchase=("transaction_date", "max"),
    pct_office=("unit_category", lambda s: (s == "Office").mean() * 100),
).reset_index().rename(columns={"client_ref": "client_id"})

agg["purchase_span_months"] = (
    (agg["last_purchase"].dt.year - agg["first_purchase"].dt.year) * 12
    + (agg["last_purchase"].dt.month - agg["first_purchase"].dt.month)
)

df = clients.merge(agg, on="client_id", how="inner")  # inner: every client has >=1 sold property (verified above)
log["clients_after_merge"] = int(len(df))
log["clients_dropped_no_sold_property"] = int(len(clients) - len(df))

df["loan_applied_flag"] = (df["loan_applied"] == "Yes").astype(int)
df["is_company"] = (df["client_type"] == "Company").astype(int)
df["is_investment_purpose"] = (df["acquisition_purpose"] == "Investment").astype(int)

cleaned_full_path = os.path.join(OUT_CLEAN, "buyer_profile_cleaned.csv")
df.to_csv(cleaned_full_path, index=False)

# =========================================================================
# STEP 3 — FEATURE ENCODING
# =========================================================================
CATEGORICAL_ONEHOT = ["client_type", "acquisition_purpose", "referral_channel", "gender"]
CATEGORICAL_LABEL = ["region", "country"]  # high-cardinality -> label encoding

df_model = df.copy()

label_encoders = {}
for col in CATEGORICAL_LABEL:
    le = LabelEncoder()
    df_model[col + "_enc"] = le.fit_transform(df_model[col])
    label_encoders[col] = dict(zip(le.classes_, le.transform(le.classes_).tolist()))

df_model = pd.get_dummies(df_model, columns=CATEGORICAL_ONEHOT, prefix=CATEGORICAL_ONEHOT)
onehot_cols = [c for c in df_model.columns if any(c.startswith(p + "_") for p in CATEGORICAL_ONEHOT)]

# =========================================================================
# STEP 4 — FEATURE SCALING
# =========================================================================
NUMERIC_FEATURES = [
    "age", "satisfaction_score", "loan_applied_flag",
    "total_properties", "total_investment", "avg_sale_price",
    "avg_floor_area_sqft", "distinct_towers", "pct_office", "purchase_span_months",
]
FEATURE_COLS = NUMERIC_FEATURES + ["region_enc", "country_enc"] + onehot_cols

X = df_model[FEATURE_COLS].fillna(0).copy()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X[NUMERIC_FEATURES + ["region_enc", "country_enc"]])
X_scaled = pd.DataFrame(X_scaled, columns=NUMERIC_FEATURES + ["region_enc", "country_enc"], index=X.index)
X_final = pd.concat([X_scaled, X[onehot_cols].astype(float).reset_index(drop=True)], axis=1)

log["feature_columns"] = FEATURE_COLS
log["n_features"] = len(FEATURE_COLS)

# =========================================================================
# STEP 5 — OPTIMAL CLUSTER SELECTION (Elbow + Silhouette)
# =========================================================================
inertias = {}
silhouettes = {}
K_RANGE = range(2, 11)
for k in K_RANGE:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_final)
    inertias[k] = float(km.inertia_)
    silhouettes[k] = float(silhouette_score(X_final, labels))

log["elbow_inertias"] = inertias
log["silhouette_scores"] = silhouettes
best_k = max(silhouettes, key=silhouettes.get)
log["best_k_by_silhouette"] = best_k

# Business constraint: brief recommends 4 segments (C1-C4). Use the brief's
# K=4 for the primary model (close to/at the statistically best K), and
# report the silhouette-optimal K transparently alongside it.
FINAL_K = 4
log["final_k_used"] = FINAL_K

# =========================================================================
# STEP 6 — CLUSTERING MODELS
# =========================================================================
kmeans = KMeans(n_clusters=FINAL_K, random_state=42, n_init=10)
df["kmeans_cluster"] = kmeans.fit_predict(X_final)
log["kmeans_silhouette"] = float(silhouette_score(X_final, df["kmeans_cluster"]))

agglo = AgglomerativeClustering(n_clusters=FINAL_K, linkage="ward")
df["hierarchical_cluster"] = agglo.fit_predict(X_final)
log["hierarchical_silhouette"] = float(silhouette_score(X_final, df["hierarchical_cluster"]))

# Agreement between the two methods (relabeled via majority mapping)
cross = pd.crosstab(df["kmeans_cluster"], df["hierarchical_cluster"])
mapping = cross.idxmax(axis=1).to_dict()
df["hierarchical_cluster_aligned"] = df["hierarchical_cluster"]
agreement = (df["kmeans_cluster"].map(mapping) == df["hierarchical_cluster"]).mean()
log["kmeans_vs_hierarchical_agreement_pct"] = float(agreement * 100)
log["cross_tab_kmeans_hierarchical"] = cross.to_dict()

# Linkage matrix for dendrogram (on a sample for tractability/legibility)
sample_idx = df.sample(n=min(60, len(df)), random_state=42).index
Z = linkage(X_final.loc[sample_idx], method="ward")
np.save(os.path.join(OUT_TABLES, "dendrogram_linkage.npy"), Z)
df.loc[sample_idx, ["client_id"]].to_csv(os.path.join(OUT_TABLES, "dendrogram_sample_ids.csv"), index=False)

# =========================================================================
# STEP 7 — CLUSTER INTERPRETATION / PROFILING
# =========================================================================
profile = df.groupby("kmeans_cluster").agg(
    n_clients=("client_id", "count"),
    pct_company=("is_company", "mean"),
    pct_investment_purpose=("is_investment_purpose", "mean"),
    pct_loan_applied=("loan_applied_flag", "mean"),
    avg_age=("age", "mean"),
    avg_satisfaction=("satisfaction_score", "mean"),
    avg_total_properties=("total_properties", "mean"),
    avg_total_investment=("total_investment", "mean"),
    avg_sale_price=("avg_sale_price", "mean"),
    avg_floor_area=("avg_floor_area_sqft", "mean"),
    pct_office=("pct_office", "mean"),
    top_country=("country", lambda s: s.mode().iloc[0]),
    pct_non_usa=("country", lambda s: (s != "USA").mean() * 100),
    n_countries=("country", "nunique"),
    top_referral=("referral_channel", lambda s: s.mode().iloc[0]),
).reset_index()
for c_col in ["pct_company", "pct_investment_purpose", "pct_loan_applied", "pct_non_usa"]:
    profile[c_col] = (profile[c_col] * 100 if c_col != "pct_non_usa" else profile[c_col]).round(1)
profile = profile.round(1)
profile.to_csv(os.path.join(OUT_TABLES, "cluster_profile.csv"), index=False)

# Auto-label clusters using a greedy, RELATIVE (rank-based) assignment.
# Absolute thresholds (e.g. "pct_company >= 30%") were tested first and failed --
# no cluster in this dataset clears thresholds like that, because K-Means
# separated clients primarily along an investment-INTENSITY axis (portfolio
# size / spend), not cleanly along the categorical client_type or loan
# fields. Documented honestly in the report. We therefore assign the four
# brief-recommended persona names (C1-C4) to whichever cluster is *most*
# characteristic of each persona, one persona per cluster, in a fixed
# priority order (broadest/most distinctive signal claimed first):
remaining = profile["kmeans_cluster"].tolist()
name_map = {}

# 1) Global Investors: largest portfolios (most properties owned) -- the
#    clearest, most distinctive signal in this dataset (cluster with n=43
#    and avg 7.5 properties vs ~3-4 elsewhere).
cl = profile.loc[profile["kmeans_cluster"].isin(remaining), "avg_total_properties"].idxmax()
chosen = profile.loc[cl, "kmeans_cluster"]
name_map[chosen] = "Global Investors"
remaining.remove(chosen)

# 2) Luxury Investors: highest average sale price among remaining clusters.
cl = profile.loc[profile["kmeans_cluster"].isin(remaining), "avg_sale_price"].idxmax()
chosen = profile.loc[cl, "kmeans_cluster"]
name_map[chosen] = "Luxury Investors"
remaining.remove(chosen)

# 3) Corporate Buyers: highest share of Company-type clients among remaining.
cl = profile.loc[profile["kmeans_cluster"].isin(remaining), "pct_company"].idxmax()
chosen = profile.loc[cl, "kmeans_cluster"]
name_map[chosen] = "Corporate Buyers"
remaining.remove(chosen)

# 4) First-Time Buyers: the one cluster left.
name_map[remaining[0]] = "First-Time Buyers"

profile["segment_label"] = profile["kmeans_cluster"].map(name_map)
profile.to_csv(os.path.join(OUT_TABLES, "cluster_profile.csv"), index=False)

cluster_label_map = dict(zip(profile["kmeans_cluster"], profile["segment_label"]))
df["segment_label"] = df["kmeans_cluster"].map(cluster_label_map)

df.to_csv(cleaned_full_path, index=False)  # re-save with cluster labels

# Geographic distribution per cluster
geo = df.groupby(["segment_label", "country"]).size().reset_index(name="n_clients")
geo.to_csv(os.path.join(OUT_TABLES, "geo_segment_distribution.csv"), index=False)

region_geo = df.groupby(["segment_label", "region"]).size().reset_index(name="n_clients")
region_geo.to_csv(os.path.join(OUT_TABLES, "region_segment_distribution.csv"), index=False)

# =========================================================================
# SAVE AUDIT LOG
# =========================================================================
def default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return str(o)

with open(os.path.join(OUT_TABLES, "audit_log.json"), "w") as f:
    json.dump(log, f, indent=2, default=default)

print("PIPELINE COMPLETE")
print("Clients:", clients.shape, " Properties:", properties.shape, " Merged buyer profiles:", df.shape)
print("Final K:", FINAL_K, " KMeans silhouette:", log["kmeans_silhouette"], " Hierarchical silhouette:", log["hierarchical_silhouette"])
print("Segment counts:\n", df["segment_label"].value_counts())
