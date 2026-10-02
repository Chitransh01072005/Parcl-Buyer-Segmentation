# Machine-Learning-Based Buyer Segmentation and Investment Profiling — Parcl Co. Limited

## Project Overview
An end-to-end machine learning project that segments Parcl's real estate buyers into
data-driven clusters using K-Means and Hierarchical clustering, built as a reproducible
Jupyter notebook, an interactive Streamlit dashboard, and a research paper.

## Business Problem
Parcl Co. Limited sells apartment and office units across a 20-tower portfolio but has
no systematic understanding of who its buyers are: individual home buyers, repeat
investors, international buyers, high-net-worth investors, and first-time buyers are
all treated the same, leading to inefficient marketing and missed investment
opportunities.

## Objectives
- Merge client demographics with actual purchase history into a unified investment profile
- Apply K-Means and Hierarchical clustering to discover natural buyer segments
- Use the Elbow Method and Silhouette Score to select and validate the number of clusters
- Interpret each segment's investment behavior, geography, and financing patterns
- Deliver a filterable dashboard for ongoing segment monitoring

## Datasets
- **`data/clients.csv`** — 2,000 client records: `client_id`, `client_type`
  (Individual/Company), demographics (`gender`, `date_of_birth`, `country`, `region`),
  `acquisition_purpose` (Home/Investment), `loan_applied`, `referral_channel`,
  `satisfaction_score`.
- **`data/properties.csv`** — 10,000 property listings across 20 towers: `listing_id`,
  `tower_number`, `transaction_date`, `unit_category` (Apartment/Office), `floor_area_sqft`,
  `sale_price`, `listing_status` (Sold/Available), `client_ref`.

**Data quality:** zero missing values in `clients.csv`; the 2,695 missing `client_ref`
values in `properties.csv` correspond exactly to unsold (`Available`) listings, not a
defect. Every client has at least 3 sold properties (mean 3.65, max 13).

## Data Cleaning Notes
- `date_of_birth` mixes two delimiters (`-` and `/`). The slash-delimited subset proves
  a month-first format (second component reaches 31), so both groups are parsed
  consistently as month-first — verified by the resulting age distribution being fully
  plausible (23–93 years).
- `transaction_date` is a **monthly snapshot** (always the 1st of the month), not a
  daily timestamp — documented as a granularity limitation.
- `sale_price` was stored as a formatted currency string and cleaned to numeric.

## Methodology
1. **Feature engineering:** per-client investment profile built from `Sold` properties
   — total properties owned, total/average spend, average unit size, portfolio breadth
   (distinct towers), % office purchases, purchase span (months).
2. **Encoding:** one-hot for `client_type`, `acquisition_purpose`, `referral_channel`,
   `gender`; label encoding for higher-cardinality `region` (57 values) and `country`
   (10 values).
3. **Scaling:** `StandardScaler` applied to all numeric features before clustering.
4. **Model selection:** K-Means and Hierarchical (Ward linkage) tested for K=2–10,
   evaluated via Elbow Method and Silhouette Score.
5. **Clustering:** K=4 used (matches the business's four-persona framework), validated
   against the silhouette-optimal K=3 — the difference in quality is small (0.107 vs.
   0.124). K-Means and Hierarchical clustering agreed on 79.6% of client assignments.

## Buyer Segments (Results)

| Segment | Clients | Avg Properties | Avg Total Investment | Avg Sale Price | Avg Age | % Loan Applied | % Company |
|---|---|---|---|---|---|---|---|
| Global Investors | 43 | 7.5 | $2,497,550 | $334,840 | 62.4 | 34.9% | 2.3% |
| Luxury Investors | 614 | 3.5 | $1,479,439 | $429,116 | 52.7 | 38.4% | 6.2% |
| Corporate Buyers | 749 | 4.1 | $1,270,493 | $309,674 | 53.7 | 36.7% | 5.3% |
| First-Time Buyers | 594 | 3.0 | $931,619 | $310,368 | 53.8 | 35.4% | 4.0% |

Segment names were assigned using a **relative** rule (not fixed thresholds), since no
cluster cleared absolute business thresholds: Global Investors = largest average
portfolio; Luxury Investors = highest average sale price among the rest; Corporate
Buyers = highest company-type share among the rest; First-Time Buyers = the remainder.

## ⚠️ Key Finding: Not every assumed persona matched its cluster
- **First-Time Buyers** does **not** skew young or loan-dependent (avg. age 53.8,
  *lowest* loan-applied rate of all four segments) — it's better understood as a
  mainstream/baseline buyer group than a literal first-time-buyer persona.
- **Company-type clients are not segment-exclusive** — Luxury Investors actually has a
  higher company share (6.2%) than the Corporate Buyers segment (5.3%).
- Silhouette scores (0.08–0.12 across all tested K) indicate **moderate, not sharp**,
  cluster separation — segment boundaries are directional tendencies, not hard
  categories.

These findings are reported directly rather than concealed or relabeled to fit
assumptions — see the research paper for full discussion.

## Technologies Used
Python, pandas, NumPy, scikit-learn, SciPy, Matplotlib, Seaborn, Plotly, Streamlit, Jupyter.

## Project Structure
```
Parcl-Buyer-Segmentation/
├── data/                      Raw datasets (clients.csv, properties.csv)
├── notebooks/                 Full analysis notebook
├── app/                       Streamlit dashboard (app.py)
├── outputs/
│   ├── charts/                Saved PNG charts (13)
│   ├── tables/                Cluster profile, audit log, dendrogram data
│   └── cleaned_data/          Cleaned & merged client/property data with cluster labels
├── report/                    Research paper (.docx)
├── src/                       Reusable pipeline code (run_pipeline.py)
├── requirements.txt
├── README.md
└── .gitignore
```

## How to Run

### Notebook
```bash
pip install -r requirements.txt
jupyter notebook notebooks/parcl_buyer_segmentation.ipynb
```

### Regenerate all tables/charts from scratch
```bash
python src/run_pipeline.py
```

### Dashboard
```bash
pip install -r requirements.txt
streamlit run app/app.py
```

## Business Recommendations
1. Build a dedicated relationship-management track for Global Investors (43 clients,
   $2.50M average lifetime investment).
2. Position Luxury Investors as the primary audience for premium unit launches.
3. Do not build a "First-Time Buyer" campaign on the assumed young/loan-dependent
   persona without re-validating against this segment's actual profile.
4. Create a combined view of high-company-share clients across Luxury and Corporate
   segments for B2B account management.
5. Investigate satisfaction drivers independently of segment (satisfaction is fairly
   uniform, 2.9–3.4/5, across all four segments).

## Limitations
- Silhouette scores indicate moderate cluster separation — not sharp, well-isolated groups.
- `transaction_date` has only monthly granularity.
- K=4 was chosen to match the business framework rather than the silhouette-optimal K=3.
- Segment names reflect a best-match assignment rule; two notable mismatches are
  documented above and in the research paper.

## Future Work
- Validate segments against actual marketing-campaign outcomes
- Daily-resolution transaction timestamps for precise timing features
- Test DBSCAN / Gaussian Mixture Models for potentially better fit
- Periodic re-clustering to track segment drift over time

## Author
*Chitransh Mathur*
