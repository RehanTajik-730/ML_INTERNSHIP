# Capstone Report — Lane 4: CTR / Engagement Opportunity Scoring

- **Author:** Rehan Ali
- **Lane:** Lane 4 — CTR / Engagement Opportunity Scoring
- **Repository:** [https://github.com/RehanTajik-730/ML_INTERNSHIP](https://github.com/RehanTajik-730/ML_INTERNSHIP)
- **Deployed Paper:** [https://rehantajik-730.github.io/ML_INTERNSHIP/](https://rehantajik-730.github.io/ML_INTERNSHIP/)
- **Date:** October 12, 2026

---

## 0. Abstract

Content teams managing large enterprise search catalogs frequently encounter quiet organic decay: high-demand pages achieve Google Page 1 rankings but fail to capture expected click-through rates (CTR), leaving significant organic traffic unharvested. Traditional fixed heuristics (such as flagging URLs with $\text{CTR} < 0.50\%$ and $\text{impressions} \ge 500$) fail because they treat position 1.5 and position 19.5 identically and are easily fooled by zero-click noise. Using FlyRank's 30,000-row enterprise search intelligence dataset spanning 32 client domains, we formulated opportunity detection as an honest classification and ranking problem evaluated strictly on unseen client holdouts. Compared to the hand-written baseline rule ($18.0\%$ Precision@50 against a $7.1\%$ test base rate), our Random Forest model achieves $50.0\%$ Precision@50 and $0.936$ ROC AUC ($2.78\times$ measured lift over baseline). The resulting ranked queue translates continuous predictions into an operational content action playbook, providing content editors with an empirical decision-support tool that maximizes recoverable organic traffic under fixed review bandwidth.

---

## 1. Problem Framing

### The Decision Supported
Enterprise editorial and content strategy teams have strictly bounded operational capacity: an editorial pod can typically review, audit, and rewrite metadata or on-page content for only 20 to 50 URLs per week across catalogs containing tens of thousands of indexed pages. The decision this research supports is **workforce prioritization**: *given limited review bandwidth, which specific pages should human editors open first to capture immediate organic traffic uplift?*

### Unit of Analysis & Outputs
- **Unit of Analysis**: Exactly one pseudonymized content item (`content_id`), tied to a pseudonymized client identifier (`client_id`) with 90-day trailing search and engagement signals.
- **Output**: A bounded opportunity score, a probability of underperformance, a prioritized catalog rank, one reason code, and one actionable editorial recommendation label.
- **Action Taken**: An editor audits the live SERP, rewrites the title tag and meta description for intent alignment, implements structured schema (FAQ, product tables), or deepens thin content.

### The Cost of a Wrong Call
- **False Positive Cost**: Editorial labor is squandered attempting to "fix" a page where CTR is structurally suppressed by SERP features (such as Google AI Overviews, knowledge panels, or dominant competitor brand intent) that no content rewrite can overcome.
- **False Negative Cost**: High-upside pages ranking in positions 2–8 on high-volume commercial queries remain neglected, leaking thousands of monthly searchers to competitors.

### Why Machine Learning Helps
Fixed heuristic rules apply arbitrary, static cutoffs that fail across diverse query spaces. Machine learning captures non-linear interactions between average position, logarithmic impression scale, search volume, CPC value, and content intent, distinguishing genuine snippet underperformance from structural SERP behavior.

---

## 2. Data Safety & Data Contract

### Dataset Scope
- **Source**: FlyRank anonymized search intelligence release (`data/raw/content_refresh_anonymized.csv`), built from production BigQuery data pipelines ingesting Google Search Console and Google Analytics 4.
- **Scale**: $30,000$ unique rows, $44$ raw columns, $32$ distinct enterprise client portfolios.
- **Valid Search Population**: $28,795$ rows exhibiting positive impressions and measured Search Console rankings ($\text{avg\_position} > 0$). Rows with $\text{avg\_position} == 0$ ($1,205$ rows) represent missing GSC instrumentation and were quarantined from position modeling.

### Public Safety Guardrails
- **Zero Identifying PII**: All client names, domains, URLs, and search query strings were cryptographically pseudonymized before publication.
- **Leak Guard Compliance**: Data files (`*.csv`, `*.parquet`) are barred from git commits via `.gitignore`. Committed deliverables consist exclusively of executed notebooks, receipts (`*.json`), and static visualization assets.

### Strict Leakage Audit & Deliberate Exclusions
1. **Target Sibling Leakage Exclusion**: `clicks_90d` and `ctr` were strictly excluded from feature inputs. Because the opportunity label identifies below-expected CTR, providing clicks and impressions allows a model to calculate CTR directly ($\text{clicks}/\text{impressions}$), creating artificial $\approx 1.0$ performance.
2. **Product Decision Decoupling**: FlyRank's existing production flags (`health_score`, `priority_score`, `action_type`) were excluded, preventing circular learning of legacy heuristics.
3. **Future Window Isolation**: Forward-looking columns (`impressions_last_30d`, `clicks_last_30d`, `sessions_last_30d`, `trend_direction`, `trend_pct`, `is_declining_label`) were kept strictly out of the feature matrix.

---

## 3. Baseline Model

### The Transparent Hand-Written Heuristic (Week 4)
Before training machine learning algorithms, we established a deterministic, hand-written baseline:
$$\text{visible} = (\text{impressions\_90d} \ge 500), \quad \text{in\_striking} = (0 < \text{avg\_position} \le 20), \quad \text{low\_ctr} = (\text{ctr} < 0.50\%)$$
$$\text{rule\_flag} = \text{visible} \times \text{in\_striking} \times \text{low\_ctr}$$
$$\text{baseline\_score} = \text{rule\_flag} \times \text{impressions\_90d} \times \left(\frac{0.50 - \text{ctr}}{100}\right)$$

### Baseline Performance on Unseen Client Holdout
- **Holdout Base Rate (Random Guessing)**: $7.1\%$
- **Baseline Precision@20**: $0.300$ ($6$ out of top 20 correct)
- **Baseline Precision@50**: $0.180$ ($9$ out of top 50 correct)
- **Baseline Precision@100**: $0.230$
- **Baseline ROC AUC**: $0.917$
- **Baseline Average Precision (PR-AUC)**: $0.304$

While the baseline rule beats random guessing, its Precision@50 of $18.0\%$ means that over $80\%$ of recommendations given to editorial teams are false positives, primarily caused by failing to adjust for rank-dependent CTR decay and zero-click SERPs.

---

## 4. Model & Analysis

### Formulation
We framed opportunity detection as supervised classification and continuous opportunity ranking.

### Target Definition
A page is labeled as an **Honest CTR Opportunity** (`is_ctr_opportunity = 1`) if:
$$\text{impressions\_90d} \ge 500 \quad \land \quad 0 < \text{avg\_position} \le 20 \quad \land \quad \text{ctr} < \text{P50}(\text{ctr} \mid \text{position\_tier})$$
This ensures that a page at position 3 is evaluated against top-3 CTR norms ($\approx 2.7\%$), while a page at position 15 is evaluated against striking-distance norms ($\approx 0.32\%$).

### Feature Set ($15$ Features)
- **Search Scale & Value**: `log_impressions_90d`, `search_volume`, `cpc`, `competition`, `competition_level`.
- **Ranking Dynamics**: `avg_position`, `position_tier`, `impression_tier`.
- **Content Attributes**: `word_count`, `char_count`, `content_type`, `main_intent`.
- **Lifecycle & Traffic Mix**: `content_age_days`, `days_since_last_update`, `ai_traffic_pct`.

### Algorithms Trained
1. **Logistic Regression (L2 penalty, scaled features)**
2. **Decision Tree Classifier ($\text{max\_depth}=4$)**
3. **Random Forest Classifier ($100$ trees, $\text{max\_depth}=8$, $\text{n\_jobs}=-1$)**
4. **Gradient Boosting Classifier ($100$ estimators, $\text{max\_depth}=4$)**

---

## 5. Evaluation

### Grouped Client-Holdout Design
To guarantee honest real-world evaluation, the dataset was split by `client_id`:
- **Training Set**: $26,472$ content items across $24$ clients ($13.7\%$ positive base rate).
- **Test Set**: $2,323$ content items across $8$ completely unseen clients ($7.1\%$ positive base rate).

### Benchmark Comparison Table (Evaluated on Unseen Client Holdout)

| Model Family | Precision@20 | Precision@50 | Precision@100 | ROC AUC | Average Precision | Lift over Baseline (P@50) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Naive Random Pick** | 0.071 | 0.071 | 0.071 | 0.500 | 0.071 | $0.39\times$ |
| **Baseline Rule (Week 4)** | 0.300 | 0.180 | 0.230 | 0.917 | 0.304 | $1.00\times$ (Ref) |
| **Logistic Regression** | 0.550 | 0.520 | 0.390 | 0.874 | 0.303 | $2.89\times$ |
| **Decision Tree (depth=4)** | **0.600** | **0.560** | 0.420 | 0.902 | 0.321 | **$3.11\times$** |
| **Random Forest (depth=8)**| 0.450 | **0.500** | **0.510** | **0.936** | **0.435** | **$2.78\times$** |
| **Gradient Boosting** | 0.500 | 0.440 | 0.440 | 0.928 | 0.403 | $2.44\times$ |

### Error Analysis
- **Top 50 False Positives**: On the holdout set, the Random Forest model made 25 false positive selections in the top 50. Analysis reveals these pages concentrate heavily between average positions $14.5$ and $19.8$, where the position-tier median CTR is low ($0.10\%$) and query-level noise causes borderline misclassifications.
- **Top 50 Hits**: The 25 true positive hits identified high-volume commercial and informational pages in positions $2.1$ through $6.8$ capturing under $0.15\%$ CTR, representing prime candidates for immediate metadata remediation.

---

## 6. Interpretation

### What the Model Found
1. **Position Decouples from Volume**: The strongest predictor of true opportunity is `avg_position` (Gini importance $0.312$), followed by `log_impressions_90d` ($0.245$) and `search_volume` ($0.187$). High impression volume alone does not indicate an actionable opportunity unless the page ranks in upper striking distance ($\text{pos} \le 10$).
2. **Intent Matters**: Commercial queries exhibit significantly higher CTR variance than informational queries at identical positions. Informational queries are increasingly displaced by zero-click Google snippets, whereas commercial queries retain click capture potential when comparison tables and pricing schema are provided.
3. **Content Depth as a Defense**: Word count correlates positively with CTR resilience in positions 5–10; thin content (< 1,200 words) suffers from accelerated CTR decay.

---

## 7. Recommendations & Content Action Playbook

### The Prioritized Playbook
Continuous model probabilities are converted into bounded operational workflows:
$$\text{Priority Score} = P(\text{Opportunity}) \times \ln(1 + \text{impressions\_90d}) \times 10$$

| Action Category | Target Profile | Operational Intervention | Share of Catalog |
| :--- | :--- | :--- | :---: |
| **`review_metadata_and_serp`** | Pos 1–5, Informational, Low CTR | Audit SERP for AI Overviews; rewrite title with compelling hook; add bulleted meta description. | 14.8% |
| **`align_intent_and_comparison`**| Pos 4–10, Commercial intent | Introduce product comparison tables, customer ratings, and structured FAQ schema. | 8.2% |
| **`optimize_conversion_snippet`**| Pos 2–8, Transactional intent | Highlight pricing, availability, and clear calls-to-action directly in the snippet. | 3.4% |
| **`expand_thin_content`** | Word count < 1,200 words | Deepen subject matter coverage to defend rank and enhance snippet relevancy. | 6.1% |
| **`monitor`** | Unflagged / low-volume pages | Passive quarterly monitoring; no immediate manual intervention. | 67.5% |

### The Human-in-the-Loop No-Go List
1. **Never Automate Title Rewrites Blindly**: Automated generation risks factual inaccuracies and brand compliance violations.
2. **SERP Layout Verification Required**: Human editors must inspect the live SERP to verify whether rich features (Shopping Ads, AI Overviews) make click capture impossible.
3. **Cannibalization Check**: Confirm that another URL on the same domain is not already ranking higher for the target term.

---

## 8. Reproducibility

### Environment & Run Commands
- **Python Version**: Python 3.14.7 (compatible with Python $\ge 3.10$)
- **Core Dependencies**: `pandas>=2.2`, `numpy>=1.26`, `scikit-learn>=1.4`, `matplotlib>=3.8`, `reportlab>=4.0`, `duckdb>=1.0`
- **Random Seed**: `42` fixed across all numpy, pandas, and scikit-learn operations.
- **Execution Workflow**:
  ```bash
  # Clone repository
  git clone https://github.com/RehanTajik-730/ML_INTERNSHIP.git
  cd ML_INTERNSHIP

  # Install requirements
  pip install -r requirements.txt

  # Run full capstone notebook
  python -c "import nbformat; from nbclient import NotebookClient; nb = nbformat.read('work/notebooks/capstone.ipynb', as_version=4); NotebookClient(nb, timeout=600).execute()"
  ```
- **Sealed Receipts**:
  - `work/outputs/baseline_metrics.json`
  - `work/outputs/playbook_queue_sample.csv`
  - `docs/img/fig1_signal_distributions.png` through `fig4_action_distribution.png`

---

## 9. Acknowledgments & Data Credit

Built on the **FlyRank ML Internship dataset** ([https://flyrank.ai](https://flyrank.ai/)). We thank the FlyRank Applied Search Intelligence team for curating and providing this large-scale enterprise search intelligence benchmark.
