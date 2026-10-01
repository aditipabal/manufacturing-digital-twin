# Statistical Manufacturing Digital Twin for Semiconductor Yield Optimization

An end-to-end statistical manufacturing analytics project that combines statistical inference, machine learning, process simulation, robustness testing, and business-impact analysis to identify candidate operating regions associated with lower manufacturing failure risk.

The project uses the UCI SECOM semiconductor manufacturing dataset and includes an interactive Streamlit Digital Twin for process what-if analysis.

> **Important:** The SECOM process variables are anonymized. Results represent predictive associations and model-based simulations, not causal process recommendations. Any candidate operating settings would require engineering review and controlled production validation before deployment.

---

## Business Problem

Modern manufacturing facilities collect hundreds of measurements across the production process.

The challenge is not simply predicting whether a unit will fail. Manufacturing teams also need to understand:

- Which process measurements are associated with failure?
- Which signals provide meaningful statistical separation between pass and fail outcomes?
- How should highly correlated or incomplete process variables be handled?
- What happens to modeled risk when process settings change?
- Are proposed process configurations historically plausible?
- Are lower-risk regions robust to normal process variation?
- What could defect reduction mean financially?

This project develops a statistical Digital Twin framework to investigate these questions.

---

## Dataset

The project uses the **UCI SECOM semiconductor manufacturing dataset**.

### Dataset characteristics

- **1,567** manufacturing observations
- **590** anonymized process/sensor measurements
- **1,463** passing observations
- **104** failing observations
- Historical failure rate: approximately **6.64%**
- Significant missing data
- Strong class imbalance
- Multiple constant and highly correlated variables

Because sensor names and physical meanings are anonymized, this project intentionally avoids assigning unsupported engineering interpretations such as temperature, pressure, flow, or voltage.

---

## Project Architecture

```text
Historical Manufacturing Data
            |
            v
     Data Quality Audit
            |
            v
 Statistical Feature Screening
            |
            v
 Redundancy + Missingness Control
            |
            v
   Predictive Risk Model
            |
            v
   Out-of-Fold Validation
            |
            v
   Digital Twin Simulation
            |
            v
 Historical Plausibility Testing
            |
            v
    Robustness Simulation
            |
            v
 Business Scenario Analysis
            |
            v
 Interactive Streamlit Dashboard
```

---

## 1. Data Quality Analysis

The initial analysis identified several important characteristics of the manufacturing data.

### Missing data

- 538 of the original 590 sensors contained at least one missing value.
- Missingness varied substantially across sensors.
- Some variables exceeded 50% missingness.

Rather than blindly removing every incomplete variable or globally imputing before validation, missingness was incorporated into the feature-selection process.

### Constant and redundant variables

- **116 constant sensors** were identified.
- 104 duplicate columns were also detected.
- The duplicate columns were entirely contained within the constant-sensor group.

Removing constant variables reduced the process representation from 590 to **474 candidate sensors**.

---

## 2. Statistical Feature Screening

Feature screening was performed using the **training partition** to reduce direct test-set leakage.

For each candidate sensor:

- PASS and FAIL distributions were compared.
- Welch's t-test was used because equal group variances were not assumed.
- Cohen's d quantified standardized effect size.
- Benjamini-Hochberg False Discovery Rate correction controlled for multiple testing.

Training-only screening produced:

- 54 sensors with raw `p < 0.05`
- 19 sensors significant after FDR correction

Highly correlated significant sensors were then grouped to reduce redundancy.

Sensors with extreme missingness were excluded from the primary model.

### Final primary representation

**10 anonymized process sensors** were retained for the main Digital Twin model.

---

## 3. Predictive Modeling

A class-weighted Random Forest was trained because manufacturing failures represent only a small fraction of observations.

### Held-out test performance

| Metric | Result |
|---|---:|
| ROC-AUC | 0.785 |
| PR-AUC | 0.352 |
| Failure prevalence | 0.067 |

PR-AUC was emphasized because accuracy can be misleading in highly imbalanced manufacturing-failure problems.

For comparison, an unweighted classifier could achieve high overall accuracy while identifying almost none of the actual failures.

---

## 4. Out-of-Fold Validation

Five-fold stratified out-of-fold predictions were generated on the training partition to obtain a more conservative view of model generalization.

OOF performance:

- ROC-AUC: approximately **0.692**
- PR-AUC: approximately **0.147**
- Training failure prevalence: approximately **0.066**

The OOF PR-AUC remained above the prevalence baseline, although substantially below the single held-out estimate.

This difference is reported rather than hidden because it provides a more realistic picture of model uncertainty.

---

## 5. Cost-Sensitive Decision Thresholds

A default probability threshold of `0.50` is not necessarily appropriate for manufacturing failure detection.

False negatives may represent missed defects, while false positives may trigger unnecessary inspection or intervention.

Out-of-fold predictions were therefore evaluated across multiple thresholds and illustrative cost assumptions.

The preferred threshold changed substantially depending on the assumed relative costs of:

- missed failures, and
- unnecessary interventions.

This demonstrates that classification thresholds should be tied to manufacturing economics rather than selected solely from generic ML metrics.

---

## 6. Digital Twin What-If Simulation

The trained model was used as a statistical Digital Twin to evaluate process interventions.

For each selected sensor, candidate settings were tested at empirical training percentiles:

- P05
- P25
- P50
- P75
- P95

All simulated settings were constrained to historically observed training ranges.

The analysis identified multiple sensor settings associated with lower model risk scores.

These results are interpreted as **candidate process settings for controlled testing**, not causal recommendations.

---

## 7. Candidate Operating Profile

A lower-risk process profile was constructed using the median process configuration among the lowest-risk observations identified through out-of-fold predictions.

### Candidate results

| Metric | Result |
|---|---:|
| Historical average RF risk score | 0.135 |
| Candidate RF risk score | 0.020 |
| Relative model risk-score change | -85.2% |
| Historical-distance percentile | 16.5% |

The **-85.2% value represents a change in the model's uncalibrated risk score**. It does **not** mean the process has been demonstrated to reduce real manufacturing failures by 85.2%.

---

## 8. Historical Plausibility

Optimizing each sensor independently can create combinations that rarely or never occur together in real manufacturing data.

To address this, candidate states were evaluated using standardized multivariate distance from historical training observations.

An initially combined independently optimized profile fell near the **97.6th percentile** of historical nearest-neighbor distance, indicating an unusually isolated process state.

That configuration was therefore not treated as a recommended operating profile.

The final OOF-derived candidate profile had a distance percentile of approximately **16.5%**, placing it much closer to historically observed manufacturing states.

---

## 9. Robustness Simulation

A useful operating region should not depend on one exact combination of sensor values.

The final candidate profile was therefore perturbed across **1,000 nearby simulated process states**.

Perturbations used 5% of each sensor's historical standard deviation and were constrained to the empirical P05-P95 operating range.

### Robustness results

| Metric | Risk Score |
|---|---:|
| Candidate risk | 0.020 |
| Mean simulated neighborhood risk | 0.022 |
| Median | 0.017 |
| 90th percentile | 0.047 |
| Maximum observed simulation | 0.153 |

The tested neighborhood generally retained substantially lower model risk scores than the historical average of approximately 0.135.

This suggests that the candidate represents a locally lower-risk model region rather than only a single isolated low-scoring point.

---

## 10. Business Impact Scenario Analysis

The dataset does not provide actual manufacturing economics.

Financial analysis therefore uses explicitly illustrative assumptions:

- Annual production: **100,000 units**
- Cost per failed unit: **$500**
- Historical failure rate: **6.64%**

Under these assumptions, the illustrative baseline annual failure cost is approximately **$3.32 million**.

| Realized Defect Reduction | Approx. Failures Avoided | Illustrative Annual Savings |
|---:|---:|---:|
| 5% | 332 | $166K |
| 10% | 664 | $332K |
| 20% | 1,327 | $664K |
| 30% | 1,991 | $996K |
| 40% | 2,655 | $1.33M |

These are scenario estimates.

The model's risk-score reduction is deliberately **not converted directly into financial savings** because the Random Forest scores are uncalibrated and the simulated interventions have not been causally validated.

---

## Interactive Digital Twin Dashboard

A Streamlit application provides an interactive interface for exploring the model.

Users can:

- Adjust the 10 selected anonymized process variables
- Keep simulations within empirical P05-P95 ranges
- Observe changes in the Random Forest risk score
- Compare simulated states with the historical baseline
- Review the OOF-derived candidate operating profile
- Explore robustness results
- Examine illustrative business-impact scenarios
- Review methodology and limitations

---

## Project Structure

```text
manufacturing-digital-twin/
│
├── dashboard/
│   └── app.py
│
├── data/
│   ├── secom.data
│   ├── secom_labels.data
│   └── secom.names
│
├── models/
│   ├── random_forest_model.joblib
│   ├── median_imputer.joblib
│   └── final_sensor_list.joblib
│
├── notebooks/
│   └── 01_data_understanding.ipynb
│
├── results/
│   ├── business_impact.png
│   ├── candidate_robustness.png
│   ├── single_sensor_simulation.png
│   └── additional analysis outputs
│
├── src/
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Running the Project Locally

Clone the repository and install the required packages:

```bash
pip install -r requirements.txt
```

Launch the Digital Twin:

```bash
streamlit run dashboard/app.py
```

---

## Technology Stack

**Programming & Analytics**

- Python
- pandas
- NumPy
- SciPy
- statsmodels
- scikit-learn

**Modeling**

- Welch's t-test
- Cohen's d
- Benjamini-Hochberg FDR
- Random Forest
- Stratified cross-validation
- Precision-recall analysis
- Cost-sensitive threshold analysis

**Digital Twin**

- What-if process simulation
- Empirical operating constraints
- Multivariate historical-distance analysis
- Monte Carlo / perturbation-based robustness testing

**Visualization & Deployment**

- Matplotlib
- Streamlit
- Joblib
- Jupyter
- Git / GitHub

---

## Key Lessons

This project demonstrates why manufacturing optimization requires more than simply training a classifier.

A useful statistical Digital Twin needs to consider:

1. statistical significance and effect size,
2. class imbalance,
3. feature-selection leakage,
4. process-variable redundancy,
5. missing data,
6. decision economics,
7. multivariate process plausibility,
8. robustness to process variation,
9. predictive versus causal interpretation, and
10. translation of analytical results into business decisions.

---

## Limitations & Future Work

Several limitations remain.

- SECOM variables are anonymized, limiting engineering interpretation.
- The dataset is observational, so simulated interventions do not establish causality.
- Random Forest risk scores are not calibrated probabilities.
- Feature selection was performed on the overall training partition before OOF cross-validation rather than independently inside each CV fold.
- The held-out test set was inspected during iterative development and should not be treated as a completely untouched production-validation set.
- Illustrative financial assumptions are not factory-specific economics.

Future improvements could include:

- nested cross-validation,
- probability calibration,
- permutation or SHAP-based model interpretation,
- engineering-defined process constraints,
- controlled intervention/A-B experimentation,
- drift monitoring,
- real-time sensor ingestion,
- optimization under explicit process constraints, and
- deployment through a production inference API.

---

## Responsible Interpretation

The objective of this project is not to claim that a machine-learning model discovered guaranteed semiconductor process settings.

Instead, the Digital Twin provides a framework for:

**historical data → statistical evidence → predictive modeling → process simulation → candidate operating regions → controlled validation → business decision**

That distinction is essential when applying data science to real manufacturing systems.