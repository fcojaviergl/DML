# Motor Vehicle Insurance Data Analysis with Causal Inference

This project performs an in-depth analysis of motor vehicle insurance data, focusing on predicting claim frequency and severity, segmenting policyholders using clustering, and estimating the causal effect of these segments on insurance premiums. It leverages various statistical and machine learning techniques, culminating in a causal inference analysis using Double Machine Learning (DML) and DoWhy.

## Table of Contents

1.  [Project Overview](#project-overview)
2.  [Features](#features)
3.  [Data](#data)
4.  [Setup and Installation](#setup-and-installation)
5.  [How to Run](#how-to-run)
6.  [Key Outputs and Interpretations](#key-outputs-and-interpretations)
7.  [Dependencies](#dependencies)

## Project Overview

The primary goal of this project is to understand the factors influencing motor vehicle insurance premiums. It follows a comprehensive analytical pipeline:

* **Data Preparation:** Cleaning raw insurance data, handling missing values, and engineering new features.
* **Predictive Modeling:** Building Generalized Linear Models (GLMs) to predict claim frequency (using Poisson regression) and claim severity (using Gamma regression).
* **Dimensionality Reduction & Clustering:** Applying Principal Component Analysis (PCA) to reduce feature dimensionality and then KMeans clustering to segment policyholders based on their characteristics.
* **Causal Inference:** Utilizing Double Machine Learning (DML) to estimate the causal effect of cluster membership on insurance premiums, controlling for confounding factors. This moves beyond mere correlation to understand direct causal impacts.
* **Comparative Analysis:** Comparing the predictive power and insights from traditional OLS regression models (with and without cluster information) against the causal estimates from DML.

## Features

* **Robust Data Cleaning:** Imputation of missing `Length` and `Type_fuel` values based on similar vehicle characteristics, and cleaning of `Distribution_channel` placeholders.
* **Feature Engineering:** Derivation of `age` and `years_licence` from date fields.
* **Frequency-Severity Modeling:** Separate GLMs for `frequency` (Poisson) and `severity` (Gamma with log link), which are common practices in actuarial science.
* **Data Transformation:** Polynomial feature generation to capture non-linear relationships and interactions, followed by StandardScaler for normalization.
* **Dimensionality Reduction:** PCA to reduce the high dimensionality of features while retaining 95% of the variance.
* **Policyholder Segmentation:** KMeans clustering to group policyholders into distinct segments based on their transformed characteristics.
* **Double Machine Learning (DML):** Application of `econml.dml.LinearDML` to estimate the causal effect of cluster membership on premium, addressing confounding.
* **Causal Graph Definition:** Explicit definition of a causal graph using DoWhy to formalize assumptions about causal relationships.
* **OLS Regression Comparison:** Standard OLS models are run to provide a baseline for predictive power and to illustrate the difference between association and causation.
* **Visualizations:** A bar plot to visualize the causal effect of each cluster on the premium.

## Data

The project uses a dataset named `Motor vehicle insurance data.csv`. This file should be located in an `Input` directory relative to the script's location (i.e., `./Input/Motor vehicle insurance data.csv`).

The dataset contains various attributes related to motor vehicle insurance policies, including:

* Policyholder demographics (e.g., `Date_birth`, `Date_driving_licence`)
* Vehicle characteristics (e.g., `Year_matriculation`, `Power`, `Type_fuel`, `Weight`)
* Claim information (e.g., `N_claims_year`, `Cost_claims_year`)
* Policy details (e.g., `Premium`, `Distribution_channel`)

## Setup and Installation

1.  **Python:** Ensure you have Python 3.8+ installed.
2.  **Libraries:** All required libraries are listed in `requirements.txt`. Install them using pip:
    ```bash
    pip install -r requirements.txt
    ```

## How to Run

1.  **Data File:** Place your `Motor vehicle insurance data.csv` file inside a folder named `Input` in the same directory as your `main.py` script.
2.  **Execute:** Run the script from your terminal:
    ```bash
    python main.py
    ```
The script will print various summaries and progress messages to the console, and it will display a bar plot visualizing the causal effects of clusters on premiums.

## Key Outputs and Interpretations

The script generates several key outputs, including:

1.  **GLM Summaries (Frequency and Severity):**
    * These summaries provide insights into which PCA components (derived from vehicle and policyholder characteristics) are statistically significant predictors of claim frequency and severity.
    * They show coefficients, standard errors, p-values, and overall model fit statistics for the Poisson (frequency) and Gamma (severity) regressions.

2.  **OLS Regression Summaries (with and without Cluster):**
    * **`model_without_cluster` (Premium ~ predicted_claims_amount):** This summary shows the linear association between the predicted claims amount and the premium. The coefficient for `predicted_claims_amount` indicates how much premium is expected to change for a unit increase in predicted claims, ignoring other factors. This represents a baseline predictive relationship.
    * **`model_with_cluster` (Premium ~ predicted_claims_amount + cluster):** This summary reveals how adding cluster membership affects the premium prediction.
        * **Interpretation:** If the R-squared value for this model is higher, it means that `cluster` information significantly improves the overall predictive power of the premium model. The coefficients for the cluster dummy variables indicate how the premium changes for each specific cluster compared to a baseline cluster, *after accounting for the predicted claims amount*. This shows that cluster membership provides additional, distinct information for predicting premiums.

3.  **DML Treatment Effect Plot ("Causal Effect of Cluster on Premium"):**
    * This bar plot is the core output of the causal inference analysis.
    * **Interpretation:** Each bar represents the **estimated causal effect** of a policyholder being in that specific cluster on their `Premium`, *after strictly controlling for `predicted_claims_amount` as a confounder*.
    * **Negative Values (e.g., -7.5, -10.5, -20):** If all cluster effects are negative, it means that, compared to an implicit reference cluster (often the first cluster ID, e.g., Cluster 0), policyholders in these clusters are estimated to have a **lower premium** by the indicated amount. For instance, a value of -20 for Cluster 3 suggests that being in Cluster 3 *causes* a premium reduction of 20 units compared to the reference cluster, even if their predicted claims amount is the same. This implies that the clusters have successfully identified segments of policyholders who are inherently "cheaper" to insure due to unobserved factors captured by the clustering.

4.  **DoWhy Causal Effect Estimation Result:**
    * This output provides a detailed summary of the causal estimate, including the identified estimand (how the causal effect was isolated based on the graph) and the average treatment effect (ATE) or conditional average treatment effect (CATE) estimated by Propensity Score Matching.

## Dependencies

* `numpy`
* `pandas`
* `scikit-learn`
* `statsmodels`
* `matplotlib`
* `seaborn`
* `python-dateutil`
* `econml`
* `dowhy`
* `networkx` (installed as a dependency of `dowhy`)