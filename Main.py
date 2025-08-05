# -*- coding: utf-8 -*-

# Import necessary libraries
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from dateutil import parser

from sklearn.model_selection import train_test_split
from sklearn.linear_model import PoissonRegressor, GammaRegressor
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.compose import ColumnTransformer # Although not used in the final version, good to keep if considering more complex preprocessing pipelines

from econml.dml import LinearDML # For Double Machine Learning
from dowhy import CausalModel # For Causal Inference with DoWhy

# --- Configuration for Pandas Display ---
# Show all columns when printing DataFrames to get a full view of the data
pd.set_option('display.max_columns', None)
# Optional: prevent rows from being truncated, useful for large datasets
pd.set_option('display.max_rows', 100)

print("--- Step 1: Data Loading and Initial Preprocessing ---")

# --- Helper Function for Date Parsing ---
def safe_parse_date(val):
    """
    Safely parses a date string into a datetime object.
    Returns pd.NaT (Not a Time) if parsing fails.
    Adjust the parse method if your dates are consistently MM/DD/YYYY.
    """
    try:
        # Using dateutil.parser.parse for robust date parsing
        return parser.parse(val)
    except (ValueError, TypeError):
        return pd.NaT

# Load the dataset
print("Loading the motor vehicle insurance data...")
df_original = pd.read_csv("./Input/Motor vehicle insurance data.csv", sep=";")
print("Data loaded successfully. Displaying original DataFrame head:")
print(df_original.head())

# Create a copy of the original DataFrame to work with, preserving the original
df = df_original.copy()

# --- Date Column Conversion and Feature Engineering ---
print("\nConverting date columns and deriving new features (age, years_licence)...")
# Convert specified columns to datetime objects using the safe_parse_date function
df['Date_birth'] = df['Date_birth'].apply(safe_parse_date)
df['Date_driving_licence'] = df['Date_driving_licence'].apply(safe_parse_date)
df['Date_last_renewal'] = df['Date_last_renewal'].apply(safe_parse_date)

# Calculate 'age' of the policyholder at the last renewal date
# by finding the difference in days and converting to years
df['age'] = (df['Date_last_renewal'] - df['Date_birth']).dt.days // 365
# Calculate 'years_licence', representing how long the policyholder has held a driving license
df['years_licence'] = (df['Date_last_renewal'] - df['Date_driving_licence']).dt.days // 365
print("Date conversions and feature derivation complete. Displaying DataFrame head with new features:")
print(df.head())

print("\n--- Step 2: Handling Missing Values and Data Cleaning ---")

# --- Impute Missing 'Length' ---
print("\nImputing missing 'Length' values...")
# Define columns to group by for 'Length' imputation. These columns are expected to be related to vehicle size.
length_group_cols = ['Year_matriculation', 'Power', 'Cylinder_capacity',
                     'Value_vehicle', 'N_doors', 'Weight']

# Fill 'Length' NaN values using the mode (most common value) within each defined group.
# If a group has no mode (e.g., all NaNs or multiple modes), it falls back to np.nan for that group.
df['Length'] = df.groupby(length_group_cols)['Length'] \
    .transform(lambda x: x.fillna(x.mode().iloc[0] if not x.mode().empty else np.nan))

# If there are still any NaN values in 'Length' after group-wise imputation (e.g., groups with no valid data),
# fill them with the global mode of the 'Length' column.
if df['Length'].isna().any():
    fallback_mode = df['Length'].mode()[0]
    df['Length'] = df['Length'].fillna(fallback_mode)
print("Missing 'Length' values imputed.")

# --- Impute Missing 'Type_fuel' ---
print("Imputing missing 'Type_fuel' values...")
# Define columns to group by for 'Type_fuel' imputation. These are similar to 'Length' as they describe the vehicle.
typefuel_group_cols = ['Year_matriculation', 'Power', 'Cylinder_capacity',
                       'Value_vehicle', 'N_doors', 'Weight']

# Fill 'Type_fuel' NaN values using the mode within each defined group.
df['Type_fuel'] = df.groupby(typefuel_group_cols)['Type_fuel'] \
    .transform(lambda x: x.fillna(x.mode().iloc[0] if not x.mode().empty else np.nan))

# Fallback to global mode if any NaNs remain after group-wise imputation.
if df['Type_fuel'].isna().any():
    fallback_mode = df['Type_fuel'].mode()[0]
    df['Type_fuel'] = df['Type_fuel'].fillna(fallback_mode)
print("Missing 'Type_fuel' values imputed.")

# --- Clean 'Distribution_channel' ---
print("Cleaning 'Distribution_channel' column...")
# Identify the placeholder value used for missing or invalid entries.
placeholder = "00/01/1900"

# Find the most common valid value in 'Distribution_channel' (excluding the placeholder).
most_common = df.loc[df['Distribution_channel'] != placeholder, 'Distribution_channel'].mode()[0]

# Replace the identified placeholder with the most common valid value.
df['Distribution_channel'] = df['Distribution_channel'].replace(placeholder, most_common)
# Ensure the column is treated as a string before converting to category.
df['Distribution_channel'] = df['Distribution_channel'].astype(str)
# Convert 'Distribution_channel' to a categorical type for efficient memory usage and proper handling in models.
df['Distribution_channel'] = df['Distribution_channel'].astype('category')

print(f"Unique values in 'Distribution_channel' after cleaning: {df['Distribution_channel'].unique()}")
print("Missing values handled and 'Distribution_channel' cleaned.")

print("\n--- Step 3: Compute Frequency and Severity ---")

# --- Calculate Frequency and Severity ---
print("Calculating 'frequency' and 'severity' target variables...")
# 'frequency' is directly taken from 'N_claims_year'.
df['frequency'] = df['N_claims_year']
# 'severity' is calculated as 'Cost_claims_year' divided by 'N_claims_year'.
# If 'N_claims_year' is 0, 'severity' is set to 0 to avoid division by zero.
df['severity'] = df.apply(
    lambda row: 0 if row['N_claims_year'] == 0 else row['Cost_claims_year'] / row['N_claims_year'],
    axis=1
)
# Replace infinite values (which can occur from division by zero before the lambda check, or other data issues)
# and drop rows where 'severity' is NaN (e.g., if Cost_claims_year was also NaN when N_claims_year was 0).
df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=['severity'])

# Clip 'severity' to the 99th percentile to mitigate the impact of extreme outliers,
# which can disproportionately affect model training.
df['severity'] = df['severity'].clip(upper=df['severity'].quantile(0.99))
print("Frequency and severity calculated and severity clipped.")
print(f"Severity descriptive statistics after clipping:\n{df['severity'].describe()}")

print("\n--- Step 4: Feature Selection and One-Hot Encoding ---")

# Select core features for modeling. These are the independent variables.
core_features = [
    'Distribution_channel', 'Type_risk', 'Area', 'Second_driver', 'Year_matriculation',
    'Power', 'Cylinder_capacity', 'Value_vehicle', 'N_doors', 'Type_fuel',
    'Length', 'Weight', 'age', 'years_licence']

# Keep target variables separately for later use.
target_vars = ['Premium', 'severity', 'frequency']

# Create a reduced DataFrame containing only the selected core features and target variables.
df_reduce = df[core_features + target_vars].copy()
print("Core features selected. Displaying reduced DataFrame head:")
print(df_reduce.head())

# Identify categorical features that need to be one-hot encoded.
categorical_vars = ['Distribution_channel', 'Type_risk', 'Area', 'Second_driver', 'Type_fuel', 'N_doors']
print(f"\nPerforming one-hot encoding for categorical variables: {categorical_vars}...")
# Perform one-hot encoding using pd.get_dummies.
# `drop_first=True` avoids multicollinearity by dropping the first category of each feature.
# `dtype=int` ensures the new dummy columns are integers (0 or 1).
df_encoded = pd.get_dummies(df_reduce, columns=categorical_vars, drop_first=True, dtype=int)
print("One-hot encoding complete. Displaying encoded DataFrame head:")
print(df_encoded.head())

# Separate features (X_base) from target variables (y).
X_base = df_encoded.drop(columns=target_vars)
y = df_encoded[target_vars]
print("\nFeatures and target variables separated.")

print("\n--- Step 5: Polynomial Feature Generation and Scaling ---")

# --- Generate Polynomial and Interaction Terms ---
print("Generating polynomial features (degree=3) and interaction terms...")
# Initialize PolynomialFeatures to create polynomial terms up to degree 3 and all interaction terms.
# `include_bias=False` prevents adding a column of all ones (intercept), as this is typically handled by models.
poly = PolynomialFeatures(degree=3, include_bias=False)
# Fit and transform the base features to create polynomial features.
X_poly = poly.fit_transform(X_base)

# Get the names of the newly generated polynomial features.
poly_feature_names = poly.get_feature_names_out(X_base.columns)

# Convert the transformed array back into a DataFrame with proper column names and index.
X_poly_df = pd.DataFrame(X_poly, columns=poly_feature_names, index=df_encoded.index)
print("Polynomial features generated. Displaying head of polynomial features DataFrame:")
print(X_poly_df.head())

# --- Scale Features ---
print("\nScaling features using StandardScaler...")
# Initialize StandardScaler to normalize features (mean=0, variance=1).
# This is crucial for PCA and many machine learning algorithms.
scaler = StandardScaler()
# Fit the scaler to the polynomial features and transform them.
X_scaled = scaler.fit_transform(X_poly_df)
# Convert the scaled array back into a DataFrame.
X_scaled_df = pd.DataFrame(X_scaled, columns=poly_feature_names, index=df_encoded.index)
print("Features scaled. Displaying head of scaled features DataFrame:")
print(X_scaled_df.head())

print("\n--- Step 6: Principal Component Analysis (PCA) ---")

# --- Perform PCA ---
print("Performing Principal Component Analysis (PCA), retaining 95% of variance...")
# Initialize PCA. `n_components=0.95` means PCA will select the minimum number of components
# required to explain at least 95% of the variance in the data.
pca = PCA(n_components=0.95)
# Fit PCA to the scaled data and transform it.
X_pca = pca.fit_transform(X_scaled_df)

# Generate column names for the PCA components.
pca_columns = [f'pca_{i+1}' for i in range(X_pca.shape[1])]
# Convert the PCA result array into a DataFrame.
X_pca_df = pd.DataFrame(X_pca, columns=pca_columns, index=df_encoded.index)
print(f"PCA complete. Retained {X_pca.shape[1]} components explaining 95% of variance. Displaying head of PCA DataFrame:")
print(X_pca_df.head())

print("\n--- Step 7: Data Preparation for GLM Modeling ---")

# Combine PCA components with the 'frequency' column for frequency modeling.
df_glm = pd.concat([X_pca_df, df_encoded[['frequency']]], axis=1)
print("Combined PCA components with 'frequency' for GLM. Displaying head:")
print(df_glm.head())

# Split the dataset indices into training and testing sets.
# This ensures that the same split can be used for both frequency and severity models,
# maintaining consistency across the analysis.
print("\nSplitting data into training and testing sets (80/20 split)...")
train_idx, test_idx = train_test_split(df_glm.index, test_size=0.2, random_state=42)

# Create training and testing DataFrames using the generated indices.
df_train = df_glm.loc[train_idx]
df_test = df_glm.loc[test_idx]
print("Data split complete.")

print("\n--- Step 8: Generalized Linear Model (GLM) for Frequency ---")

# Build the GLM formula for frequency.
# 'frequency' is the dependent variable, and all PCA components are independent variables.
formula = 'frequency ~ ' + ' + '.join(X_pca_df.columns)
print(f"\nFitting Poisson regression model for frequency with formula: '{formula}'...")

# Fit a Poisson regression model to the training data.
# Poisson family is appropriate for count data like 'frequency'.
glm_freq = smf.glm(
    formula=formula,
    data=df_train,
    family=sm.families.Poisson()
).fit()

# Print the summary of the fitted Poisson model, which includes coefficients, p-values, etc.
print("\nPoisson Regression Model Summary for Frequency:")
print(glm_freq.summary())

# Predict 'frequency' on the test set using the fitted model.
print("\nPredicting frequency on the test set...")
y_pred_test = glm_freq.predict(df_test)
print("Frequency predictions generated. Displaying first few predictions:")
print(y_pred_test.head())

print("\n--- Step 9: Generalized Linear Model (GLM) for Severity ---")

# Prepare data for severity modeling by combining PCA components with 'severity'.
df_severity = pd.concat([X_pca_df, df_encoded[['severity']]], axis=1)
# Create training and testing DataFrames for severity using the same indices as frequency.
df_sev_train = df_severity.loc[train_idx]
df_sev_test = df_severity.loc[test_idx]
print("\nPrepared data for severity GLM. Displaying head of severity training data:")
print(df_sev_train.head())

# Build the GLM formula for severity.
formula_sev = 'severity ~ ' + ' + '.join(X_pca_df.columns)
print(f"\nFitting Gamma regression model for severity with formula: '{formula_sev}'...")

# Fit a Gamma regression model to the training data.
# Gamma distribution with a log link is commonly used for positively skewed continuous data like 'severity'.
glm_sev = smf.glm(
    formula=formula_sev,
    data=df_sev_train,
    family=sm.families.Gamma(link=sm.families.links.log())
).fit()

# Print the summary of the fitted Gamma model.
print("\nGamma Regression Model Summary for Severity:")
print(glm_sev.summary())

# Predict 'severity' on the test set.
print("\nPredicting severity on the test set...")
y_sev_pred_test = glm_sev.predict(df_sev_test)
print("Severity predictions generated. Displaying first few predictions:")
print(y_sev_pred_test.head())

print("\n--- Step 10: Combining Predictions and Preparing Final Test Data ---")

# Remove the actual 'frequency' column from the test dataset to avoid confusion with predictions.
df_test = df_test.drop(columns=['frequency'])

# Add the predicted frequency and severity to the test dataset.
df_test['predicted_frequency'] = y_pred_test
df_test['predicted_severity'] = y_sev_pred_test
print("\nAdded predicted frequency and severity to the test DataFrame. Displaying head:")
print(df_test[['predicted_frequency', 'predicted_severity']].head())

print("\n--- Step 11: KMeans Clustering on PCA Components ---")

# Perform KMeans clustering on the PCA components from the training set.
# This groups similar policyholders based on their vehicle and demographic characteristics.
print("Performing KMeans clustering (n_clusters=10) on PCA components for training data...")
kmeans = KMeans(n_clusters=10, random_state=42, n_init=10) # Added n_init for robustness
# Fit the KMeans model on the training PCA components and assign cluster labels.
df_train['cluster'] = kmeans.fit_predict(X_pca_df.loc[train_idx])
print("KMeans clustering fitted on training data. Displaying first few cluster assignments for training data:")
print(df_train[['cluster']].head())

# Apply the same KMeans model to the test set to assign clusters based on the learned centroids.
print("\nAssigning clusters to the test data using the fitted KMeans model...")
df_test['cluster'] = kmeans.predict(X_pca_df.loc[test_idx])
print("Clusters assigned to test data. Displaying first few cluster assignments for test data:")
print(df_test[['cluster']].head())

print("\n--- Step 12: Calculate Predicted Claims Amount and Prepare for DML ---")

# Calculate the predicted claims amount as the product of predicted frequency and predicted severity.
df_test['predicted_claims_amount'] = df_test['predicted_frequency'] * df_test['predicted_severity']

# Add the original 'Premium' column from the encoded DataFrame back to the test DataFrame
# to use it as the outcome variable for causal inference.
df_test['Premium'] = df_encoded.loc[test_idx, 'Premium']

# Prepare the final test DataFrame with relevant columns for DML analysis.
test_df = df_test[['Premium', 'predicted_claims_amount', 'cluster']]
print("\nCalculated predicted claims amount and prepared final test DataFrame for DML. Displaying head:")
print(test_df.head())

print("\n--- Step 13: Double Machine Learning (DML) for Causal Effect Estimation ---")

# Ensure that 'test_df' has all the columns needed for DML.
# X: Confounders (variables that influence both treatment and outcome).
# T: Treatment variable (the variable whose causal effect we want to estimate).
# y: Outcome variable (the variable that is affected by the treatment).
X_dml = test_df[['predicted_claims_amount']]  # Confounders: Predicted claims amount
T_dml = test_df['cluster']  # Treatment variable: Cluster assignment
y_dml = test_df['Premium']  # Outcome variable: Actual Premium

# Split the data into train/test for cross-fitting within DML.
# This split is internal to the DML process and helps in robust estimation.
print("\nSplitting data for DML cross-fitting...")
X_train_dml, X_test_dml, T_train_dml, T_test_dml, y_train_dml, y_test_dml = train_test_split(
    X_dml, T_dml, y_dml, test_size=0.2, random_state=42
)
print("DML data split complete.")

# Initialize the LinearDML estimator.
# model_y: Machine learning model for the outcome (y) given confounders (X). RandomForestRegressor is used.
# model_t: Machine learning model for the treatment (T) given confounders (X). RandomForestClassifier is used
#          because the treatment (cluster) is discrete.
# discrete_treatment=True: Specifies that the treatment variable is categorical.
print("\nInitializing LinearDML model...")
model_dml = LinearDML(model_y=RandomForestRegressor(random_state=42),
                      model_t=RandomForestClassifier(random_state=42),
                      discrete_treatment=True,
                      random_state=42)

# Fit the DML model. This involves training the outcome and treatment models
# on different folds of the data (cross-fitting) to estimate the nuisance functions.
print("Fitting the LinearDML model...")
model_dml.fit(y_train_dml, T_train_dml, X=X_train_dml)
print("LinearDML model fitted.")

# Estimate the treatment effect on the test set.
# The `effect` method calculates the conditional average treatment effect (CATE)
# or average treatment effect (ATE) based on the model.
print("\nEstimating treatment effect on the DML test set...")
treatment_effect = model_dml.effect(X_test_dml)
print("Treatment effects estimated. Displaying first few estimated effects:")
print(treatment_effect[:5]) # Display first 5 effects

# Estimate the treatment effect for the full dataset (X_full is the same as X_dml).
print("\nEstimating treatment effect for the full dataset...")
treatment_effect_full = model_dml.effect(X_dml)

# Add the estimated treatment effect back to the original df_test DataFrame.
df_test['treatment_effect'] = treatment_effect_full
print("Treatment effects added to df_test. Displaying head of cluster and treatment effect:")
print(df_test[['cluster', 'treatment_effect']].head())

print("\n--- Step 14: Visualize Treatment Effects ---")

# Set up the plot size.
plt.figure(figsize=(10, 6))

# Plot the average treatment effects for each cluster using a bar plot.
# This visualizes how being in a particular cluster (treatment) affects the Premium (outcome).
print("Generating bar plot of causal effect of cluster on Premium...")
sns.barplot(x='cluster', y='treatment_effect', data=df_test, palette="viridis")

# Customize the plot for better readability.
plt.title('Causal Effect of Cluster on Premium (Estimated by DML)')
plt.xlabel('Cluster')
plt.ylabel('Treatment Effect (Change in Premium)')
plt.xticks(rotation=45) # Rotate x-axis labels for better visibility if many clusters
plt.tight_layout() # Adjust layout to prevent labels from overlapping

# Show the plot.
plt.show()
print("Plot displayed.")

# --- Summary Statistics for the Treatment Effect ---
print("\n--- Step 15: Summary Statistics of Treatment Effect ---")
print("Calculating summary statistics for the estimated treatment effect...")
treatment_effect_summary = df_test['treatment_effect'].describe()
print("Treatment Effect Summary:")
print(treatment_effect_summary)

print("\n--- Step 16: OLS Regression for Comparison ---")

# --- Method 1: OLS Regression (Linear Model) ---
# This section performs traditional OLS regressions to compare with causal inference results.

# Prepare the data for regression with both 'predicted_claims_amount' and 'cluster'.
# `sm.add_constant` adds an intercept term to the independent variables.
print("\nFitting OLS model with 'predicted_claims_amount' and 'cluster'...")
X_train_ols = test_df[['predicted_claims_amount', 'cluster']]
X_train_ols = sm.add_constant(X_train_ols)
y_train_ols = test_df['Premium']

# Fit the OLS model.
model_with_cluster = sm.OLS(y_train_ols, X_train_ols).fit()

# Display the summary of the model including the cluster effect.
print("\nOLS Model Summary (with predicted_claims_amount and cluster):")
print(model_with_cluster.summary())

# Fit the model with only 'predicted_claims_amount' (without 'cluster') for comparison.
print("\nFitting OLS model with only 'predicted_claims_amount'...")
X_train_ols_no_cluster = test_df[['predicted_claims_amount']]
X_train_ols_no_cluster = sm.add_constant(X_train_ols_no_cluster)

model_without_cluster = sm.OLS(y_train_ols, X_train_ols_no_cluster).fit()

# Display the summary of the model without the cluster effect.
print("\nOLS Model Summary (without cluster):")
print(model_without_cluster.summary())

print("\n--- Step 17: Causal Inference with DoWhy ---")

# Define the causal graph as a string in DOT language (DAG).
# This graph represents our assumptions about the causal relationships between variables.
# Predicted_claims_amount -> Premium: Higher predicted claims lead to higher premium.
# Cluster -> Premium: Being in a certain cluster affects the premium.
# Predicted_claims_amount -> Cluster: Predicted claims amount might influence cluster assignment.
causal_graph = """
digraph {
    Predicted_claims_amount -> Premium;
    cluster -> Premium;
    Predicted_claims_amount -> cluster;
}
"""
print("\nDefined causal graph for DoWhy:")
print(causal_graph)

# Create the causal model using DoWhy.
# data: The DataFrame containing the variables.
# graph: The causal graph defined above.
# treatment: The variable whose causal effect we want to estimate.
# outcome: The variable that is affected by the treatment.
print("Creating DoWhy Causal Model...")
model_dowhy = CausalModel(
    data=test_df, # Using test_df which contains 'Premium', 'predicted_claims_amount', 'cluster'
    graph=causal_graph,
    treatment="cluster",  # 'cluster' is the treatment variable
    outcome="Premium",  # 'Premium' is the outcome variable
)
print("DoWhy Causal Model created.")

# Identify the causal effect using the back-door criterion.
# This step finds a set of confounders that need to be controlled for to estimate the causal effect.
print("\nIdentifying causal effect using back-door criterion...")
identified_estimand = model_dowhy.identify_effect()
print("Causal effect identified.")
print(identified_estimand) # Print the identified estimand for review

# Estimate the causal effect using Propensity Score Matching.
# This method matches treated and control units based on their propensity scores (probability of receiving treatment).
print("\nEstimating causal effect using Propensity Score Matching (backdoor.propensity_score_matching)...")
causal_estimate = model_dowhy.estimate_effect(identified_estimand,
                                              method_name="backdoor.propensity_score_matching")

# Print the causal effect estimation result.
print("\nDoWhy Causal Effect Estimation Result:")
print(causal_estimate)

# Visualize the causal graph. This will open a new window or display the graph inline if supported.
print("\nVisualizing the causal model graph...")
model_dowhy.view_model()
print("Analysis complete!")