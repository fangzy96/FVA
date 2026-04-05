import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import roc_curve, auc, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, average_precision_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from scipy.stats import chi2
from matplotlib import rcParams

from datetime import timedelta
import statsmodels.api as sm

# Load data
file_path = "clustering_results/All_types_of_surgery_sf36_clustered_results.csv"
file_path_approach = "Chart_review_20250616_processed.csv"
# input_folder = "sleep_disturb_lessthan3hrs_pearson_correlation_csv_results"
data = pd.read_csv(file_path)
data['cluster'] = 3 - data['cluster']  # swap 1<->2
approach_data = pd.read_csv(file_path_approach, encoding='ISO-8859-1')


# Classify surgery approach (Open vs Minimally Invasive)
def classify_approach(row):
    open_cols = ['thoracic_approach___1', 'thoracic_approach___3', 'thoracic_approach___5']
    minimally_cols = ['thoracic_approach___2', 'thoracic_approach___4']

    if any(row.get(col) == 1 for col in open_cols):
        return "Open Surgery"
    elif any(row.get(col) == 1 for col in minimally_cols):
        return "Minimally Invasive Surgery"
    else:
        return ""


# Select and rename to match data's uuid column
approach_cols = ['patient_uuid',
                 'thoracic_approach___1', 'thoracic_approach___2',
                 'thoracic_approach___3', 'thoracic_approach___4',
                 'thoracic_approach___5']
approach_data = approach_data[approach_cols].rename(columns={'patient_uuid': 'uuid'})

# Merge data
data = pd.merge(data, approach_data, on='uuid', how='left')
# Apply classification
data['approach_type'] = data.apply(classify_approach, axis=1)

# Preview results
# print(data[['uuid', 'approach_type']])
# print(data['approach_type'].value_counts())

# === MCS/PCS computation ===

# Z-score parameters and mapping
zscore_params = {
    'PF': (84.52404, 22.89490), 'RP': (81.19907, 33.79729), 'BP': (75.49196, 23.55879),
    'GH': (72.21316, 20.16964), 'VT': (61.05453, 20.86942), 'SF': (83.59753, 22.37642),
    'RE': (81.29467, 33.02717), 'MH': (74.84212, 18.01189)
}

feature_map = {
    'physical_functioning': 'PF', 'limitation_phys': 'RP', 'limitation_emotion': 'RE',
    'energy_fatigue': 'VT', 'emotional_wellbeing': 'MH', 'social_functioning': 'SF',
    'pain': 'BP', 'general_health': 'GH'
}

radar_features = {
    "Pre-Surgery": ['physical_functioning_pre', 'limitation_phys_pre', 'limitation_emotion_pre',
                    'energy_fatigue_pre', 'emotional_wellbeing_pre', 'social_functioning_pre',
                    'pain_pre', 'general_health_pre'],
    "Post 30": ['physical_functioning_post', 'limitation_phys_post', 'limitation_emotion_post',
                'energy_fatigue_post', 'emotional_wellbeing_post', 'social_functioning_post',
                'pain_post', 'general_health_post'],
    "Post 90": ['physical_functioning_90days', 'limitation_phys_90days', 'limitation_emotion_90days',
                'energy_fatigue_90days', 'emotional_wellbeing_90days', 'social_functioning_90days',
                'pain_90days', 'general_health_90days']
}

# Initialize result container
data_scores = pd.DataFrame()

# Compute MCS/PCS per time point
for time_point, features in radar_features.items():
    z_scores = pd.DataFrame(index=data.index)
    sf36_z = {}
    for col in features:
        base_name = col.rsplit('_', 1)[0]
        sf36_code = feature_map[base_name]
        mu, std = zscore_params[sf36_code]
        z = (data[col] - mu) / std
        sf36_z[sf36_code] = z

    agg_phys = (
        sf36_z['PF'] * 0.42402 + sf36_z['RP'] * 0.35119 + sf36_z['BP'] * 0.31754 +
        sf36_z['GH'] * 0.24954 + sf36_z['VT'] * 0.02877 + sf36_z['SF'] * -0.00753 +
        sf36_z['RE'] * -0.19206 + sf36_z['MH'] * -0.22069
    )
    agg_ment = (
        sf36_z['PF'] * -0.22999 + sf36_z['RP'] * -0.12329 + sf36_z['BP'] * -0.09731 +
        sf36_z['GH'] * -0.01571 + sf36_z['VT'] * 0.23534 + sf36_z['SF'] * 0.26876 +
        sf36_z['RE'] * 0.43407 + sf36_z['MH'] * 0.48581
    )

    data[f'PCS_{time_point}'] = 50 + agg_phys * 10
    data[f'MCS_{time_point}'] = 50 + agg_ment * 10

# Compute Delta (change from pre-surgery)
data["PCS_Delta30"] = data["PCS_Post 30"] - data["PCS_Pre-Surgery"]
data["PCS_Delta90"] = data["PCS_Post 90"] - data["PCS_Pre-Surgery"]
data["MCS_Delta30"] = data["MCS_Post 30"] - data["MCS_Pre-Surgery"]
data["MCS_Delta90"] = data["MCS_Post 90"] - data["MCS_Pre-Surgery"]


# Preprocess data
# data = data.dropna(subset=["PCS_Post 30"])
# data = data.dropna(subset=["PCS_Delta30"])
data = data.dropna(subset=["general_health_pre", "general_health_post"])
# data = data.dropna(subset=["energy_fatigue_pre", "energy_fatigue_post"])

# data = data.dropna(subset=["energy_fatigue_post", "energy_fatigue_pre"])
# data['delta'] = data['energy_fatigue_post'] - data['energy_fatigue_pre']
# data = data.dropna(subset=["general_health_post", "general_health_pre"])
data['delta'] = data['general_health_post'] - data['general_health_pre']
# median_delta = data['delta'].median()
# data['target'] = (data['delta'] <= median_delta).astype(int)
threshold = data['delta'].quantile(0.25)
# print(threshold)
data['target'] = (data['delta'] <= threshold).astype(int)
# Count per cluster: how many have delta <= threshold (target=1)
for c in sorted(data['cluster'].unique()):
    n_target1 = ((data['cluster'] == c) & (data['target'] == 1)).sum()
    n_total = (data['cluster'] == c).sum()
    print(f"Cluster {c}: {n_target1}/{n_total} with delta <= threshold (target=1)")
# median_delta = data['PCS_Post 30'].median()
# threshold = data['PCS_Post 30'].quantile(0.5)
# data['target'] = (data['PCS_Post 30'] <= threshold).astype(int)
# data['target'] = (data['PCS_Post 30'] <= median_delta).astype(int)
# threshold = data['PCS_Delta30'].quantile(0.5)
# data['target'] = (data['PCS_Delta30'] <= threshold).astype(int)
# print(data['target'])
# print('Median Delta:', median_delta)

# Ensure cluster column is categorical and convert to binary labels (0, 1)
data['cluster'] = data['cluster'].astype(str)
data['cluster'] = data['cluster'].astype(int) - 1  # Convert "1,2" -> "0,1"
# Count target distribution
cluster_counts = data['target'].value_counts()
print(cluster_counts)

data['surgery_date'] = pd.to_datetime(data['surgery_date'])
data['complication_dates'] = data['complication_dates'].astype(str)
def count_complications(row):
    surgery_date = row['surgery_date']
    complication_dates = row['complication_dates']

    # Return 0 if no complication dates
    if pd.isna(complication_dates) or complication_dates.strip() == "":
        return 0

    # Split complication dates into list
    dates_list = complication_dates.split(',')

    # Count complications within POD2-7 (surgery+2 to surgery+7 days, inclusive)
    pod2 = surgery_date + timedelta(days=2)
    pod7 = surgery_date + timedelta(days=7)
    count = 0
    for date_str in dates_list:
        if date_str.strip() == "":
            continue
        complication_date = pd.to_datetime(date_str)
        if pod2 <= complication_date <= pod7:
            count += 1

    return count
# Apply function
data['complication_count'] = data.apply(count_complications, axis=1)
print("\n--- complication_count (POD2-7) ---")
print(data["complication_count"].value_counts().sort_index())
print(
    f"  n={len(data)}, sum(count)={data['complication_count'].sum()}, "
    f"mean={data['complication_count'].mean():.3f}, max={data['complication_count'].max()}"
)
print(
    f"  count==0: {(data['complication_count'] == 0).sum()}, "
    f"count>=1: {(data['complication_count'] >= 1).sum()}"
)
data['complication_label'] = np.where(data['complication_count'] > 0, 1, 0)
print(f"  complication_label 0/1: {data['complication_label'].value_counts().to_dict()}")
data['sex'] = np.where(data['sex'] > 0, 'female', 'male')
# data['age'] = np.where(data['age'] >= 60, '>=60', '<60')
# data['pft_dlco'] = np.where(data['pft_dlco'] <50, 'High Risk', 'Non High Risk')
data['cluster'] = np.where(data['cluster'] > 0, 'Poor Sleep', 'Normal Sleep')

# Mapping dictionaries
smoking_mapping = {
    0: "Never",
    1: "Ever",
    2: "Ever",
    3: "Ever"
}
# Convert numeric to labels
data['smoking_status'] = data['smoking_status'].replace(smoking_mapping)
surgery_mapping = {
    'Wedge Resection, Lobectomy': "Lobectomy",
    'Wedge Resection, Segmentectomy': "Sublobar Resection",
    'Segmentectomy': "Sublobar Resection",
    'Wedge Resection': "Sublobar Resection",
    'Extended Lobectomy': "Lobectomy"
}
race_mapping = {
    'White, American Indian or Alaska Native': "Other",
}

data['surgery_type'] = data['surgery_type'].replace(surgery_mapping)
data['patient_race'] = data['patient_race'].apply(lambda x: "White" if x == "White" else "Other")
# Convert categorical features with reference groups
data['sex'] = pd.Categorical(data['sex'], categories=['female', 'male'])  # Female = reference
# data['patient_race'] = pd.Categorical(data['patient_race'], categories=['Asian', 'White'])  # White = reference
data['cluster'] = pd.Categorical(data['cluster'], categories=['Normal Sleep', 'Poor Sleep'])
data['surgery_type'] = pd.Categorical(data['surgery_type'], categories=['Sublobar Resection', 'Lobectomy'])
data['smoking_status'] = pd.Categorical(data['smoking_status'], categories=['Never', 'Ever'])  # Non-smoker = reference
data['approach_type'] = pd.Categorical(data['approach_type'], categories=['Minimally Invasive Surgery', 'Open Surgery'])

# category_features = ['cluster', 'sex', 'surgery_type', 'patient_race', 'approach_type'] # 'sex', 'patient_race', 'surgery_type'
# numerical_features = ['age', 'pft_dlco', 'pft_fev1', 'bmi'] # 'pft_dlco', 'pft_fev1', 'pft_fvc', 'pft_fev1_fvc',
category_features = ['cluster', 'sex', 'surgery_type', 'patient_race', 'approach_type', 'smoking_status', 'complication_label'] # 'sex', 'patient_race', 'surgery_type'
numerical_features = ['age', 'pft_dlco', 'pft_fev1', 'bmi', 'general_health_pre'] # 'pft_dlco', 'pft_fev1', 'pft_fvc', 'pft_fev1_fvc', , 'general_health_pre'
all_features = category_features + numerical_features
dependent_variable = 'target'

def preprocess_data(X):
    X = X.copy()
    numerical_cols = X.select_dtypes(include=['float64', 'int64']).columns
    categorical_cols = X.select_dtypes(include=['object']).columns

    for col in numerical_cols:
        X[col] = X[col].fillna(X[col].mean())
    for col in categorical_cols:
        mode_value = X[col].mode()
        X[col] = X[col].fillna(mode_value.iloc[0] if not mode_value.empty else "Unknown")
    return X

# Prepare independent (X) and dependent (Y) variables
X = data[all_features]
X = preprocess_data(X)

# X = X.dropna()
y = data[dependent_variable]
# print(X)
# One-hot encode categorical features
# X = X.dropna()
# y = y[X.index]  # Ensure y matches X's index
# correlation_matrix = X.corr()
# print(correlation_matrix)

# Convert categorical features to numerical using one-hot encoding
X = pd.get_dummies(X, drop_first=True, dtype=float)  # Ensure numeric dtype

# Ensure y is numeric
y = y.astype(int)


# Add constant term for the intercept
X = sm.add_constant(X)
print(X.columns)
# Fit logistic regression model
model = sm.Logit(y, X)
result = model.fit()

# Extract coefficients, standard errors, and p-values
beta = result.params  # Coefficients
SE = result.bse  # Standard errors
p_values = result.pvalues  # P-values

# Compute ARR (Odds Ratio) and 95% CI
ARR = np.exp(beta)
CI_lower = np.exp(beta - 1.96 * SE)
CI_upper = np.exp(beta + 1.96 * SE)

# Create a summary table
summary_df = pd.DataFrame({
    "Feature": beta.index,
    "Beta": beta.values,
    "ARR (Odds Ratio)": ARR.values,
    "95% CI Lower": CI_lower.values,
    "95% CI Upper": CI_upper.values,
    "P-value": p_values.values
})

# Print the summary table
# Pandas display options
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.colheader_justify', 'left')
print(summary_df)

# Optional: Save results to CSV
summary_df.to_csv("logistic_regression_summary.csv", index=False)

# Reduced model without cluster (for LRT)
features_without_cluster = [col for col in X.columns if not col.startswith('cluster')]
X_reduced = X[features_without_cluster]

# Fit reduced model
reduced_model = sm.Logit(y, X_reduced)
result_reduced = reduced_model.fit()

# Compute LRT statistic
LL_full = result.llf
LL_reduced = result_reduced.llf
LRT_stat = 2 * (LL_full - LL_reduced)

# df = difference in number of parameters between full and reduced model
df_diff = X.shape[1] - X_reduced.shape[1]

# P-value
p_value = chi2.sf(LRT_stat, df_diff)

# Print LRT results
print(f"Likelihood Ratio Test statistic: {LRT_stat:.4f}")
print(f"Degrees of freedom: {df_diff}")
print(f"P-value: {p_value:.4e}")




