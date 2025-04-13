import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer


df_converted = pd.read_excel(r"../data/cleaned_data.xlsx")
print(df_converted)
# Correlation heatmap for numeric features
numeric_df = df_converted.select_dtypes(include=[np.number])
corr_matrix = numeric_df.corr()

# Optional: Drop highly correlated features (correlation > 0.95)
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop = [column for column in upper.columns if any(upper[column].abs() > 0.95)]
df_cleaned_filtered = df_converted.drop(columns=to_drop, errors='ignore')

# Re-impute numeric columns (just in case new NaNs were introduced)
columns_to_convert = df_cleaned_filtered.select_dtypes(include=[np.number]).columns.tolist()
# --- Handle numeric columns ---
numeric_cols = df_cleaned_filtered.select_dtypes(include=[np.number]).columns.tolist()

# Only impute columns that are not fully null
numeric_cols_to_impute = [col for col in numeric_cols if df_cleaned_filtered[col].notnull().any()]

imputer = SimpleImputer(strategy='median')
df_cleaned_filtered[numeric_cols_to_impute] = imputer.fit_transform(df_cleaned_filtered[numeric_cols_to_impute])

for col in ['field/lab', 'texturalclass']:
    if col in df_cleaned_filtered.columns:
        mode_val = df_cleaned_filtered[col].mode()
        if not mode_val.empty:
            df_cleaned_filtered[col] = df_cleaned_filtered[col].fillna(mode_val[0])
        else:
            print(f"⚠️ Skipped mode imputation for '{col}' — no mode found (all values may be NaN).")

# Normalize case (only for field/lab if present)
if 'field/lab' in df_cleaned_filtered.columns:
    df_cleaned_filtered['field/lab'] = df_cleaned_filtered['field/lab'].astype(str).str.lower()

# Drop sitelabel (not useful for modeling)
df_cleaned_filtered = df_cleaned_filtered.drop(columns=['sitelabel'])

# Normalize 'field/lab' values to lowercase
df_cleaned_filtered['field/lab'] = df_cleaned_filtered['field/lab'].str.lower()

# Convert Ksat to cm/hr based on unit conversion factors
unit_conversion = {
    'cm/hr': 1, 'cm/min': 60, 'in/hr': 2.54,
    'm d^-1': 100 / 24, 'mm h^-1': 0.1, 'mm/h': 0.1, 'um s^-1': 0.00036
}

def convert_ksat(row):
    unit = row['units']
    try:
        ksat = float(row['ksat'])
        factor = unit_conversion.get(unit)
        return ksat * factor if factor is not None else None
    except (ValueError, TypeError):
        return None

# Apply conversion and clean final dataset
df_cleaned_filtered['ksat_cm_hr'] = df_cleaned_filtered.apply(convert_ksat, axis=1)
print(df_cleaned_filtered.columns)
df_cleaned_filtered = df_cleaned_filtered[df_cleaned_filtered['ksat_cm_hr'].notnull()]
print(df_cleaned_filtered)
df_cleaned_filtered = df_cleaned_filtered.drop(columns=['units', 'ksat'])
print(df_cleaned_filtered)
df_cleaned_filtered['log_ksat_cm_hr'] = np.log1p(df_cleaned_filtered['ksat_cm_hr'])


# One-hot encode final categorical features
df_cleaned_filtered = pd.get_dummies(df_cleaned_filtered, columns=['field/lab', 'method', 'texturalclass'], drop_first=True)

print(df_cleaned_filtered)
# Save preprocessed data to Excel
df_cleaned_filtered.to_excel("../data/preprocessed_data.xlsx", index=False)

# Final dataset is now saved and ready for modeling.