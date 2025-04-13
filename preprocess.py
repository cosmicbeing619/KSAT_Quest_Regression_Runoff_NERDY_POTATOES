import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer

# Load Excel file
xls = pd.ExcelFile(r"C:\Users\jahna\Downloads\data.xlsx")

# Filter sheets that start with "Ref #"
ref_sheets = [sheet for sheet in xls.sheet_names if sheet.startswith("Ref #")]

# Standardize column names
standardized_dfs = []
def standardize_colname(col):
    return col.lower().replace(" ", "")

for sheet in ref_sheets:
    df = xls.parse(sheet)
    df.columns = [standardize_colname(col) for col in df.columns]
    standardized_dfs.append(df)

# Combine all sheets and keep only common columns
combined_df = pd.concat(standardized_dfs, ignore_index=True)
column_sets = [set(df.columns) for df in standardized_dfs]
common_columns = sorted(list(set.intersection(*column_sets)))
df_common = combined_df[common_columns].copy()

# Drop columns with more than 80% missing values
null_threshold = 0.8
columns_to_keep = df_common.columns[df_common.isnull().mean() <= null_threshold]
df_common = df_common[columns_to_keep]

# Convert known numeric columns (with object dtype) to float
categorical_cols = ['field/lab', 'method', 'texturalclass', 'units', 'sitelabel']
df_converted = df_common.copy()

for col in df_converted.select_dtypes(include='object').columns:
    if col not in categorical_cols:
        try:
            df_converted[col] = pd.to_numeric(df_converted[col], errors='coerce')
        except:
            print(f"⚠️ Could not convert column: {col}")

# Impute missing values
numerical_cols = df_converted.select_dtypes(include=[np.number]).columns
categorical_cols = df_converted.select_dtypes(exclude=[np.number]).columns

# Median imputation for numeric columns
num_imputer = SimpleImputer(strategy="median")
df_converted[numerical_cols] = num_imputer.fit_transform(df_converted[numerical_cols])

# Mode imputation for categorical columns
cat_imputer = SimpleImputer(strategy="most_frequent")
df_converted[categorical_cols] = cat_imputer.fit_transform(df_converted[categorical_cols])

# Correlation heatmap for numeric features
numeric_df = df_converted.select_dtypes(include=[np.number])
corr_matrix = numeric_df.corr()
plt.figure(figsize=(14, 10))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", square=True)
plt.title("Correlation Matrix of Numeric Features")
plt.show()

# Drop manually identified unnecessary or redundant features
reduced_df = df_converted.drop(columns=['verycoarse', 'sourcereference', 'fine', 'coarse'])

# Filter columns with <= 80% missing values again (post-drop)
df_cleaned = reduced_df.copy()
null_percent = df_cleaned.isnull().mean()
columns_to_keep = null_percent[null_percent <= 0.8].index.tolist()
df_cleaned_filtered = df_cleaned[columns_to_keep].copy()

# Re-impute numeric columns (just in case new NaNs were introduced)
columns_to_convert = df_cleaned_filtered.select_dtypes(include=[np.number]).columns.tolist()
imputer = SimpleImputer(strategy='median')
df_cleaned_filtered.loc[:, columns_to_convert] = imputer.fit_transform(df_cleaned_filtered[columns_to_convert])

# Impute remaining categorical columns with mode
for col in ['field/lab', 'texturalclass', 'units']:
    df_cleaned_filtered[col] = df_cleaned_filtered[col].fillna(df_cleaned_filtered[col].mode()[0])

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
    ksat = row['ksat']
    factor = unit_conversion.get(unit)
    return ksat * factor if factor is not None else None

# Apply conversion and clean final dataset
df_cleaned_filtered['ksat_cm_hr'] = df_cleaned_filtered.apply(convert_ksat, axis=1)
df_cleaned_filtered = df_cleaned_filtered[df_cleaned_filtered['ksat_cm_hr'].notnull()]
df_cleaned_filtered = df_cleaned_filtered.drop(columns=['units', 'ksat'])

# One-hot encode final categorical features
df_cleaned_filtered = pd.get_dummies(df_cleaned_filtered, columns=['field/lab', 'method', 'texturalclass'], drop_first=True)

# Save preprocessed data to Excel
df_cleaned_filtered.to_excel("preprocessed_data.xlsx", index=False)

# Final dataset is now saved and ready for modeling.