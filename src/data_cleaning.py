import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer

# Load Excel file
xls = pd.ExcelFile(r"../data/data.xlsx")

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

# Combine all sheets
combined_df = pd.concat(standardized_dfs, ignore_index=True)

# Get common columns across all sheets
column_sets = [set(df.columns) for df in standardized_dfs]
common_columns = sorted(list(set.intersection(*column_sets)))

# Define important columns to preserve even if not in all sheets
important_cols = ['ksat', 'units', 'method', 'field/lab', 'texturalclass', 'sitelabel']
standardized_important_cols = [standardize_colname(c) for c in important_cols]

# Combine common columns + important columns that are available
available_important = list(set(combined_df.columns) & set(standardized_important_cols))
safe_columns = sorted(set(common_columns).union(available_important))

# Create the final common DataFrame with safe columns
df_common = combined_df[safe_columns].copy()

# Drop columns with more than 80% missing values
null_threshold = 0.8
columns_to_keep = df_common.columns[df_common.isnull().mean() <= null_threshold]
df_common = df_common[columns_to_keep]

# Drop rows where target 'ksat' is missing
df_common = df_common.dropna(subset=["ksat"])

# Define known categorical columns
categorical_cols = ['field/lab', 'method', 'texturalclass', 'units', 'sitelabel']
categorical_cols = [col.lower().replace(" ", "") for col in categorical_cols]

# Make a copy for conversion
df_converted = df_common.copy()

# Convert only numeric-looking object columns (exclude categorical)
for col in df_converted.select_dtypes(include='object').columns:
    if col not in categorical_cols:
        try:
            df_converted[col] = pd.to_numeric(df_converted[col], errors='coerce')
        except:
            print(f"⚠️ Could not convert column: {col}")

# Optional: Show summary
print("✅ Final columns:", df_converted.columns.tolist())
print("🔍 Sample values in 'units':", df_converted['units'].dropna().unique() if 'units' in df_converted else "Missing")

# Save preprocessed data to Excel
df_converted.to_excel("../data/cleaned_data.xlsx", index=False)
print("📁 Saved cleaned data to ../data/cleaned_data.xlsx")
