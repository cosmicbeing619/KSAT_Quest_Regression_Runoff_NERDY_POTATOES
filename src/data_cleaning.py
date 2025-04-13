import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer

# Load Excel file
xls = pd.ExcelFile(r"./data/data.xlsx")

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

df_common = df_common.dropna(subset=["ksat"])

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

# Save preprocessed data to Excel
df_converted.to_excel("./data/cleaned_data.xlsx", index=False)