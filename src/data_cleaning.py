# src/data_cleaning.py
from pathlib import Path
import re
import numpy as np
import pandas as pd

# ---------- Robust paths (works from any CWD) ----------
ROOT = Path(__file__).resolve().parents[1]   # repo root
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

RAW_PATH = DATA_DIR / "data.xlsx"
CLEAN_PATH = DATA_DIR / "cleaned_data.xlsx"
PREPROC_PATH = DATA_DIR / "preprocessed_data.xlsx"

# ---------- Helpers ----------
def std_name(s: str) -> str:
    """lowercase and replace any non [a-z0-9] with underscore; collapse repeats"""
    s = re.sub(r'[^a-z0-9]+', '_', str(s).lower()).strip('_')
    s = re.sub(r'_{2,}', '_', s)
    return s

def norm_unit(u: str) -> str:
    return std_name(u)

def to_cm_per_hr(value: float, unit_raw: str) -> float:
    """
    Convert Ksat value to cm/hr.
    Supports units seen in your sample:
      cm/hr, mm/hr (or mm h^-1), in/hr, cm/min, cm/day,
      m/s (or m s^-1), m/day (m day^-1), cm/s, um s^-1,
      log m s^-1, log10(cm/hr), cm hr^-1, mm/h, etc.
    Returns np.nan if unknown.
    """
    if pd.isna(value) or pd.isna(unit_raw):
        return np.nan

    u = norm_unit(unit_raw)
    v = float(value)

    # direct (not log) units
    if u in {"cm_hr", "cm_per_hr", "cm_hr_1"}:
        return v
    if u in {"mm_hr", "mm_per_hr", "mm_h_1"}:
        return v * 0.1                     # 1 mm = 0.1 cm
    if u in {"in_hr", "inch_hr", "in_per_hr"}:
        return v * 2.54                    # 1 in = 2.54 cm
    if u in {"cm_min", "cm_per_min"}:
        return v * 60.0                    # cm/min → cm/hr
    if u in {"cm_day", "cm_per_day"}:
        return v / 24.0                    # cm/day → cm/hr
    if u in {"cm_s", "cm_per_s"}:
        return v * 3600.0                  # cm/s → cm/hr
    if u in {"m_s", "m_per_s", "m_s_1"}:
        return v * 100.0 * 3600.0          # m/s → cm/hr
    if u in {"m_day", "m_per_day", "m_day_1"}:
        return v * 100.0 / 24.0            # m/day → cm/hr
    if u in {"um_s_1", "um_per_s"}:        # micrometer per second
        return v * 1e-4 * 3600.0           # 1 μm = 1e-4 cm; *3600 to hr

    # log10[...] units (assume base-10 logs)
    if u.startswith("log") or "log10" in u:
        # try to infer the underlying unit
        # Examples seen: 'log m s^-1', 'log10 (cm3/h)' (treat as cm/hr), etc.
        if "m_s" in u or "m_per_s" in u or "m_s_1" in u:
            m_per_s = 10.0 ** v
            return m_per_s * 100.0 * 3600.0
        if "cm_hr" in u or "cm_per_hr" in u or "cm3_h" in u or "cm_h" in u:
            # Treat as log10(cm/hr)
            return 10.0 ** v
        # fallback: unknown log unit
        return np.nan

    # more textual variants
    if u in {"m_s", "m_s_1"}:
        return v * 360000.0

    # unknown
    return np.nan

# ---------- Load Excel ----------
if not RAW_PATH.exists():
    raise FileNotFoundError(f"Raw data not found at: {RAW_PATH}")

xls = pd.ExcelFile(RAW_PATH)

# Prefer sheets starting with "Ref #", else use all
ref_sheets = [s for s in xls.sheet_names if s.lower().startswith("ref #")]
sheets_to_use = ref_sheets if ref_sheets else xls.sheet_names

frames = []
for sheet in sheets_to_use:
    df = xls.parse(sheet)
    df.columns = [std_name(c) for c in df.columns]
    frames.append(df)

combined = pd.concat(frames, ignore_index=True)

# Compute common columns across sheets + keep some important ones when present
column_sets = [set(f.columns) for f in frames]
common_cols = set.intersection(*column_sets) if column_sets else set(combined.columns)

important_raw = ['ksat', 'units', 'method', 'field/lab', 'texturalclass', 'sitelabel']
important_std = [std_name(c) for c in important_raw]  # field_lab, texturalclass, etc.

available_important = set(combined.columns).intersection(set(important_std))
safe_cols = sorted(set(common_cols).union(available_important))
df = combined[safe_cols].copy()

# Drop columns with >80% missing
null_thresh = 0.8
keep_cols = df.columns[df.isnull().mean() <= null_thresh]
df = df[keep_cols]

# Need ksat + units for conversion
if "ksat" not in df.columns or "units" not in df.columns:
    raise ValueError("Required columns 'ksat' and/or 'units' not found after standardization.")

# Convert non-categorical object columns to numeric (best effort)
categorical_cols = [c for c in ["field_lab", "method", "texturalclass", "units", "sitelabel"] if c in df.columns]
for col in df.select_dtypes(include="object").columns:
    if col not in categorical_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# Drop rows without ksat
df = df.dropna(subset=["ksat"])

# ---------- Unit normalization to cm/hr ----------
df["ksat_cm_hr"] = df.apply(lambda r: to_cm_per_hr(r["ksat"], r["units"]), axis=1)

# Remove rows we failed to convert
before = len(df)
df = df.dropna(subset=["ksat_cm_hr"])
after = len(df)
print(f"Converted units to cm/hr for {after}/{before} rows.")

# Save cleaned & preprocessed tables
df.to_excel(CLEAN_PATH, index=False)
print(f"📁 Saved cleaned data to {CLEAN_PATH}")

# You can create a modeling-ready table here if you want (select features, encode cats, etc.)
# For now, just mirror cleaned to preprocessed:
df.to_excel(PREPROC_PATH, index=False)
print(f"📁 Saved preprocessed data to {PREPROC_PATH}")
