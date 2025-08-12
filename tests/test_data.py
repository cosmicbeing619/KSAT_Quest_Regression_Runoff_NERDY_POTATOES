from src.data_cleaning import load_and_clean_data

def test_dataset_structure():
    df = load_and_clean_data()
    assert 'ksat_cm_hr' in df.columns, "Target column missing"
    assert len(df) > 50, "Dataset too small"
