import pytest
import pandas as pd
from pathlib import Path
import sys

# Set up paths dynamically
current_dir = Path(__file__).resolve().parent
root_dir = current_dir.parent.parent  # Adjust based on your structure
sys.path.append(str(root_dir))  # Add root_dir to Python path

from main import preprocess  # Import preprocess function from main.py

@pytest.fixture
def sample_raw_data():
    # Use the correct path to sample_data.csv
    data_path = current_dir / "data" / "sample_data.csv"
    if not data_path.exists():
        pytest.fail(f"Sample data file not found: {data_path}")
    return pd.read_csv(data_path)

def test_processed(sample_raw_data):
    columns_to_drop = ['Unnamed: 0', 'agecat', 'incomecat','smoking2cat','alcohol.cat',
                               'marital_status', 'number_of_children',
       'smoking_status', 'physical_activity_level',
        'alcohol_consumption','history_of_substance_abuse',
       'family_history_of_depression', 'chronic_medical_conditions','age']
    processed_data = preprocess(sample_raw_data,columns_to_drop)
    assert processed_data.shape == (194622, 6)  