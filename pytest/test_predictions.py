import pytest
import pandas as pd
from pathlib import Path
import sys
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction import DictVectorizer

# Set up paths dynamically
current_dir = Path(__file__).resolve().parent.parent  # Adjusted to root directory
root_dir = current_dir  # Root directory of the project
sys.path.append(str(root_dir))  # Add root_dir to Python path

# Import the necessary function
from main import predict

# Define numerical and categorical columns
numerical = ['income']
categorical = ['education_level', 'employment_status', 'dietary_habits', 'sleep_patterns']

@pytest.fixture
def train_test_data():
    # Load training and validation data
    train_path = current_dir / "data" / "processed" / "train.csv"
    val_path = current_dir / "data" / "processed" / "val.csv"
    y_train_path = current_dir / "data" / "processed" / "y_train.csv"
    y_val_path = current_dir / "data" / "processed" / "y_val.csv"

    # Check if files exist
    for path in [train_path, val_path, y_train_path, y_val_path]:
        if not path.exists():
            pytest.fail(f"Data file not found: {path}")

    # Load data
    train = pd.read_csv(train_path)
    val = pd.read_csv(val_path)
    y_train = pd.read_csv(y_train_path).values.ravel()  # Flatten target array
    y_val = pd.read_csv(y_val_path).values.ravel()

    return train, val, y_train, y_val

def test_prediction_shape(train_test_data):
    # Unpack data
    train, val, y_train, y_val = train_test_data

    # Preprocess training data
    train_dict = train[categorical + numerical].to_dict(orient='records')
    dv = DictVectorizer(sparse=False)
    X_train = dv.fit_transform(train_dict)

    # Preprocess validation data
    val_dict = val[categorical + numerical].to_dict(orient='records')
    X_val = dv.transform(val_dict)

    # Train model
    model = LogisticRegression(solver='liblinear', random_state=1)
    model.fit(X_train, y_train)

    # Predictions
    pred_train = model.predict(X_train)
    pred_val = model.predict(X_val)

    # Assertions
    assert pred_train.shape[0] == len(y_train), "Mismatch in training prediction shape"
    assert pred_val.shape[0] == len(y_val), "Mismatch in validation prediction shape"