import sys
from pathlib import Path
import joblib
from sklearn.ensemble import GradientBoostingClassifier
from imblearn.over_sampling import SMOTE
from sklearn.feature_extraction import DictVectorizer

# Resolve current and root directories
current_dir = Path(__file__).resolve().parent
root_dir = current_dir.parent.parent
sys.path.append(str(root_dir / "src"))  

# Import modules after updating the path
from data.load_data import load_data
from features.data_preprocessing import preprocess
from features.feature_engineering import feature_engineering

def serialize_model(model, dv, filename):
    """
    Save the model and DictVectorizer to the specified file path.
    :param model: Trained model
    :param dv: DictVectorizer
    :param filename: Path to save the model and DictVectorizer
    """
    joblib.dump((dv, model), filename)
    print(f"Model and DictVectorizer serialized and saved as '{filename}'")

if __name__ == "__main__":
    # Load and preprocess data
    data = load_data()
    columns_to_drop = [
        'Unnamed: 0', 'agecat', 'incomecat', 'smoking2cat',
        'marital_status', 'number_of_children', 'smoking_status',
        'alcohol.cat', 'physical_activity_level', 'alcohol_consumption',
        'history_of_substance_abuse', 'family_history_of_depression',
        'chronic_medical_conditions'
    ]
    preprocessed_data = preprocess(data, columns_to_drop)

    # Feature engineering
    df_train_full, df_test, df_train, df_val, y_train, y_val, X_train, X_val = feature_engineering(preprocessed_data)

    # Define the hyperparameters for Gradient Boosting
    best_params = {
        'learning_rate': 0.19606332782621888,
        'max_depth': 3,
        'min_samples_leaf': 1,
        'min_samples_split': 3,
        'n_estimators': 96,
        'subsample': 0.8855158027999261
    }

    # Resample training data using SMOTE
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

    # Initialize DictVectorizer
    numerical = ['income', 'age']
    categorical = ['education_level', 'employment_status', 'dietary_habits', 'sleep_patterns']
    dv = DictVectorizer(sparse=False)
    dv.fit(df_train[categorical + numerical].to_dict(orient="records"))

    # Transform training data using DictVectorizer
    X_train_transformed = dv.transform(df_train[categorical + numerical].to_dict(orient="records"))

    # Train the Gradient Boosting model with best parameters
    gb_model = GradientBoostingClassifier(
        learning_rate=best_params['learning_rate'],
        max_depth=best_params['max_depth'],
        min_samples_leaf=best_params['min_samples_leaf'],
        min_samples_split=best_params['min_samples_split'],
        n_estimators=best_params['n_estimators'],
        subsample=best_params['subsample'],
        random_state=42
    )
    gb_model.fit(X_resampled, y_resampled)

    # Define the save path for the model
    model_dir = root_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_save_path = model_dir / "gradient_boosting_model.pkl"

    # Serialize the trained model and DictVectorizer
    serialize_model(gb_model, dv, str(model_save_path))

    print(f"Model training and serialization complete. Saved at: {model_save_path}")