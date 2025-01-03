# model_training.py
import sys
import os
from pathlib import Path

current_dir = Path(__file__).resolve().parent
root_dir = current_dir.parent.parent
sys.path.append(str(root_dir))

from src.data.load_data import load_data
from src.features.data_preprocessing import preprocess
from sklearn.ensemble import GradientBoostingClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_score, recall_score
from src.features.feature_engineering import feature_engineering
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction import DictVectorizer

def train(X_train, y_train):
    """
    Trains a Gradient Boosting model, preprocesses validation data, makes predictions,
    computes the AUC score, and returns the trained model and AUC score.

    """
    # Train Gradient Boosting model
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    gb_model = GradientBoostingClassifier(random_state=42)
    gb_model.fit(X_resampled, y_resampled)

    print("Gradient Boost Model instantiated")
    return gb_model

if __name__ == "__main__":
    data = load_data()
    columns_to_drop = ['Unnamed: 0', 'agecat', 'incomecat','smoking2cat',
                               'marital_status', 'number_of_children',
       'smoking_status','alcohol.cat', 'physical_activity_level',
        'alcohol_consumption','history_of_substance_abuse',
       'family_history_of_depression', 'chronic_medical_conditions']
    data2 = preprocess(data,columns_to_drop)
    df_train_full, df_test, df_train, df_val, y_train, y_val, X_train, X_val = feature_engineering(data2)
    model = train(X_train, y_train)

    print("GradientBoosting Classifier instantiated")
    print(model)

