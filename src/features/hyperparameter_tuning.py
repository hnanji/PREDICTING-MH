# hyperparameter_tuning.py
import sys
import os
from pathlib import Path

current_dir = Path(__file__).resolve().parent
root_dir = current_dir.parent.parent
sys.path.append(str(root_dir))

from src.data.load_data import load_data # from data_loading.py file
from src.features.data_preprocessing import preprocess # from data_preprocessing.py file
from src.models.model_training import train
from src.features.model_evaluation import evaluate_model
from feature_engineering import feature_engineering
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import RandomizedSearchCV
from imblearn.over_sampling import SMOTE
from scipy.stats import randint, uniform

def tune_hyperparameters(X_train, y_train):
    """
    Perform hyperparameter tuning for GradientBoostingClassifier using RandomizedSearchCV.
    
    :param X_train: Training feature matrix
    :param y_train: Training target values
    :return: Dictionary of the best hyperparameters
    """
    param_distributions = {
        'n_estimators': randint(50, 100),          # Number of boosting stages to be run
        'learning_rate': uniform(0.05, 0.15),      # Learning rate shrinks the contribution of each tree
        'max_depth': randint(3, 4),                # Maximum depth of the individual estimators
        'subsample': uniform(0.7, 0.3),            # Fraction of samples used for fitting the individual base learners
        'min_samples_split': randint(2, 5),        # Minimum number of samples required to split an internal node
        'min_samples_leaf': randint(1, 3)          # Minimum number of samples required to be at a leaf node
    }

    # Initialize the randomized search
    random_search = RandomizedSearchCV(
        GradientBoostingClassifier(random_state=42),
        param_distributions,
        n_iter=30,         # Number of parameter settings that are sampled
        cv=2,              # 2-fold cross-validation
        scoring='roc_auc', # Use AUC as the scoring metric
        verbose=1,
        n_jobs=-1,         # Use all available processors
        random_state=42    # Ensure reproducibility
    )
    
    # Fit the randomized search
    random_search.fit(X_train, y_train)
    
    return random_search.best_params_

if __name__ == "__main__":
    # Assuming load_data(), preprocess(), and feature_engineering() functions are already defined
    data = load_data()
    columns_to_drop = ['Unnamed: 0', 'agecat', 'incomecat', 'smoking2cat',
                       'marital_status', 'number_of_children', 'smoking_status',
                       'alcohol.cat', 'physical_activity_level', 'alcohol_consumption',
                       'history_of_substance_abuse', 'family_history_of_depression',
                       'chronic_medical_conditions']
    data2 = preprocess(data, columns_to_drop)

    df_train_full, df_test, df_train, df_val, y_train, y_val, X_train, X_val = feature_engineering(data2)

    # Resample training data using SMOTE
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

    # Tune hyperparameters for GradientBoostingClassifier
    best_params = tune_hyperparameters(X_resampled, y_resampled)

    print(f"The best hyperparameters are {best_params}")