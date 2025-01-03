# model_evaluation.py
import sys
import os
from pathlib import Path

current_dir = Path(__file__).resolve().parent
root_dir = current_dir.parent.parent
sys.path.append(str(root_dir))

from src.data.load_data import load_data
from src.features.data_preprocessing import preprocess

from sklearn.feature_extraction import DictVectorizer 
from sklearn.model_selection import KFold 
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from src.models.model_training import train
from sklearn.metrics import precision_score, recall_score
from src.features.feature_engineering import feature_engineering

def evaluate_model(model, df_val, y_val, dv):
    """
    Evaluates a trained model using validation data, computes predictions, and computes the AUC score.

    :param model: Trained logistic regression model
    :param df_val: Validation DataFrame (features before transformation)
    :param y_val: Target values for validation
    :param dv: DictVectorizer instance used during training
    :return: AUC score
    """

    # Columns to scale
    numerical = ['income','age']
    sc = StandardScaler()

    # Scale numerical columns in validation set
    cols_to_scale = df_val[numerical].columns.tolist()
    df_val.loc[:, cols_to_scale] = sc.fit_transform(df_val.loc[:, cols_to_scale])

    # Define categorical columns
    categorical = [
        'education_level', 'employment_status',
        'dietary_habits', 'sleep_patterns'
    ]

    # Encode categorical and numerical columns using the same DictVectorizer instance
    val_dict = df_val[categorical + numerical].to_dict(orient='records')
    X_val = dv.transform(val_dict)

    # Predictions
    y_pred = model.predict(X_val)

    # Compute Recal
    Recall = recall_score(y_val, y_pred)
    print(f"Recall: {Recall:.3f}")

    return  Recall 

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

# Use the DictVectorizer instance from training
    dv = DictVectorizer(sparse=False)
    train_dict = df_train.to_dict(orient='records')
    dv.fit(train_dict)

    Recall = evaluate_model(model, df_val, y_val,dv)
    print(f"The recall score is {Recall:.2f}")