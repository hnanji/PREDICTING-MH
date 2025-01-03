import sys
import os
import joblib
from pathlib import Path
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction import DictVectorizer

# Define directory paths
current_dir = Path(__file__).resolve().parent
root_dir = current_dir.parent.parent  
model_path = root_dir / 'models' / 'gradient_boosting_model.pkl'
valid_path = root_dir / 'data' / 'processed' / 'test.csv'

def predict(df):
    """
    Predicts probabilities for the given DataFrame using the loaded DictVectorizer and model.

    :param df: pandas DataFrame containing input data
    :return: Array of predicted probabilities
    """
       # Load the model and DictVectorizer
    with open(model_path, 'rb') as f_in:  
       dv,model = joblib.load(f_in)


    # Scale numerical columns
    numerical = ['income','age']
    sc = StandardScaler()
    df[numerical] = sc.fit_transform(df[numerical]) 

    # Define categorical columns
    categorical = [
        'education_level', 'employment_status',
        'dietary_habits', 'sleep_patterns'
    ]

    # Transform categorical and numerical columns
    cat = df[categorical + numerical].to_dict(orient='records')
    dv = DictVectorizer(sparse=False)
    dv.fit(cat)
    X = dv.transform(cat)  


    # Prediction
    y_pred = model.predict(X)
    return y_pred

if __name__ == "__main__":
    # Load validation data
    y_val = pd.read_csv(valid_path)

    # Call the predict function
    predictions = predict(y_val)
    print(predictions)