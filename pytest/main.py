
def preprocess(data,columns_to_drop):
    columns_to_drop = ['Unnamed: 0', 'agecat', 'incomecat','smoking2cat','alcohol.cat',
                               'marital_status', 'number_of_children',
       'smoking_status', 'physical_activity_level',
        'alcohol_consumption','history_of_substance_abuse',
       'family_history_of_depression', 'chronic_medical_conditions','age']
    data = data.drop(columns=columns_to_drop)
    return data


def predict(df):
    """
    Predicts probabilities for the given DataFrame using the loaded DictVectorizer and model.

    :param df: pandas DataFrame containing input data
    :return: Array of predicted probabilities
    """
       # Load the model and DictVectorizer
    with open(model_path, 'rb') as f_in:  # Use the correct path to the model
       model = joblib.load(f_in)


    # Scale numerical columns
    numerical = ['income']
    sc = StandardScaler()
    df[numerical] = sc.fit_transform(df[numerical])  # Use the fitted scaler

    # Define categorical columns
    categorical = [
        'education_level', 'employment_status',
        'dietary_habits', 'sleep_patterns'
    ]

    # Transform categorical and numerical columns
    cat = df[categorical + numerical].to_dict(orient='records')
    dv = DictVectorizer(sparse=False)
    dv.fit(cat)
    X = dv.transform(cat)  # Use the fitted DictVectorizer


    # Predict probabilities
    y_pred = model.predict_proba(X)[:, 1]
    return y_pred
   