from sklearn.preprocessing import LabelEncoder

def transform_categorical_columns(df, categorical_columns):
    label_encoders = {}
    for column in categorical_columns:
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column])
        label_encoders[column] = le
    return df, label_encoders

def inverse_transform_categorical_columns(df, label_encoders):
    for column, le in label_encoders.items():
        df[column] = df[column].astype(int) # ensure the column has integer dtype for indexing
        df[column] = le.inverse_transform(df[column])
    return df
