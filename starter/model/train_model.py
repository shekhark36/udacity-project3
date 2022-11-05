# Script to train machine learning model.
import os
import pandas as pd
from sklearn.model_selection import train_test_split

from data import process_data
from model import train_model, compute_model_metrics, inference, save_files

## Slice testing
def slice_census_data(df, model, encoder, scaler, lb, cat_features, 
                                    cat_col='education', label='salary'):
    if os.path.isfile("slice_output.txt"):
        os.remove("slice_output.txt")
    
    print(cat_features)
    for value in df[cat_col].unique():
        df_temp = df[df[cat_col] == value]
        # y = df_temp[label]
        # X = df_temp.drop([label], axis=1)

        X_train, y_train, _, _, _ = process_data(df_temp, 
                                                categorical_features=cat_features, 
                                                label=label, training=False,
                                                encoder=encoder, scaler=scaler, lb=lb)
        
        # mean_hours = X_train["hours_per_week"].mean()
        # stddev_hours = X_train["hours_per_week"].std()

        preds = inference(model, X_train)
        precision, recall, fbeta = compute_model_metrics(y_train, preds)
        
        file_object = open('slice_output.txt', 'a')
        file_object.write(f'\n------Holding Categorical feature {cat_col} and value {value} fixed -------')
        # file_object.write(f"\nMean Hours Per Week: {mean_hours}")
        # file_object.write(f"\nStd.Dev in Hours Per Week: {stddev_hours}")
        file_object.write(f"\nPrecision: {precision}")
        file_object.write(f"\nRecall: {recall}")
        file_object.write(f"\nF1Score: {fbeta}")
        file_object.write('\n------------------------------------------')
        file_object.close()


data = pd.read_csv("../data/cleaned_census_data.csv")

# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(data, test_size=0.20)

cat_features = [
    "workclass",
    "education",
    "marital_status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native_country",
]

X_train, y_train, encoder, scaler, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

X_test, y_test, encoder, scaler, lb = process_data(
    test, categorical_features=cat_features, label="salary", training=False,
    encoder=encoder, scaler=scaler, lb=lb
)

lg_model = train_model(X_train, y_train)
preds = inference(lg_model, X_test)

precision, recall, fbeta = compute_model_metrics(y_test, preds)

print(f"The test results are Precision={precision}, recall={recall}, fbeta={fbeta}")

save_files(lg_model, scaler, encoder, lb)

slice_census_data(data, lg_model, encoder, scaler, lb, cat_features, cat_col="education", label="salary")
