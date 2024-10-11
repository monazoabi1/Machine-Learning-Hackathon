import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from xgboost import XGBRegressor, plot_importance
from argparse import ArgumentParser
import logging


def preprocess_data(data, is_test_set=False):
    # Select relevant features
    features = ['line_id', 'direction', 'station_index', 'latitude', 'longitude']

    # Handle missing values
    if not is_test_set:
        data = data.dropna(subset=['passengers_up'])  # Drop rows where target is missing
        y = data['passengers_up']
    else:
        y = None

    data = data.fillna(0)  # Fill other missing values with 0
    critical_columns = ['trip_id_unique', 'station_id']
    data = data.dropna(subset=critical_columns)

    # Ensure the features are of numeric type
    X = data[features].astype(float)

    return X, y


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--training_set', type=str, required=True,
                        help="path to the training set")
    parser.add_argument('--test_set', type=str, required=True,
                        help="path to the test set")
    parser.add_argument('--out', type=str, required=True,
                        help="path of the output file as required in the task description")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    # 1. Load the training set
    training_data = pd.read_csv(args.training_set, encoding='ISO-8859-8')
    logging.info("Training data loaded")

    # 2. Preprocess the training set
    logging.info("Preprocessing train...")
    X_train, y_train = preprocess_data(training_data)

    # 3. Train a model
    logging.info("Training...")
    poly = PolynomialFeatures(degree=2)
    X_train_poly = poly.fit_transform(X_train)

    model = XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1, max_depth=5, random_state=0)
    model.fit(X_train_poly, y_train)

    # 4. Load the test set
    test_data = pd.read_csv(args.test_set, encoding='ISO-8859-8')
    logging.info("Test data loaded")

    # 5. Preprocess the test set
    logging.info("Preprocessing test...")
    X_test, _ = preprocess_data(test_data, is_test_set=True)
    X_test_poly = poly.transform(X_test)

    # 6. Predict the test set using the trained model
    logging.info("Predicting...")
    test_predictions = model.predict(X_test_poly)

    # 7. Save the predictions to args.out
    output = pd.DataFrame({
        'trip_id_unique_station': test_data['trip_id_unique_station'],
        'passengers_up': np.maximum(np.round(test_predictions).astype(int), 1)
        # Ensure predictions are positive integers
    })
    output.to_csv(args.out, index=False)
    logging.info("Predictions saved to {}".format(args.out))


