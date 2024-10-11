import pandas as pd
import xgboost as xgb
import logging
from argparse import ArgumentParser
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder


def preprocess_data(df):
    # Drop irrelevant columns
    irrelevant_columns = ['trip_id', 'part', 'station_name']
    df.drop(columns=irrelevant_columns, inplace=True)

    # Handle missing values for 'arrival_time' and 'door_closing_time' by forward fill
    df['arrival_time'].fillna(method='ffill', inplace=True)
    df['door_closing_time'].fillna(method='ffill', inplace=True)

    # Convert time columns to datetime
    df['arrival_time'] = pd.to_datetime(df['arrival_time'], format='%H:%M:%S')
    df['door_closing_time'] = pd.to_datetime(df['door_closing_time'], format='%H:%M:%S')

    # Handle missing values for non-numeric columns
    non_numeric_cols = df.select_dtypes(exclude=['float64', 'int64', 'datetime64']).columns
    imputer_non_num = SimpleImputer(strategy='most_frequent')
    df[non_numeric_cols] = imputer_non_num.fit_transform(df[non_numeric_cols])

    # Handle missing values for numeric columns
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    imputer_num = SimpleImputer(strategy='mean')
    df[numeric_cols] = imputer_num.fit_transform(df[numeric_cols])

    # Encoding categorical columns
    for col in non_numeric_cols:
        df[col] = LabelEncoder().fit_transform(df[col])

    # Scaling numeric features (if needed)
    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    return df


def aggregate_trip_features(df, is_train=True):
    # Aggregate features for each trip
    agg_funcs = {
        'line_id': 'first',
        'direction': 'first',
        'alternative': 'first',
        'cluster': 'first',
        'station_index': 'max',
        'latitude': 'mean',
        'longitude': 'mean',
        'passengers_up': 'sum',
        'passengers_continue': 'sum',
        'mekadem_nipuach_luz': 'mean',
        'passengers_continue_menupach': 'mean',
    }

    if is_train:
        # Calculate trip duration in minutes
        df['trip_duration_in_minutes'] = (df.groupby('trip_id_unique')['door_closing_time']
                                          .transform('max') - df.groupby('trip_id_unique')['arrival_time']
                                          .transform('min')).dt.total_seconds() / 60
        agg_funcs['trip_duration_in_minutes'] = 'first'

    aggregated_df = df.groupby('trip_id_unique').agg(agg_funcs).reset_index()

    return aggregated_df


def main():
    parser = ArgumentParser()
    parser.add_argument('--training_set', type=str, required=True,
                        help="path to the training set")
    parser.add_argument('--test_set', type=str, required=True,
                        help="path to the test set")
    parser.add_argument('--out', type=str, required=True,
                        help="path of the output file as required in the task description")
    args = parser.parse_args()

    # 1. Load the training set
    train_df = pd.read_csv(args.training_set, encoding="ISO-8859-8")

    # 2. Preprocess the training set
    logging.info("preprocessing train...")
    train_df = preprocess_data(train_df)

    # Aggregate features for training set
    train_df = aggregate_trip_features(train_df, is_train=True)

    # Split features and target
    X_train = train_df.drop(columns=['trip_duration_in_minutes'])
    y_train = train_df['trip_duration_in_minutes']

    # 3. Train a model
    logging.info("training...")
    model = xgb.XGBRegressor()
    model.fit(X_train, y_train)

    # 4. Load the test set
    test_df = pd.read_csv(args.test_set, encoding="ISO-8859-8")

    # 5. Preprocess the test set
    logging.info("preprocessing test...")
    test_df = preprocess_data(test_df)

    # Aggregate features for test set
    test_df = aggregate_trip_features(test_df, is_train=False)

    # Ensure that X_train and X_test have the same columns
    X_test = test_df[X_train.columns]

    # 6. Predict the test set using the trained model
    logging.info("predicting...")
    predictions = model.predict(X_test)

    # Ensure no negative predictions
    predictions = [max(0, pred) for pred in predictions]

    # Prepare the output dataframe
    output_df = pd.DataFrame({
        'trip_id_unique': test_df['trip_id_unique'],
        'trip_duration_in_minutes': predictions
    })

    # 7. Save the predictions to args.out
    logging.info("predictions saved to {}".format(args.out))
    output_df.to_csv(args.out, index=False, encoding="ISO-8859-8")



if __name__ == '__main__':
    main()
