from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import joblib

from gold_clean import load_and_clean_data
from gold_feature_engineering import create_features, train_test_split_time_series


def train_linear():

    # Load & clean
    df = load_and_clean_data("gold_price_data.csv")

    # Feature engineering
    df = create_features(df)

    # Time-series split
    X_train, X_test, y_train, y_test = train_test_split_time_series(df)

    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predictions
    predictions = model.predict(X_test)

    # Evaluation
    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, predictions)

    print("Linear Regression Results:")
    print("MAE:", mae)
    print("RMSE:", rmse)
    print("R2 Score:", r2)

    # Save model
    joblib.dump(model, "gold_model_linear.pkl")

    return r2


if __name__ == "__main__":
    train_linear()