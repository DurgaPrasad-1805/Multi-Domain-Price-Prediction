import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

from gold_model.gold_clean import load_and_clean_data
from gold_model.gold_feature_engineering import create_features, train_test_split_time_series


def visualize_predictions():

    df = load_and_clean_data("gold_price_data.csv")
    df = create_features(df)

    X_train, X_test, y_train, y_test = train_test_split_time_series(df)

    model = LinearRegression()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    fig, ax = plt.subplots()
    ax.plot(y_test.values, label="Actual GLD")
    ax.plot(predictions, label="Predicted GLD")
    ax.set_title("Gold Price Prediction - Linear Regression")
    ax.legend()

    return fig