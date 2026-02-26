import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

from gold_clean import load_and_clean_data

from tensorflow.keras import Input

def create_sequences(data, seq_length=30):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length, -1])  # GLD is last column
    return np.array(X), np.array(y)


def train_lstm():

    # Load data
    df = load_and_clean_data("gold_price_data.csv")

    # Use all features
    data = df.values

    # Scale data
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)

    # Create sequences
    seq_length = 30
    X, y = create_sequences(scaled_data, seq_length)

    # Time-series split
    split = int(len(X) * 0.8)

    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # Build LSTM model
    model = Sequential([
        Input(shape=(seq_length, X.shape[2])),
        LSTM(32),
        Dense(1)
    ])

    model.compile(optimizer='adam', loss='mse')

    model.fit(
        X_train, y_train,
        epochs=10,
        batch_size=32,
        verbose=1
    )

    # Predictions
    predictions = model.predict(X_test)

    # Inverse scaling for GLD only
    temp = np.zeros((len(predictions), scaled_data.shape[1]))
    temp[:, -1] = predictions.flatten()
    predictions_actual = scaler.inverse_transform(temp)[:, -1]

    temp[:, -1] = y_test
    y_test_actual = scaler.inverse_transform(temp)[:, -1]

    # Metrics
    mae = mean_absolute_error(y_test_actual, predictions_actual)
    mse = mean_squared_error(y_test_actual, predictions_actual)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test_actual, predictions_actual)

    print("LSTM Results:")
    print("MAE:", mae)
    print("RMSE:", rmse)
    print("R2 Score:", r2)

    model.save("gold_model_lstm.h5")

    return r2


if __name__ == "__main__":
    train_lstm()