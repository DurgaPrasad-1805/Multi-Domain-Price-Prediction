import pandas as pd


def create_features(df):

    # Create lag features for GLD
    df['GLD_lag_1'] = df['GLD'].shift(1)
    df['GLD_lag_7'] = df['GLD'].shift(7)
    df['GLD_lag_30'] = df['GLD'].shift(30)

    # Rolling averages
    df['GLD_roll_7'] = df['GLD'].rolling(window=7).mean()
    df['GLD_roll_30'] = df['GLD'].rolling(window=30).mean()

    # Drop NaN rows created by shift
    df.dropna(inplace=True)

    return df


def train_test_split_time_series(df):

    X = df.drop("GLD", axis=1)
    y = df["GLD"]

    split = int(len(df) * 0.8)

    X_train = X.iloc[:split]
    X_test = X.iloc[split:]
    y_train = y.iloc[:split]
    y_test = y.iloc[split:]

    return X_train, X_test, y_train, y_test