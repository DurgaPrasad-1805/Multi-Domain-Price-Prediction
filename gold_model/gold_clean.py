import os
import pandas as pd

def load_and_clean_data(file_name):

    base_path = os.path.dirname(__file__)
    file_path = os.path.join(base_path, file_name)

    df = pd.read_csv(file_path)

    # Convert Date column
    df['Date'] = pd.to_datetime(df['Date'])

    # Sort by date
    df = df.sort_values('Date')

    # Set Date as index
    df.set_index('Date', inplace=True)

    return df