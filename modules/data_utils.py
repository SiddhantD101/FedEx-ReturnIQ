import pandas as pd
from pathlib import Path

def load_returns(path=Path("data/returniq_dataset_10000.csv")):
    return pd.read_csv(path)

def load_rates(path=Path("data/fedex_rates.csv")):
    return pd.read_csv(path)
