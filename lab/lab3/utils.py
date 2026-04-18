import pandas as pd
import torch
import os

from sklearn.model_selection import train_test_split
from typing import Tuple, List, Dict, Any

def load_csv_data(csv_path : str = './data/imdb_sentiment_data.csv'):

    df = pd.read_csv(csv_path)
    print(f"Dataset loaded. Total samples: {len(df)}")
    return df

def split_csv_data(csv_path : str = './data/imdb_sentiment_data.csv', 
                   train_path : str = './data/train.csv',
                   val_path : str = './data/val.csv',
                   test_path : str = './data/test.csv',
                   train_ratio : float = 0.72,
                   val_ratio : float = 0.08):
    df = load_csv_data(csv_path)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        df['review'].values, df['sentiment'].values, 
        test_size=1-train_ratio-val_ratio, random_state=42
    )
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        train_texts, train_labels, 
        test_size=val_ratio/(val_ratio+train_ratio), random_state=42
    )
    train_df = pd.DataFrame({"review": train_texts, "sentiment": train_labels})
    val_df = pd.DataFrame({"review": val_texts, "sentiment": val_labels})
    test_df = pd.DataFrame({"review": test_texts, "sentiment": test_labels})
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)
    print(f"Data split and saved to {train_path}, {val_path}, {test_path}")
    print(f"Train samples: {len(train_df)}, Val samples: {len(val_df)}, Test samples: {len(test_df)}")
    