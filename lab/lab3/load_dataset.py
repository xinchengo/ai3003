import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os

from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset

def load_data(csv_path='imdb_sentiment_data.csv'):

    df = pd.read_csv(csv_path)
    print(f"Dataset loaded. Total samples: {len(df)}")
    return df

def load_and_preprocess_data(csv_path='imdb_sentiment_data.csv', max_vocab_size=30000, max_seq_len=200, train_ratio=0.8):

    df = load_data(csv_path)
    
    print("\nDataset Statistics:")
    print(f"Total samples: {len(df)}")
    print(f"Positive samples: {(df['label'] == 1).sum()}")
    print(f"Negative samples: {(df['label'] == 0).sum()}")
    
    text_lengths = df['text'].apply(lambda x: len(x.split()))
    print(f"\nText length statistics:")
    print(f"Mean: {text_lengths.mean():.2f}")
    print(f"Median: {text_lengths.median():.2f}")
    print(f"Max: {text_lengths.max()}")
    print(f"95th percentile: {text_lengths.quantile(0.95):.2f}")
    
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        df['text'].values, df['label'].values, 
        test_size=1-train_ratio, random_state=42
    )
    
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        train_texts, train_labels, 
        test_size=0.1, random_state=42
    )
    
    print(f"\nData split:")
    print(f"Train: {len(train_texts)}, Val: {len(val_texts)}, Test: {len(test_texts)}")
    
    return train_texts, val_texts, test_texts, train_labels, val_labels, test_labels

def build_vocab(texts, max_vocab_size):

    from collections import Counter
    
    word_counts = Counter()
    for text in texts:
        words = text.lower().split()
        word_counts.update(words)
    
    vocab = {'<pad>': 0, '<unk>': 1}
    for word, count in word_counts.most_common(max_vocab_size - 2):
        vocab[word] = len(vocab)
    
    print(f"Vocabulary size: {len(vocab)}")
    return vocab

def text_to_indices(text, vocab, max_seq_len):

    words = text.lower().split()
    indices = [vocab.get(w, vocab['<unk>']) for w in words]
    
    if len(indices) >= max_seq_len:
        indices = indices[:max_seq_len]
    else:
        indices = [vocab['<pad>']] * (max_seq_len - len(indices)) + indices
    
    return indices

def create_dataloaders(train_texts, val_texts, test_texts, train_labels, val_labels, test_labels, vocab, max_seq_len, batch_size=32):

    train_data = TensorDataset(
        torch.tensor([text_to_indices(t, vocab, max_seq_len) for t in train_texts], dtype=torch.long),
        torch.tensor(train_labels, dtype=torch.float)
    )
    val_data = TensorDataset(
        torch.tensor([text_to_indices(t, vocab, max_seq_len) for t in val_texts], dtype=torch.long),
        torch.tensor(val_labels, dtype=torch.float)
    )
    test_data = TensorDataset(
        torch.tensor([text_to_indices(t, vocab, max_seq_len) for t in test_texts], dtype=torch.long),
        torch.tensor(test_labels, dtype=torch.float)
    )
    
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader

if __name__ == "__main__":
    load_and_preprocess_data()
