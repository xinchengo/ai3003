# This file defines the tokenizer configurations
# for the AI3003 lab experiment

from __future__ import annotations
import os
import pickle
import json
import torch

from typing import Dict, Iterable, List, Tuple, Optional, Any
from dataclasses import dataclass
from collections import Counter

import unicodedata
from config_utils import load_config

@dataclass(frozen=True)
class TokenizerConfig:
    name: str
    type: str
    normalization_scheme: str
    vocab_size: Optional[int] = None
    min_occurences: Optional[int] = None
    clip_length: Optional[int] = None
    
    @staticmethod
    def differs_in(config1: TokenizerConfig, config2: TokenizerConfig,
                   keys: List[str]) -> bool:
        for key in keys:
            if getattr(config1, key) != getattr(config2, key):
                return True
        return False
    
    @classmethod
    def from_json(cls, json_file: str, name: str) -> TokenizerConfig:
        config_dict = load_config(json_file)["preprocess"].get(name, None)
        if config_dict is None:
            raise ValueError(f"Tokenizer config {name} not found in {json_file}")
        return cls(name=name, **config_dict)

@dataclass(frozen=True)
class Encoding:
    tokens: List[str]
    ids: List[int]
    
class BaseTokenizer:
    "Abstract base class for tokenizers."
    def encode(self, text: str) -> Encoding:
        raise NotImplementedError("BaseTokenizer is an abstract class, "
                                  "use its subclasses instead.")
        
    def decode(self, ids: Iterable[int]) -> str:
        raise NotImplementedError("BaseTokenizer is an abstract class, "
                                  "use its subclasses instead.")
        
    @staticmethod
    def _normalize(text: str, normalization_scheme: str) -> str:
        """Static helper to apply normalization with a given scheme."""
        if normalization_scheme == "none":
            return text.strip()
        elif normalization_scheme == "lower":
            return text.strip().lower()
        elif normalization_scheme == "sch1":
            "remove accents and lowercase"
            return ''.join(
                c for c in unicodedata.normalize('NFD', text.strip())
                if unicodedata.category(c) != 'Mn'
            ).strip().lower()
        elif normalization_scheme == "sch2":
            "sch1 + replace non-alphanumeric with space"
            normalized_text = ''.join(
                c for c in unicodedata.normalize('NFD', text.strip())
                if unicodedata.category(c) != 'Mn'
            ).strip().lower()
            normalized_text = ''.join(
                c if c.isalnum() else ' ' for c in normalized_text
            )
            return normalized_text
        else:
            raise ValueError(f"Unknown normalization scheme: "
                             f"{normalization_scheme}")
            
class PreTrainedTokenizer(BaseTokenizer):
    def __init__(self, pretrained_model_name: str):
        raise NotImplementedError("PreTrainedTokenizer is not implemented yet.")

class VocabBasedTokenizer(BaseTokenizer):
    """
    Base class for tokenizers.
    """
    
    def __init__(self,
                 vocab: Optional[Dict[str, int]] = None,
                 inv_vocab: Optional[List[str]] = None,
                 unk_token: Optional[str] = None,
                 pad_token: str = "<pad>",
                 normalization_scheme: str = "none",
                 ):
        self.unk_token = unk_token
        self.pad_token = pad_token
        self.normalization_scheme = normalization_scheme
        
        if vocab is None:
            vocab = {}
        self.vocab = vocab
        
        if inv_vocab is None:
            # We assume that the token idxes are contiguous and start from 0
            assert(max(vocab.values()) == len(vocab) - 1) # a weak assertion
            inv_vocab = [None] * (max(vocab.values()) + 1)
            for token, idx in vocab.items():
                inv_vocab[idx] = token
        self.inv_vocab = inv_vocab
        
        self.pad_id = vocab.get(pad_token, None)
        
        if unk_token is not None:
            self.unk_id = vocab.get(unk_token, None)
        else:
            self.unk_id = None

    @staticmethod
    def build_vocab(self, corpus: Iterable[str]) -> Dict[str, int]:
        raise NotImplementedError("VocabBasedTokenizer is an abstract class, "
                                  "use its subclasses instead.")
        
    @classmethod
    def from_corpus(cls):
        raise NotImplementedError("VocabBasedTokenizer is an abstract class, "
                                  "use its subclasses instead.")
        
class WordLevelTokenizer(VocabBasedTokenizer):
    def __init__(
        self,
        vocab: Optional[Dict[str, int]] = None,
        inv_vocab: Optional[List[str]] = None,
        normalization_scheme: str = "none",
    ):
        if normalization_scheme not in "sch2":
            raise Warning("WordLevelTokenizer works best with"
                          "sch2 normalization scheme, "
                          f"but got {normalization_scheme}")
        super().__init__(vocab, inv_vocab, unk_token="<unk>", 
                         pad_token="<pad>", 
                         normalization_scheme=normalization_scheme)
    
    def encode(self, text: str) -> Encoding:
        normalized_text = self._normalize(text, self.normalization_scheme)
        tokens = normalized_text.split()
        ids = [self.vocab.get(token, self.unk_id) for token in tokens]
        return Encoding(tokens, ids)
    
    def decode(self, ids: Iterable[int]) -> str:
        tokens = [self.inv_vocab[id] for id in ids]
        return ' '.join(tokens)
    
    @staticmethod
    def build_vocab(corpus: Iterable[str], vocab_size: int = 30000, 
                    normalization_scheme: str = "none"):
        # Normalize the corpus using the specified scheme
        normalized_corpus = []
        for text in corpus:
            normalized_corpus.append(
                BaseTokenizer._normalize(text, normalization_scheme))
        
        # Count the occurences of each word
        word_counts = Counter()
        for text in normalized_corpus:
            word_counts.update(text.split())
        
        # Build the vocab and inv_vocab
        vocab = {}
        vocab["<pad>"] = 0
        vocab["<unk>"] = 1
        idx = 2
        for word, count in sorted(word_counts.items(), key=lambda x: x[1], 
                                  reverse=True):
            if idx >= vocab_size:
                break
            vocab[word] = idx
            idx += 1
        inv_vocab = [None] * len(vocab)
        for token, idx in vocab.items():
            inv_vocab[idx] = token
        return vocab, inv_vocab, normalization_scheme
    
    @classmethod
    def from_corpus(cls, corpus: Iterable[str], 
                    vocab_size: int = 30000,
                    normalization_scheme: str = "none"):
        vocab, inv_vocab, normalization_scheme = cls.build_vocab(
            corpus, vocab_size, normalization_scheme)
        return cls(vocab=vocab, inv_vocab=inv_vocab, 
                   normalization_scheme=normalization_scheme)

class CharLevelTokenizer(VocabBasedTokenizer):
    def __init__(
        self,
        vocab: Optional[Dict[str, int]] = None,
        inv_vocab: Optional[List[str]] = None,
        normalization_scheme: str = "none",
    ):
        super().__init__(vocab, inv_vocab=inv_vocab, unk_token="<unk>", 
                         pad_token="<pad>", 
                         normalization_scheme=normalization_scheme)
    
    def encode(self, text: str) -> Encoding:
        normalized_text = self._normalize(text, self.normalization_scheme)
        tokens = list(normalized_text)
        ids = [self.vocab.get(token, self.unk_id) for token in tokens]
        return Encoding(tokens, ids)
    
    def decode(self, ids: Iterable[int]) -> str:
        tokens = [self.inv_vocab[id] for id in ids]
        return ''.join(tokens)
    
    @staticmethod
    def build_vocab(
            corpus: Iterable[str],
            min_occurences: int = 100,
            normalization_scheme: str = "none",
        ):
        # Normalize the corpus using the specified scheme
        normalized_corpus = []
        for text in corpus:
            normalized_corpus.append(
                BaseTokenizer._normalize(text, normalization_scheme))
        
        # Count the occurences of each character
        char_counts = Counter()
        for text in normalized_corpus:
            char_counts.update(text)
        
        # Build the vocab and inv_vocab
        vocab = {}
        vocab["<pad>"] = 0
        vocab["<unk>"] = 1
        idx = 2
        for char, count in char_counts.items():
            if count >= min_occurences:
                vocab[char] = idx
                idx += 1
        inv_vocab = [None] * len(vocab)
        for token, idx in vocab.items():
            inv_vocab[idx] = token
        return vocab, inv_vocab, normalization_scheme
    
    @classmethod
    def from_corpus(cls, corpus: Iterable[str], 
                    min_occurences: int = 100,
                    normalization_scheme: str = "none"):
        vocab, inv_vocab, normalization_scheme = cls.build_vocab(
            corpus, min_occurences, normalization_scheme)
        return cls(vocab=vocab, inv_vocab=inv_vocab, 
                   normalization_scheme=normalization_scheme)
        

class BPETokenizer(BaseTokenizer):
    """A naive implementation of BPE tokenizer, not optimized"""
    def __init__(self, 
                 normalization_scheme: str = "none"):
        self.inv_vocab = [bytes([i]) for i in range(256)]
        self.merges = []
        self.normalization_scheme = normalization_scheme
    
    @staticmethod
    def _merge_sequence(seq: List[int], 
                        byte1: int, byte2: int, 
                        next_token_id: int) -> List[int]:
        merged_seq = []
        i = 0
        while i < len(seq):
            if i < len(seq) - 1 and seq[i] == byte1 and seq[i + 1] == byte2:
                merged_seq.append(next_token_id)
                i += 2
            else:
                merged_seq.append(seq[i])
                i += 1
        return merged_seq
    
    def encode(self, text: str) -> Encoding:
        normalized_text = self._normalize(text, self.normalization_scheme)
        byte_seq = list(normalized_text.encode('utf-8'))
        next_token_id = 256
        for byte1, byte2 in self.merges:
            byte_seq = BPETokenizer._merge_sequence(byte_seq, byte1, byte2, 
                                            next_token_id)
            next_token_id += 1
        tokens = [self.inv_vocab[id] for id in byte_seq]
        return Encoding(tokens, byte_seq)
    
    def decode(self, ids: Iterable[int]) -> str:
        byte_seq = [self.inv_vocab[id] for id in ids]
        byte_seq = b''.join(byte_seq)
        return byte_seq.decode('utf-8')
        
    @staticmethod
    def build_vocab(corpus: Iterable[str], 
                    vocab_size: int = 30000,
                    normalization_scheme: str = "none"):
        merges = []
        inv_vocab = [bytes([i]) for i in range(256)]
        next_token_id = 256
        
        normalized_corpus = []
        for text in corpus:
            normalized_corpus.append(
                BaseTokenizer._normalize(text, normalization_scheme))
            
        byte_sequences = [list(text.encode('utf-8')) 
                          for text in normalized_corpus]
        
        # Use tqdm to show progress of vocab building
        from tqdm import tqdm
        for _ in tqdm(range(vocab_size - 256), desc="Building BPE vocab"):
            # Find most frequent adjacent pair
            pair_counts = Counter()
            for seq in byte_sequences:
                for i in range(len(seq) - 1):
                    pair_counts[(seq[i], seq[i + 1])] += 1
            
            if not pair_counts:
                break
            
            (byte1, byte2), count = pair_counts.most_common(1)[0]
            inv_vocab.append(inv_vocab[byte1] + inv_vocab[byte2])
            
            merges.append((byte1, byte2))
            
            byte_sequences = [BPETokenizer._merge_sequence(
                seq, byte1, byte2, next_token_id) for seq in byte_sequences]
            next_token_id += 1
        
        return merges, inv_vocab, normalization_scheme
    
    @classmethod
    def from_corpus(cls, corpus: Iterable[str], 
                    vocab_size: int = 30000,
                    normalization_scheme: str = "none"):
        merges, inv_vocab, normalization_scheme = cls.build_vocab(
            corpus, vocab_size, normalization_scheme)
        instance = cls(normalization_scheme=normalization_scheme)
        instance.merges = merges
        instance.inv_vocab = inv_vocab
        return instance


class BPETokenizerHF(BaseTokenizer):
    """Wrapper of HuggingFace tokenizers library for BPE tokenization
    This part is generated by Github Copilot"""

    def __init__(self, 
                 tokenizer=None,
                 normalization_scheme: str = "none"):
        """
        Args:
            tokenizer: A tokenizers.Tokenizer instance
            normalization_scheme: Normalization scheme to apply
        """
        self.tokenizer = tokenizer
        self.normalization_scheme = normalization_scheme
    
    def encode(self, text: str) -> Encoding:
        """Encode text to token ids."""
        normalized_text = self._normalize(text, self.normalization_scheme)
        encoding = self.tokenizer.encode(normalized_text)
        return Encoding(tokens=encoding.tokens, ids=encoding.ids)
    
    def decode(self, ids: Iterable[int]) -> str:
        """Decode token ids to text."""
        return self.tokenizer.decode(list(ids))
    
    @staticmethod
    def build_vocab(corpus: Iterable[str], 
                    vocab_size: int = 10000,
                    normalization_scheme: str = "none"):
        """Build BPE vocabulary from corpus using HuggingFace tokenizers."""
        from tokenizers import Tokenizer
        from tokenizers.models import BPE
        from tokenizers.normalizers import Sequence, NFD, Lowercase, StripAccents
        from tokenizers.pre_tokenizers import Whitespace
        from tokenizers.trainers import BpeTrainer
        
        # Create tokenizer with BPE model
        tokenizer = Tokenizer(BPE())
        
        # Set up normalizer based on normalization_scheme
        if normalization_scheme == "sch2":
            # NFD + strip accents + lowercase + whitespace handling
            tokenizer.normalizer = Sequence([
                NFD(),
                StripAccents(),
                Lowercase(),
            ])
        elif normalization_scheme == "sch1":
            # NFD + strip accents + lowercase
            tokenizer.normalizer = Sequence([
                NFD(),
                StripAccents(),
                Lowercase(),
            ])
        elif normalization_scheme == "lower":
            tokenizer.normalizer = Lowercase()
        
        # Set pre-tokenizer to split on whitespace
        tokenizer.pre_tokenizer = Whitespace()
        
        # Create trainer
        trainer = BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=1,
            special_tokens=["<pad>", "<unk>", "<eos>"],
            show_progress=True
        )
        
        # Train on corpus - tokenizer.train expects file paths or needs to write to temp files
        # So we'll use train_from_iterator instead
        import tempfile
        
        # Write corpus to temporary file
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
            for text in corpus:
                f.write(text + '\n')
            temp_path = f.name
        
        try:
            tokenizer.train([temp_path], trainer=trainer)
        finally:
            import os
            os.remove(temp_path)
        
        return tokenizer, normalization_scheme
    
    @classmethod
    def from_corpus(cls, corpus: Iterable[str], 
                    vocab_size: int = 10000,
                    normalization_scheme: str = "none"):
        """Create BPETokenizerHF from corpus."""
        tokenizer, normalization_scheme = cls.build_vocab(
            corpus, vocab_size, normalization_scheme)
        return cls(tokenizer=tokenizer, normalization_scheme=normalization_scheme)


class WordPieceTokenizer(BaseTokenizer):
    def __init__(self):
        raise NotImplementedError("WordPieceTokenizer is not implemented yet.")

class UnigramTokenizer(BaseTokenizer):
    def __init__(self):
        raise NotImplementedError("UnigramTokenizer is not implemented yet.")

# Utility functions for executing configs
# and caching tokenizers and tokenized datasets

tokenizer_base_path = "results/tokenizer"

def _get_tokenizer(tokenizer_name: str) -> type:
    if tokenizer_name == "char-level":
        return CharLevelTokenizer
    elif tokenizer_name == "word-level":
        return WordLevelTokenizer
    elif tokenizer_name == "bpe":
        return BPETokenizer
    elif tokenizer_name == "bpe-hf":
        return BPETokenizerHF
    else:
        raise ValueError(f"Unknown tokenizer name: {tokenizer_name}")

def _get_cached_config(tokenizer_name: str) -> Optional[Dict[str, Any]]:
    config_file = f"{tokenizer_base_path}/{tokenizer_name}/config.json"
    if os.path.exists(config_file):
        with open(config_file, "r") as f:
            return json.load(f)
    return None

def _load_cached_tokenizer(config: TokenizerConfig) -> Optional[BaseTokenizer]:
    cached_config = _get_cached_config(config.name)
    if cached_config is None \
        or TokenizerConfig.differs_in(config, TokenizerConfig(**cached_config),
                            keys=["type", "normalization_scheme",
                                  "vocab_size", "min_occurences"]):
        return None
    cached_tokenizer_path = f"{tokenizer_base_path}/{config.name}/tokenizer.pkl"
    if not os.path.exists(cached_tokenizer_path):
        return None
    with open(cached_tokenizer_path, "rb") as f:
        return pickle.load(f)

def get_tokenizer(
    name: Optional[str] = None,
    config: TokenizerConfig = None,
    dataset: str = "train",
) -> BaseTokenizer:
    if config is None:
        if name is None:
            raise ValueError("Either name or config must be provided")
        config = TokenizerConfig.from_json("config.json", name)

    tokenizer = _load_cached_tokenizer(config)
    if tokenizer is not None:
        return tokenizer

    tokenizer_type = config.type
    dataset_path = load_config("config.json").get("dataset").get(dataset, None)
    if dataset_path is None:
        raise ValueError(f"Dataset path not found for {dataset} in config.json")

    import pandas as pd
    df = pd.read_csv(dataset_path)
    kwargs = {"normalization_scheme": config.normalization_scheme}
    if tokenizer_type == "char-level" and config.min_occurences is not None:
        kwargs["min_occurences"] = config.min_occurences
    elif tokenizer_type in ["word-level", "bpe", "bpe-hf"] and config.vocab_size is not None:
        kwargs["vocab_size"] = config.vocab_size

    tokenizer = _get_tokenizer(tokenizer_type).from_corpus(
        corpus=df["review"].values,
        **kwargs
    )

    tokenizer_cache_path = f"{tokenizer_base_path}/{config.name}/tokenizer.pkl"
    os.makedirs(os.path.dirname(tokenizer_cache_path), exist_ok=True)
    with open(tokenizer_cache_path, "wb") as f:
        pickle.dump(tokenizer, f)
    with open(f"{tokenizer_base_path}/{config.name}/config.json", "w") as f:
        json.dump(config.__dict__, f)
    return tokenizer

def _load_clipped_dataset(config: TokenizerConfig, dataset: str):
    cached_config = _get_cached_config(config.name)
    if cached_config is None \
        or cached_config.get("clip_length", None) != config.clip_length:
        return None
    clipped_data_path = f"{tokenizer_base_path}/{config.name}/" \
                        f"{dataset}_clipped.pth"
    if not os.path.exists(clipped_data_path):
        return None
    
    print(f"Loading clipped dataset {dataset} from cache for {config.name}...")
    
    data = torch.load(clipped_data_path)
    return data

def _clip_dataset_from_cache(config: TokenizerConfig, dataset: str):
    cached_config = _get_cached_config(config.name)
    if cached_config is None \
        or TokenizerConfig.differs_in(config, TokenizerConfig(**cached_config), 
                            keys=["type", "normalization_scheme", 
                                  "vocab_size", "min_occurences"]):
        return None
    tokenized_data_path = f"{tokenizer_base_path}/" \
                          f"{config.name}/{dataset}_tokenized.pkl"
    if not os.path.exists(tokenized_data_path):
        return None
    with open(tokenized_data_path, "rb") as f:
        tokenized_data = pickle.load(f)
        
        print(f"Clipping dataset {dataset} with clip length {config.clip_length}...")
        
        clipped_data = torch.Tensor([
            [0] * (config.clip_length - len(sample)) + sample
            if len(sample) < config.clip_length 
            else sample[-config.clip_length:]
            for sample in tokenized_data["data"]
        ])
        clipped_labels = torch.Tensor(tokenized_data["labels"])
        
        # Save the clipped dataset to cache
        clipped_data_path = f"{tokenizer_base_path}/{config.name}/" \
                            f"{dataset}_clipped.pth"
        torch.save((clipped_data, clipped_labels), clipped_data_path)
        return _load_clipped_dataset(config, dataset)
    
def _tokenize_dataset_from_cache(config: TokenizerConfig, dataset: str):
    cached_config = _get_cached_config(config.name)
    if cached_config is None \
        or TokenizerConfig.differs_in(config, TokenizerConfig(**cached_config), 
                            keys=["type", "normalization_scheme", 
                                  "vocab_size", "min_occurences"]):
        return None
    cached_tokenizer_path = f"{tokenizer_base_path}/{config.name}/tokenizer.pkl"
    if not os.path.exists(cached_tokenizer_path):
        return None
    with open(cached_tokenizer_path, "rb") as f:
        tokenizer = pickle.load(f)
        
        print(f"Tokenizing dataset {dataset} using cached tokenizer"
              f"for {config.name}...")
        dataset_path = load_config("config.json").get("dataset").get(dataset, None)
        if dataset_path is None:
            raise ValueError(f"Dataset path not found for {dataset} in config.json")
        import pandas as pd
        df = pd.read_csv(dataset_path)
        tokenized_data = {
            "data": [tokenizer.encode(text).ids for text in df["review"].values],
            "labels": [1 if label == "positive" else 0 for label in 
                       df["sentiment"].values],
        }
        tokenized_data_path = f"{tokenizer_base_path}/" \
                              f"{config.name}/{dataset}_tokenized.pkl"
        with open(tokenized_data_path, "wb") as f:
            pickle.dump(tokenized_data, f)
        return _clip_dataset_from_cache(config, dataset)
        
def _train_tokenizer_from_config(
    config: TokenizerConfig, dataset: str) -> BaseTokenizer:
    tokenizer_type = config.type
    
    # build the corpus from the dataset
    corpus = []
    dataset_path = load_config("config.json").get("dataset").get(dataset, None)
    if dataset_path is None:
        raise ValueError(f"Dataset path not found for {dataset} in config.json")
    import pandas as pd
    df = pd.read_csv(dataset_path)
    corpus = df["review"].values
    
    # train the tokenizer with the corpus and the config
    # Build kwargs based on tokenizer type
    kwargs = {"normalization_scheme": config.normalization_scheme}
    if tokenizer_type == "char-level" and config.min_occurences is not None:
        kwargs["min_occurences"] = config.min_occurences
    elif tokenizer_type in ["word-level", "bpe", "bpe-hf"] and config.vocab_size is not None:
        kwargs["vocab_size"] = config.vocab_size
    
    tokenizer = _get_tokenizer(tokenizer_type).from_corpus(
        corpus=corpus,
        **kwargs
    )
    
    # save tokenizer to cache
    tokenizer_cache_path = f"{tokenizer_base_path}/{config.name}/tokenizer.pkl"
    os.makedirs(os.path.dirname(tokenizer_cache_path), exist_ok=True)
    with open(tokenizer_cache_path, "wb") as f:
        pickle.dump(tokenizer, f)
    with open(f"{tokenizer_base_path}/{config.name}/config.json", "w") as f:
        json.dump(config.__dict__, f)
    return _tokenize_dataset_from_cache(config, dataset)

fallback_path = [
    _load_clipped_dataset,
    _clip_dataset_from_cache,
    _tokenize_dataset_from_cache,
    _train_tokenizer_from_config
]

def get_tokenized_dataset(name: Optional[str] = None, config: TokenizerConfig = None, dataset: str = None):
    if config is None:
        if name is None:
            raise ValueError("Either name or config must be provided")
        # load config from main config.json
        config = TokenizerConfig.from_json("config.json", name)
    
    for func in fallback_path:
        result = func(config, dataset)
        if result is not None:
            return result
    raise RuntimeError(f"Failed to get tokenized dataset for {dataset} with "
                       f"config {config}")
