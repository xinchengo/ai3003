# Training pipeline

import json
import os
from typing import Dict, Any

import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
from model import Transformer

import wandb

from tokenizer import get_tokenized_dataset

def get_device():
    """Get the best available device"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

def get_loader(tokenizer_name: str, dataset: str, batch_size: int, shuffle: bool = True):
    """Create dataloader for given tokenizer and dataset"""
    data, labels = get_tokenized_dataset(name=tokenizer_name, dataset=dataset)
    tensor_dataset = TensorDataset(data.long(), labels.long())
    return DataLoader(
        tensor_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=4,
        pin_memory=True,
        multiprocessing_context="spawn",
        persistent_workers=True,
    )

def train_with_config(config: Dict[str, Any]):
    # Initialize wandb
    wandb.init(
        project="ai3003-lab3",
        config=config,
        name=f"{config['tokenizer']}-{config.get('run_name', 'run')}"
    )

    # Setup device
    device = get_device()
    print(f"Using device: {device}")

    # 1. Create dataloader
    train_loader = get_loader(config["tokenizer"], "train", config["batch_size"], shuffle=True)

    # 4. Initialize model, optimizer, loss function
    model = Transformer(
        vocab_size=config["vocab_size"],
        embedding_dim=config["embedding_dim"],
        num_heads=config["num_heads"],
        num_layers=config["num_layers"],
        ff_dim=config["ff_dim"],
        dropout_rate=config["dropout_rate"],
        positional_encoding=config["positional_encoding"]
    ).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=config["learning_rate"])
    scheduler = optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[
            optim.lr_scheduler.LinearLR(optimizer, 
                                        start_factor=0.1, 
                                        total_iters=config["warmup_steps"]),
            optim.lr_scheduler.CosineAnnealingLR(
                optimizer, 
                T_max=config["num_epochs"] - config["warmup_steps"])
        ],
        milestones=[config["warmup_steps"]]
    )
    
    # Select loss function based on training type
    training_type = config["training_type"]
    if training_type == "label_supervised":
        loss_fn = torch.nn.CrossEntropyLoss()
    elif training_type == "pretrain":
        loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100)
    elif training_type == "finetune":
        loss_fn = torch.nn.CrossEntropyLoss()
    else:
        raise ValueError(f"Unknown training type: {training_type}")
    
    # 5. Training loop
    for epoch in range(config["num_epochs"]):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            inputs, labels = batch
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            
            # TODO: Different forward passes for different training types
            # if training_type == "pretrain":
            #     outputs = model(inputs, mode="pretrain")
            # elif training_type in ["label_supervised", "finetune"]:
            #     outputs = model(inputs, mode="binary_cls").squeeze(-1)
            
            outputs = model(inputs, mode="binary_cls").squeeze(-1)
            loss = loss_fn(outputs, labels.float())
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()
        avg_loss = total_loss / len(train_loader)
        current_lr = optimizer.param_groups[0]["lr"]
        
        # Log metrics to wandb
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": avg_loss,
            "learning_rate": current_lr,
            "training_type": training_type
        })
        
        print(f"Epoch {epoch+1}/{config['num_epochs']}, Loss: {avg_loss:.4f}, LR: {current_lr:.6f}")

    # Save final checkpoint
    checkpoint_dir = f"results/checkpoints/{config['tokenizer']}-{config.get('run_name', 'run')}"
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = f"{checkpoint_dir}/final.pth"
    torch.save(model.state_dict(), checkpoint_path)
    print(f"Saved checkpoint to {checkpoint_path}")

    # Finish wandb run
    wandb.finish()

    return checkpoint_path
    

def train_pipeline(name: str):
    """
    Execute a full training pipeline from config.json.
    
    Args:
        name: The training config name (e.g., "end2end-config1")
    """
    with open("config.json", "r") as f:
        config_dict = json.load(f)
    
    train_config = config_dict["train"].get(name, None)
    if train_config is None:
        raise ValueError(f"Training config {name} not found in config.json")
    
    # Load tokenizer and model configs
    tokenizer_name = train_config["tokenizer"]
    model_name = train_config["model"]
    
    tokenizer_config = config_dict["preprocess"][tokenizer_name]
    model_config = config_dict["model"][model_name]
    
    # Get vocab size from tokenizer config or results
    vocab_size = tokenizer_config.get("vocab_size", 256)  # default for char-level
    
    sequence = train_config["sequence"]
    
    # Execute each stage in the training sequence
    for stage_idx, stage in enumerate(sequence):
        print(f"\n{'='*60}")
        print(f"Stage {stage_idx + 1}/{len(sequence)}: {stage['type'].upper()}")
        print(f"{'='*60}\n")
        
        # Merge configs for this stage
        stage_config = {
            "tokenizer": tokenizer_name,
            "vocab_size": vocab_size,
            "batch_size": stage["batch_size"],
            "num_epochs": stage["num_epochs"],
            "learning_rate": stage["learning_rate"],
            "warmup_steps": stage["scheduler"].get("warmup_steps", 0),
            "training_type": stage["type"],
            "data_augmentation": stage.get("data_augmentation", []),
            # Model hyperparameters
            "embedding_dim": model_config["embedding_dim"],
            "num_layers": model_config["num_layers"],
            "num_heads": model_config["num_heads"],
            "ff_dim": model_config.get("ff_dim", model_config["embedding_dim"] * 4),
            "dropout_rate": model_config["dropout_rate"],
            "positional_encoding": model_config["positional_encoding"],
            "run_name": f"{name}-stage{stage_idx+1}"
        }
        
        # TODO: Apply data augmentation based on stage_config["data_augmentation"]
        # TODO: Modify training objective based on stage_config["training_type"]
        #       - "label_supervised": standard classification
        #       - "pretrain": masked language modeling
        #       - "finetune": fine-tune on downstream task
        
        # Execute training for this stage
        checkpoint_path = train_with_config(stage_config)

    print("\n" + "="*60)
    print("Training pipeline completed!")
    print("="*60)

    return checkpoint_path


def evaluate(config_name: str, checkpoint_path: str):
    """Evaluate model on test set"""
    with open("config.json", "r") as f:
        config_dict = json.load(f)

    train_config = config_dict["train"][config_name]
    tokenizer_name = train_config["tokenizer"]
    model_name = train_config["model"]

    tokenizer_config = config_dict["preprocess"][tokenizer_name]
    model_config = config_dict["model"][model_name]

    vocab_size = tokenizer_config.get("vocab_size", 256)

    # Setup device
    device = get_device()
    print(f"Using device: {device}")

    # Load test dataset
    test_loader = get_loader(tokenizer_name, "test", batch_size=64, shuffle=False)

    # Initialize model
    model = Transformer(
        vocab_size=vocab_size,
        embedding_dim=model_config["embedding_dim"],
        num_heads=model_config["num_heads"],
        num_layers=model_config["num_layers"],
        ff_dim=model_config.get("ff_dim", model_config["embedding_dim"] * 4),
        dropout_rate=model_config["dropout_rate"],
        positional_encoding="sinusoidal"
    ).to(device)

    # Load checkpoint
    model.load_state_dict(torch.load(checkpoint_path))
    model.eval()

    # Evaluate
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs, mode="binary_cls").squeeze(-1)
            predictions = (outputs > 0).long()
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total
    print(f"Test Accuracy: {accuracy:.4f}")
    return accuracy
