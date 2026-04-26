# Training pipeline

import json
import os
from typing import Dict, Any
from datetime import datetime

import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
from torch.amp import autocast
from sklearn.metrics import accuracy_score, f1_score, recall_score, \
    precision_score

from model import Transformer, RNN, MLPClassifier
from config_utils import load_config

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

def evaluate_binary_cls_model(model, loader, device, loss_fn=None):
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0.0
    num_batches = 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs, mode="binary_cls").squeeze(-1)
            if loss_fn is not None:
                total_loss += loss_fn(outputs, labels.float()).detach()
                num_batches += 1
            predictions = (outputs > 0).long()
            all_preds.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    metrics = {
        "accuracy": accuracy_score(all_labels, all_preds),
        "f1": f1_score(all_labels, all_preds, zero_division=0),
        "recall": recall_score(all_labels, all_preds, zero_division=0),
        "precision": precision_score(all_labels, all_preds, zero_division=0)
    }
    if loss_fn is not None:
        metrics["loss"] = (total_loss / num_batches).item()
    return metrics

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

    # Create timestamp for this training run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_dir = f"results/checkpoints/{config['tokenizer']}" \
                     f"-{config.get('run_name', 'run')}/{timestamp}"
    os.makedirs(checkpoint_dir, exist_ok=True)

    # 1. Create the training data source
    load_full_data = config.get("load_full_data", False)
    label_dtype = torch.float if config["training_type"] == "label_supervised" \
        else torch.long
        
    if load_full_data:
        train_data, train_labels = get_tokenized_dataset(
            name=config["tokenizer"], dataset="train"
        )
        train_data = train_data.long().to(device)
        train_labels = train_labels.to(device=device, dtype=label_dtype)
        num_train_samples = train_data.size(0)
        num_train_batches = (
            num_train_samples + config["batch_size"] - 1
        ) // config["batch_size"]
    else:
        train_loader = get_loader(
            config["tokenizer"], "train", config["batch_size"], shuffle=True
        )
        num_train_batches = len(train_loader)

    # 4. Initialize model, optimizer, loss function
    model_type = config.get("model_type", "transformer")
    if model_type == "transformer":
        model = Transformer(
            vocab_size=config["vocab_size"],
            embedding_dim=config["embedding_dim"],
            num_heads=config["num_heads"],
            num_layers=config["num_layers"],
            ff_dim=config["ff_dim"],
            dropout_rate=config["dropout_rate"],
            positional_encoding=config["positional_encoding"],
            max_seq_len=config["max_seq_len"],
            causal_attention=config.get("causal_attention", False),
            pad_token_id=config["pad_token_id"],
            use_kaiming_init=config.get("use_kaiming_init", True)
        ).to(device)
    elif model_type == "rnn":
        model = RNN(
            vocab_size=config["vocab_size"],
            embedding_dim=config["embedding_dim"],
            hidden_dim=config["hidden_dim"],
            num_layers=config["num_layers"],
            dropout_rate=config["dropout_rate"],
            bidirectional=config.get("bidirectional", False),
            use_kaiming_init=config.get("use_kaiming_init", True)
        ).to(device)
    elif model_type == "mlp":
        model = MLPClassifier(
            vocab_size=config["vocab_size"],
            embedding_dim=config["embedding_dim"],
            hidden_dim=config["hidden_dim"],
            mlp_depth=config.get("mlp_depth", 2),
            dropout_rate=config["dropout_rate"],
            pad_token_id=config["pad_token_id"],
        ).to(device)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    optimizer = optim.AdamW(model.parameters(), lr=config["learning_rate"])
    if config["scheduler"] == "cosine":
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
    elif config["scheduler"] == "constant":
        scheduler = optim.lr_scheduler.ConstantLR(optimizer)
    else:
        raise ValueError(f"Unknown scheduler type: {config['scheduler']}")
    
    # Select loss function based on training type
    training_type = config["training_type"]
    if training_type == "label_supervised":
        loss_fn = torch.nn.BCEWithLogitsLoss()
    elif training_type == "pretrain":
        loss_fn = torch.nn.CrossEntropyLoss()
    elif training_type == "finetune":
        loss_fn = torch.nn.CrossEntropyLoss()
    else:
        raise ValueError(f"Unknown training type: {training_type}")
    
    # Initialize mixed precision training (BF16)
    use_amp = device.type == "cuda" and config.get("mixed_precision", False)
    save_interval = int(config.get("save_interval", 0) or 0)
    validate_during_training = config.get(
        "validate_during_training",
        training_type in ("label_supervised", "finetune"),
    )
    validation_dataset = config.get("validation_dataset", "val")
    validation_batch_size = config.get("validation_batch_size", 64)
    if validate_during_training:
        train_eval_loader = get_loader(
            config["tokenizer"],
            "train",
            validation_batch_size,
            shuffle=False,
        )
        val_loader = get_loader(
            config["tokenizer"],
            validation_dataset,
            validation_batch_size,
            shuffle=False,
        )
    else:
        train_eval_loader = None
        val_loader = None
    best_metric = float("-inf")
    best_f1 = float("-inf")
    best_summary = None
    history = []

    if training_type == "pretrain":
        def compute_loss(inputs, labels):
            outputs = model(inputs, mode="prob")
            return loss_fn(
                outputs.reshape(-1, outputs.size(-1)),
                inputs.reshape(-1)
            )
    elif training_type == "label_supervised":
        def compute_loss(inputs, labels):
            outputs = model(inputs, mode="binary_cls").squeeze(-1)
            return loss_fn(outputs, labels)
    else:
        def compute_loss(inputs, labels):
            outputs = model(inputs, mode="binary_cls").squeeze(-1)
            return loss_fn(torch.stack((-outputs, outputs), dim=1), labels)
    
    # 5. Training loop
    for epoch in range(config["num_epochs"]):
        model.train()
        total_loss = 0.0
        
        # Manual dataloader to support full data loading
        if load_full_data:
            indices = torch.randperm(num_train_samples, device=device)
            batch_iter = (
                (
                    train_data[indices[start:start + config["batch_size"]]],
                    train_labels[indices[start:start + config["batch_size"]]]
                )
                for start in range(0, num_train_samples, config["batch_size"])
            )
        else:
            batch_iter = train_loader

        for inputs, labels in batch_iter:
            if not load_full_data:
                inputs = inputs.to(device)
                labels = labels.to(device=device, dtype=label_dtype)
            
            optimizer.zero_grad(set_to_none=True)
            
            # Forward pass with optional mixed precision
            if use_amp:
                with autocast(device_type="cuda", dtype=torch.bfloat16):
                    loss = compute_loss(inputs, labels)
                loss.backward()
                optimizer.step()
            else:
                loss = compute_loss(inputs, labels)
                loss.backward()
                optimizer.step()
            
            total_loss += loss.detach()
        
        scheduler.step()
        avg_loss = (total_loss / num_train_batches).item()
        current_lr = optimizer.param_groups[0]["lr"]
        epoch_record = {
            "epoch": epoch + 1,
            "train_loss": avg_loss,
            "learning_rate": current_lr,
        }
        log_payload = {
            "epoch": epoch + 1,
            "train_loss": avg_loss,
            "learning_rate": current_lr,
            "training_type": training_type
        }

        if val_loader is not None:
            train_metrics = evaluate_binary_cls_model(
                model, train_eval_loader, device, loss_fn=loss_fn
            )
            val_metrics = evaluate_binary_cls_model(
                model, val_loader, device, loss_fn=loss_fn
            )
            epoch_record["train"] = train_metrics
            epoch_record[validation_dataset] = val_metrics
            epoch_record.update({
                "train_loss": train_metrics["loss"],
                "train_acc": train_metrics["accuracy"],
                "train_f1": train_metrics["f1"],
                "val_accuracy": val_metrics["accuracy"],
                "val_loss": val_metrics["loss"],
            })
            log_payload.update({
                "train_acc": train_metrics["accuracy"],
                "train_f1": train_metrics["f1"],
                "train_loss": train_metrics["loss"],
                "val_accuracy": val_metrics["accuracy"],
                "val_loss": val_metrics["loss"],
                f"{validation_dataset}_f1": val_metrics["f1"],
            })

            is_best = (
                val_metrics["accuracy"] > best_metric
                or (
                    val_metrics["accuracy"] == best_metric
                    and val_metrics["f1"] > best_f1
                )
            )
            if is_best:
                best_metric = val_metrics["accuracy"]
                best_f1 = val_metrics["f1"]
                best_path = f"{checkpoint_dir}/best.pth"
                torch.save(model.state_dict(), best_path)
                best_summary = {
                    "best_epoch": epoch + 1,
                    "best_checkpoint": "best.pth",
                    "best_checkpoint_path": best_path,
                    "selection_metric": f"{validation_dataset}.accuracy",
                    "train_loss": train_metrics["loss"],
                    "metrics": {
                        "train": train_metrics,
                        validation_dataset: val_metrics,
                    },
                }
                with open(f"{checkpoint_dir}/best_checkpoint.json", "w") as f:
                    json.dump(best_summary, f, indent=2)
                print(
                    f"Saved best checkpoint to {best_path} "
                    f"({validation_dataset} Acc: {val_metrics['accuracy']:.4f}, "
                    f"F1: {val_metrics['f1']:.4f})"
                )

        history.append(epoch_record)
        with open(f"{checkpoint_dir}/training_history.json", "w") as f:
            json.dump(history, f, indent=2)

        # Log metrics to wandb
        wandb.log(log_payload)
        
        status = f"Epoch {epoch+1}/{config['num_epochs']}, Loss: {avg_loss:.4f}, LR: {current_lr:.6f}"
        if val_loader is not None:
            status += (
                f", train Acc: {train_metrics['accuracy']:.4f}, "
                f", {validation_dataset} Acc: {val_metrics['accuracy']:.4f}, "
                f"Loss: {val_metrics['loss']:.4f}"
            )
        print(status)
        
        # Save checkpoint every save_interval epochs
        if save_interval > 0 and (epoch + 1) % save_interval == 0:
            checkpoint_path = f"{checkpoint_dir}/epoch{epoch+1}.pth"
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")

    # Save final checkpoint
    checkpoint_path = f"{checkpoint_dir}/final.pth"
    torch.save(model.state_dict(), checkpoint_path)
    print(f"Saved checkpoint to {checkpoint_path}")
    if best_summary is None and validate_during_training:
        print("No best checkpoint was saved during training.")

    # Finish wandb run
    wandb.finish()

    return checkpoint_path
    

def train_pipeline(name: str):
    """
    Execute a full training pipeline from config.json.
    
    Args:
        name: The training config name (e.g., "end2end-config1")
    """
    config_dict = load_config("config.json")
    
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
            "scheduler": stage["scheduler"]["type"],
            "warmup_steps": stage["scheduler"].get("warmup_steps", 0),
            "mixed_precision": model_config.get("mixed_precision", False),
            "load_full_data": stage.get("load_full_data", False),
            "training_type": stage["type"],
            "data_augmentation": stage.get("data_augmentation", []),
            "save_interval": stage.get("save_interval", 0),
            "validate_during_training": stage.get("validate_during_training", stage["type"] != "pretrain"),
            "validation_dataset": stage.get("validation_dataset", "val"),
            "validation_batch_size": stage.get("validation_batch_size", 64),
            "pad_token_id": 0,
            "type": stage["type"],
            "model_type": model_config["type"],
            "embedding_dim": model_config["embedding_dim"],
            "dropout_rate": model_config["dropout_rate"],
            "max_seq_len": model_config.get("max_seq_len", tokenizer_config.get("clip_length", 512)),
            "run_name": f"{name}-stage{stage_idx+1}"
        }

        # Add model-specific parameters
        if model_config["type"] == "transformer":
            stage_config.update({
                "num_layers": model_config["num_layers"],
                "num_heads": model_config["num_heads"],
                "ff_dim": int(model_config["embedding_dim"] * model_config.get("ff_ratio", 4)),
                "positional_encoding": model_config.get("positional_encoding", "sinusoidal"),
                "causal_attention": model_config.get("causal_attention", False)
            })
        elif model_config["type"] == "rnn":
            stage_config.update({
                "num_layers": model_config["num_layers"],
                "hidden_dim": model_config["hidden_dim"],
                "bidirectional": model_config.get("bidirectional", False)
            })
        elif model_config["type"] == "mlp":
            stage_config.update({
                "hidden_dim": model_config["hidden_dim"],
                "mlp_depth": model_config.get("mlp_depth", 2)
            })
        
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


def evaluate(config_name: str, checkpoint_path: str, datasets = ["train", "val"]):
    """Evaluate model on train/val/test sets and return accuracy and F1 scores"""
    config_dict = load_config("config.json")

    train_config = config_dict["train"][config_name]
    tokenizer_name = train_config["tokenizer"]
    model_name = train_config["model"]

    tokenizer_config = config_dict["preprocess"][tokenizer_name]
    model_config = config_dict["model"][model_name]

    vocab_size = tokenizer_config.get("vocab_size", 256)

    # Setup device
    device = get_device()

    # Load datasets
    loaders = []
    for dataset_name in datasets:
        loader = get_loader(tokenizer_name, dataset_name, batch_size=64, shuffle=False)
        loaders.append((dataset_name, loader))

    # Initialize model
    model_type = model_config["type"]
    if model_type == "transformer":
        model = Transformer(
            vocab_size=vocab_size,
            embedding_dim=model_config["embedding_dim"],
            num_heads=model_config["num_heads"],
            num_layers=model_config["num_layers"],
            ff_dim=int(model_config["embedding_dim"] * model_config.get("ff_ratio", 4)),
            dropout_rate=model_config["dropout_rate"],
            positional_encoding=model_config.get("positional_encoding", "sinusoidal"),
            max_seq_len=model_config.get("max_seq_len", tokenizer_config.get("clip_length", 512)),
            causal_attention=model_config.get("causal_attention", False),
            pad_token_id=0,
        ).to(device)
    elif model_type == "rnn":
        model = RNN(
            vocab_size=vocab_size,
            embedding_dim=model_config["embedding_dim"],
            hidden_dim=model_config["hidden_dim"],
            num_layers=model_config["num_layers"],
            dropout_rate=model_config["dropout_rate"],
            bidirectional=model_config.get("bidirectional", False)
        ).to(device)
    elif model_type == "mlp":
        model = MLPClassifier(
            vocab_size=vocab_size,
            embedding_dim=model_config["embedding_dim"],
            hidden_dim=model_config["hidden_dim"],
            mlp_depth=model_config.get("mlp_depth", 2),
            dropout_rate=model_config["dropout_rate"],
            pad_token_id=0,
        ).to(device)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Load checkpoint
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    results = {}
    
    # Evaluate on each dataset
    for dataset_name, loader in loaders:
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for inputs, labels in loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs, mode="binary_cls").squeeze(-1)
                predictions = (outputs > 0).long()
                all_preds.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        accuracy = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds)
        results[dataset_name] = {"accuracy": accuracy, "f1": f1}
    
    return results
