import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, log_loss

from model import SimCLRModel

try:
    import wandb
except ImportError:
    wandb = None


def _log_wandb(metrics, wandb_run=None):
    run = wandb_run
    if run is None and wandb is not None:
        run = wandb.run
    if run is not None:
        run.log(metrics)


def _save_training_state(path, model, optimizer, epoch, history=None, config=None):
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "epoch": epoch,
        "history": history,
        "config": config,
    }, path)

def nt_xent_loss(z1, z2, temperature=0.5):
    batch_size = z1.size(0)

    z = torch.cat([z1, z2], dim=0)
    z = F.normalize(z, dim=1)

    logits = torch.mm(z, z.t()) / temperature
    logits.fill_diagonal_(float("-inf"))

    labels = torch.arange(2 * batch_size, device=z.device)
    labels = (labels + batch_size) % (2 * batch_size)

    return F.cross_entropy(logits, labels)

def nt_logistic_loss(z1, z2, temperature=0.5):
    B = z1.size(0)
    
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    
    similarity_matrix = torch.mm(z1, z2.t()) / temperature
    
    labels = torch.eye(B, device=z1.device)
    
    loss_fn = nn.BCEWithLogitsLoss()
    loss = loss_fn(similarity_matrix, labels)
    
    return loss

def triplet_loss(z1, z2, margin=1.0):
    B = z1.size(0)
    
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    
    negative_indices = torch.roll(torch.arange(B, device=z1.device), shifts=1)
    negatives = z2[negative_indices]

    loss_fn = nn.TripletMarginLoss(margin=margin)
    loss = loss_fn(z1, z2, negatives)
    
    return loss

def pretrain_simclr(
    model: SimCLRModel,
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    loss_fn = nt_xent_loss,
    num_epochs: int = 100,
    wandb_run = None,
    log_prefix: str = "pretrain",
    checkpoint_dir = None,
    save_interval: int = 0,
    mixed_precision: bool = True,
    start_epoch: int = 0,
    config = None,
):
    model.to(device)
    history = []
    use_amp = mixed_precision and device.type == "cuda"
    
    for epoch in range(num_epochs):
        global_epoch = start_epoch + epoch + 1
        if device.type == "cuda":
            torch.cuda.synchronize()
        start_time = time.perf_counter()
        model.train()
        total_loss = 0.0
        num_samples = 0
        data_time = 0.0
        batch_wait_start = time.perf_counter()
        
        for (view_1, view_2) in dataloader:
            data_time += time.perf_counter() - batch_wait_start
            view_1 = view_1.to(device, non_blocking=True)
            view_2 = view_2.to(device, non_blocking=True)
            
            num_samples += view_1.size(0)
            
            views = torch.cat([view_1, view_2], dim=0)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
                projections = model(views, mode='projection')
                z1, z2 = torch.chunk(projections, chunks=2, dim=0)
                loss = loss_fn(z1, z2)
            
            # Backpropagation and optimization
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            batch_wait_start = time.perf_counter()
        
        avg_loss = total_loss / len(dataloader)
        if device.type == "cuda":
            torch.cuda.synchronize()
        epoch_time = time.perf_counter() - start_time
        compute_time = max(epoch_time - data_time, 0.0)
        print(f"Epoch [{global_epoch}], Loss: {avg_loss:.4f}")
        epoch_record = {
            "epoch": global_epoch,
            "loss": avg_loss,
            "epoch_time_sec": epoch_time,
            "data_time_sec": data_time,
            "compute_time_sec": compute_time,
            "data_time_ratio": data_time / epoch_time if epoch_time > 0 else 0.0,
            "samples_per_sec": num_samples / epoch_time,
            "views_per_sec": (2 * num_samples) / epoch_time,
        }
        log_payload = {
            "epoch": epoch_record["epoch"],
            f"{log_prefix}/loss": epoch_record["loss"],
            f"{log_prefix}/epoch_time_sec": epoch_record["epoch_time_sec"],
            f"{log_prefix}/data_time_sec": epoch_record["data_time_sec"],
            f"{log_prefix}/compute_time_sec": epoch_record["compute_time_sec"],
            f"{log_prefix}/data_time_ratio": epoch_record["data_time_ratio"],
            f"{log_prefix}/samples_per_sec": epoch_record["samples_per_sec"],
            f"{log_prefix}/views_per_sec": epoch_record["views_per_sec"],
        }
        history.append(epoch_record)
        _log_wandb(log_payload, wandb_run)
        if checkpoint_dir is not None and save_interval > 0 and global_epoch % save_interval == 0:
            _save_training_state(
                checkpoint_dir / f"{log_prefix}_epoch{global_epoch}.pth",
                model,
                optimizer,
                global_epoch,
                history=history,
                config=config,
            )

    return history

@torch.inference_mode()
def extract_encoder_features(
    model: SimCLRModel,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
):
    model.to(device)
    model.eval()

    features = []
    labels = []
    for images, batch_labels in dataloader:
        images = images.to(device, non_blocking=True)
        batch_features = model.encoder(images)
        features.append(batch_features.cpu().numpy())
        labels.append(batch_labels.cpu().numpy())

    return np.concatenate(features, axis=0), np.concatenate(labels, axis=0)


def evaluate_logistic_regression_classifier(
    classifier: LogisticRegression,
    features,
    labels,
    mode: str = "eval",
):
    probabilities = classifier.predict_proba(features)[:, 1]
    preds = classifier.predict(features)
    metrics = {
        "loss": float(log_loss(labels, probabilities, labels=[0, 1])),
        "accuracy": float(accuracy_score(labels, preds)),
        "f1": float(f1_score(labels, preds, zero_division=0)),
    }
    print(
        f"{mode.capitalize()}: Loss: {metrics['loss']:.4f}, "
        f"Accuracy: {metrics['accuracy']:.4f}, F1-Score: {metrics['f1']:.4f}"
    )
    return metrics


def fit_logistic_regression_classifier(
    model: SimCLRModel,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    train_eval_dataloader: torch.utils.data.DataLoader = None,
    test_eval_dataloader: torch.utils.data.DataLoader = None,
    wandb_run = None,
    log_prefix: str = "linear_probe",
    max_iter: int = 1000,
    random_state: int = 42,
):
    start_time = time.perf_counter()
    train_features, train_labels = extract_encoder_features(model, dataloader, device)

    classifier = LogisticRegression(
        max_iter=max_iter,
        random_state=random_state,
    )
    classifier.fit(train_features, train_labels)
    fit_time = time.perf_counter() - start_time

    train_metrics = evaluate_logistic_regression_classifier(
        classifier,
        train_features,
        train_labels,
        "train",
    )
    record = {
        "fit_time_sec": fit_time,
        "num_train_samples": int(train_labels.shape[0]),
        "feature_dim": int(train_features.shape[1]),
        "train": train_metrics,
        "solver": classifier.solver,
        "max_iter": max_iter,
        "n_iter": int(classifier.n_iter_[0]),
    }
    log_payload = {
        f"{log_prefix}/fit_time_sec": record["fit_time_sec"],
        f"{log_prefix}/num_train_samples": record["num_train_samples"],
        f"{log_prefix}/feature_dim": record["feature_dim"],
        f"{log_prefix}/train_loss": train_metrics["loss"],
        f"{log_prefix}/train_accuracy": train_metrics["accuracy"],
        f"{log_prefix}/train_f1": train_metrics["f1"],
        f"{log_prefix}/n_iter": record["n_iter"],
    }

    if train_eval_dataloader is not None and train_eval_dataloader is not dataloader:
        train_eval_features, train_eval_labels = extract_encoder_features(
            model,
            train_eval_dataloader,
            device,
        )
        train_eval_metrics = evaluate_logistic_regression_classifier(
            classifier,
            train_eval_features,
            train_eval_labels,
            "train",
        )
        record["train"] = train_eval_metrics
        log_payload.update({
            f"{log_prefix}/train_loss": train_eval_metrics["loss"],
            f"{log_prefix}/train_accuracy": train_eval_metrics["accuracy"],
            f"{log_prefix}/train_f1": train_eval_metrics["f1"],
        })

    if test_eval_dataloader is not None:
        test_features, test_labels = extract_encoder_features(
            model,
            test_eval_dataloader,
            device,
        )
        test_metrics = evaluate_logistic_regression_classifier(
            classifier,
            test_features,
            test_labels,
            "test",
        )
        record["test"] = test_metrics
        log_payload.update({
            f"{log_prefix}/test_loss": test_metrics["loss"],
            f"{log_prefix}/test_accuracy": test_metrics["accuracy"],
            f"{log_prefix}/test_f1": test_metrics["f1"],
        })

    _log_wandb(log_payload, wandb_run)
    return classifier, record


def train_end2end(
    model: SimCLRModel,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    num_epochs: int = 50,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    eval_dataloader: torch.utils.data.DataLoader = None,
    wandb_run = None,
    log_prefix: str = "end2end",
    checkpoint_dir = None,
    save_interval: int = 0,
):
    model.to(device)
    history = []
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    loss_fn = nn.BCEWithLogitsLoss()

    for epoch in range(num_epochs):
        start_time = time.perf_counter()
        model.train()
        total_loss = 0.0
        num_samples = 0

        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            num_samples += images.size(0)

            outputs = model(images, mode='classification').squeeze(-1)
            loss = loss_fn(outputs, labels.float())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        epoch_time = time.perf_counter() - start_time
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

        epoch_record = {
            "epoch": epoch + 1,
            "loss": avg_loss,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "epoch_time_sec": epoch_time,
            "samples_per_sec": num_samples / epoch_time,
        }
        log_payload = {
            "epoch": epoch_record["epoch"],
            f"{log_prefix}/loss": epoch_record["loss"],
            f"{log_prefix}/learning_rate": epoch_record["learning_rate"],
            f"{log_prefix}/epoch_time_sec": epoch_record["epoch_time_sec"],
            f"{log_prefix}/samples_per_sec": epoch_record["samples_per_sec"],
        }
        if eval_dataloader is not None:
            metrics = evaluate_classifier(model, eval_dataloader, device, "eval")
            epoch_record["eval"] = metrics
            log_payload.update({
                f"{log_prefix}/eval_accuracy": metrics["accuracy"],
                f"{log_prefix}/eval_f1": metrics["f1"],
                f"{log_prefix}/eval_loss": metrics["loss"],
            })
        history.append(epoch_record)
        _log_wandb(log_payload, wandb_run)
        if checkpoint_dir is not None and save_interval > 0 and (epoch + 1) % save_interval == 0:
            _save_training_state(
                checkpoint_dir / f"{log_prefix}_epoch{epoch+1}.pth",
                model,
                optimizer,
                epoch + 1,
                history=history,
            )

    return history


@torch.inference_mode()
def evaluate_classifier(model: SimCLRModel, dataloader: torch.utils.data.DataLoader, 
                        device: torch.device, mode='eval'):
    model.to(device)
    model.eval()
    
    all_preds = []
    all_labels = []
    total_loss = 0.0
    
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        
        outputs = model(images, mode='classification').squeeze(-1)
        loss = F.binary_cross_entropy_with_logits(outputs, labels.float())
        preds = (outputs > 0).float()
        
        total_loss += loss.item()
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    avg_loss = total_loss / len(dataloader)
    
    print(
        f"{mode.capitalize()}: Loss: {avg_loss:.4f}, "
        f"Accuracy: {accuracy:.4f}, F1-Score: {f1:.4f}"
    )
    return {
        "loss": avg_loss,
        "accuracy": accuracy,
        "f1": f1,
    }
