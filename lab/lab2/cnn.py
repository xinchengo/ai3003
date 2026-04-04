import copy
import time

import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
from tqdm.auto import tqdm, trange


class CNN(nn.Module):
    """
    A simple configurable CNN for LAB2 of
    *Fundamentals of Deep Learning*

    The model outputs raw logits, it should be used with
    `nn.CrossEntropyLoss` directly
    """

    def __init__(
        self,
        input_size: tuple[int, int] = (28, 28),
        input_channels: int = 1,
        output_size: int = 10,
        hidden_layers: int = 3,
        hidden_channels: list[int] | None = None,
        apply_pooling: list[bool] | None = None,
        kernel_size: int = 3,
        pooling_strategy: str = "max",
        use_batch_norm: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()

        assert hidden_layers > 0, "hidden_layers must be > 0"
        assert hidden_channels is None or len(hidden_channels) == hidden_layers, \
            "hidden_channels length must match hidden_layers"
        assert apply_pooling is None or len(apply_pooling) == hidden_layers, \
            "apply_pooling length must match hidden_layers"
        assert kernel_size > 0 and kernel_size % 2 == 1, "kernel_size must be a positive odd integer"
        assert pooling_strategy in ("max", "avg"), "pooling_strategy must be 'max' or 'avg'"

        if hidden_channels is None:
            hidden_channels = [32] * hidden_layers
        if apply_pooling is None:
            apply_pooling = [True] * hidden_layers

        self.input_size = input_size
        self.input_channels = input_channels
        self.output_size = output_size
        self.hidden_layers = hidden_layers
        self.hidden_channels = hidden_channels
        self.apply_pooling = apply_pooling
        self.kernel_size = kernel_size
        self.pooling_strategy = pooling_strategy
        self.use_batch_norm = use_batch_norm
        self.dropout = dropout

        pool_cls = nn.MaxPool2d if pooling_strategy == "max" else nn.AvgPool2d

        feature_layers = []
        in_channels = input_channels
        padding = kernel_size // 2
        height, width = input_size

        for out_channels, should_pool in zip(hidden_channels, apply_pooling):
            feature_layers.append(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    padding=padding,
                )
            )
            if use_batch_norm:
                feature_layers.append(nn.BatchNorm2d(out_channels))
            feature_layers.append(nn.ReLU())
            if should_pool:
                feature_layers.append(pool_cls(kernel_size=2, stride=2))
                height //= 2
                width //= 2
            if dropout > 0:
                feature_layers.append(nn.Dropout2d(p=dropout))
            in_channels = out_channels

        self.features = nn.Sequential(*feature_layers)

        flattened_dim = in_channels * height * width
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened_dim, output_size),
        )

        self.model = nn.Sequential(self.features, self.classifier)
        self.reset_parameters()

    def reset_parameters(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x):
        return self.model(x)


def _move_batch_to_device(X_batch: torch.Tensor, y_batch: torch.Tensor, device: torch.device):
    return X_batch.to(device), y_batch.to(device)


def evaluate_model(
    model: nn.Module,
    criterion: nn.Module,
    dataloader: DataLoader,
    device: str | torch.device | None = None,
):
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = model.to(device)
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch, y_batch = _move_batch_to_device(X_batch, y_batch, device)
            logits = model(X_batch)
            loss = criterion(logits, y_batch)

            total_loss += loss.item() * X_batch.size(0)
            total_correct += (logits.argmax(dim=1) == y_batch).sum().item()
            total_samples += X_batch.size(0)

    return {
        "loss": total_loss / total_samples,
        "accuracy": total_correct / total_samples,
    }


def train_model(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    train_loader: DataLoader,
    val_loader: DataLoader | None = None,
    criterion: nn.Module = nn.CrossEntropyLoss(),
    num_epochs: int = 20,
    device: str | torch.device | None = None,
    show_progress: bool = True,
    return_details: bool = False,
):
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = model.to(device)

    history = {
        "train_losses": [],
        "train_accuracies": [],
        "val_losses": [],
        "val_accuracies": [],
        "epoch_times": [],
        "cumulative_times": [],
        "best_epoch": None,
        "best_train_loss": None,
        "best_val_loss": None,
        "best_val_accuracy": None,
        "best_state_dict": None,
    }

    total_elapsed = 0.0
    best_val_accuracy = float("-inf")

    epoch_bar = (
        trange(num_epochs, desc="Epoch", dynamic_ncols=True, mininterval=0.5)
        if show_progress
        else range(num_epochs)
    )

    for epoch in epoch_bar:
        epoch_start = time.perf_counter()
        model.train()

        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        batch_bar = (
            tqdm(
                train_loader,
                desc=f"Batch {epoch + 1}/{num_epochs}",
                leave=False,
                dynamic_ncols=True,
                mininterval=0.5,
            )
            if show_progress
            else train_loader
        )

        for X_batch, y_batch in batch_bar:
            X_batch, y_batch = _move_batch_to_device(X_batch, y_batch, device)

            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * X_batch.size(0)
            total_correct += (logits.argmax(dim=1) == y_batch).sum().item()
            total_samples += X_batch.size(0)

        train_loss = total_loss / total_samples
        train_accuracy = total_correct / total_samples
        history["train_losses"].append(train_loss)
        history["train_accuracies"].append(train_accuracy)

        val_metrics = None
        if val_loader is not None:
            val_metrics = evaluate_model(
                model=model,
                criterion=criterion,
                dataloader=val_loader,
                device=device,
            )
            history["val_losses"].append(val_metrics["loss"])
            history["val_accuracies"].append(val_metrics["accuracy"])

            if val_metrics["accuracy"] > best_val_accuracy:
                best_val_accuracy = val_metrics["accuracy"]
                history["best_epoch"] = epoch + 1
                history["best_train_loss"] = train_loss
                history["best_val_loss"] = val_metrics["loss"]
                history["best_val_accuracy"] = val_metrics["accuracy"]
                history["best_state_dict"] = copy.deepcopy(model.state_dict())

        epoch_elapsed = time.perf_counter() - epoch_start
        total_elapsed += epoch_elapsed
        history["epoch_times"].append(epoch_elapsed)
        history["cumulative_times"].append(total_elapsed)

        if show_progress:
            postfix = {
                "train_loss": f"{train_loss:.4f}",
                "train_acc": f"{train_accuracy:.4f}",
            }
            if val_metrics is not None:
                postfix["val_loss"] = f"{val_metrics['loss']:.4f}"
                postfix["val_acc"] = f"{val_metrics['accuracy']:.4f}"
            epoch_bar.set_postfix(**postfix)

    if history["best_state_dict"] is not None:
        model.load_state_dict(history["best_state_dict"])

    return history


def predict(
    model: nn.Module,
    dataloader: DataLoader,
    device: str | torch.device | None = None,
):
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = model.to(device)
    model.eval()

    predictions = []

    with torch.no_grad():
        for batch in dataloader:
            X_batch = batch[0].to(device)
            logits = model(X_batch)
            predictions.append(logits.argmax(dim=1).cpu())

    return torch.cat(predictions, dim=0)


def evaluate_test_set(
    model: nn.Module,
    test_loader: DataLoader,
    criterion: nn.Module | None = None,
    device: str | torch.device | None = None,
):
    if criterion is None:
        criterion = nn.CrossEntropyLoss()

    metrics = evaluate_model(
        model=model,
        criterion=criterion,
        dataloader=test_loader,
        device=device,
    )

    y_pred = predict(model=model, dataloader=test_loader, device=device)
    y_true = torch.cat([y_batch.cpu() for _, y_batch in test_loader], dim=0)
    cm = confusion_matrix(y_true.numpy(), y_pred.numpy())

    return {
        "test_loss": metrics["loss"],
        "test_accuracy": metrics["accuracy"],
        "predictions": y_pred,
        "confusion_matrix": cm,
    }
