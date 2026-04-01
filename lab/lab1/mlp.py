import torch
import torch.nn as nn
import numpy as np
import time
from tqdm.auto import tqdm, trange
from sklearn.model_selection import KFold

class Swish(nn.Module):
    "Learnable Swish activation function"
    def __init__(self):
        super().__init__()
        self.beta = nn.Parameter(torch.tensor(1.0))
    
    def forward(self, x):
        return x * torch.sigmoid(self.beta * x)

class MultiLayerPerceptron(nn.Module):
    """
    A simple multi-layer perceptron for HW1 of
    *Fundamentals of Deep Learning*
    """

    def __init__(
            self,
            in_channels : int = 10,
            out_channels : int = 1,
            num_hidden_layers : int = 2,
            hidden_layer_sizes : list = [32, 32],
            activation : str = 'relu',
            init_method : str = 'none',
        ):

        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_hidden_layers = num_hidden_layers
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.init_method = init_method

        assert num_hidden_layers >= 0, "Number of hidden layers must be non-negative"
        assert len(hidden_layer_sizes) == num_hidden_layers, "Length of hidden_layer_sizes must match num_hidden_layers"

        layers = []
        input_size = in_channels

        for hidden_size in hidden_layer_sizes:
            layers.append(nn.Linear(input_size, hidden_size))
            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'tanh':
                layers.append(nn.Tanh())
            elif activation == 'sigmoid':
                layers.append(nn.Sigmoid())
            elif activation == 'leaky_relu':
                layers.append(nn.LeakyReLU())
            elif activation == 'swish':
                layers.append(Swish())
            else:
                raise ValueError(f"Unsupported activation function: {activation}")
            input_size = hidden_size

        layers.append(nn.Linear(input_size, out_channels))
        
        self.model = nn.Sequential(*layers)

        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.model:
            if isinstance(layer, nn.Linear):
                if self.init_method == 'xavier':
                    nn.init.xavier_uniform_(layer.weight, gain=nn.init.calculate_gain(self.activation))
                elif self.init_method == 'kaiming':
                    nn.init.kaiming_uniform_(layer.weight, nonlinearity=self.activation)
                else:
                    layer.reset_parameters()
    
    def forward(self, x):
        return self.model(x)


def collect_hidden_preactivations(
        model: nn.Module,
        X: torch.Tensor,
        batch_size: int = 256,
    ):
    """Collect pre-activation tensors for all hidden linear layers."""

    model.eval()
    dataset = torch.utils.data.TensorDataset(X)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

    linear_layers = [layer for layer in model.model if isinstance(layer, nn.Linear)]
    hidden_linear_count = max(len(linear_layers) - 1, 0)
    preactivation_chunks = [[] for _ in range(hidden_linear_count)]

    with torch.no_grad():
        for (X_batch,) in dataloader:
            h = X_batch
            linear_idx = 0
            for layer in model.model:
                h = layer(h)
                if isinstance(layer, nn.Linear) and linear_idx < hidden_linear_count:
                    preactivation_chunks[linear_idx].append(h.detach().cpu())
                    linear_idx += 1

    return [torch.cat(chunks, dim=0) if chunks else torch.empty(0) for chunks in preactivation_chunks]


def analyze_activation_pathologies(
        model: nn.Module,
        X: torch.Tensor,
        sample_threshold: float = 0.9,
        saturation_threshold: float = 37.0,
        batch_size: int = 256,
    ):
    """Analyze dead neurons (ReLU) or saturated neurons (Sigmoid/Tanh)."""

    preactivations = collect_hidden_preactivations(
        model=model,
        X=X,
        batch_size=batch_size,
    )

    activation = getattr(model, 'activation', None)
    if activation not in {'relu', 'sigmoid', 'tanh'}:
        return {
            'activation': activation,
            'issue_type': None,
            'total_hidden_units': int(sum(p.shape[1] for p in preactivations if p.ndim == 2)),
            'affected_units': 0,
            'affected_ratio': 0.0,
            'sample_threshold': sample_threshold,
            'saturation_threshold': saturation_threshold,
            'layer_breakdown': [],
        }

    issue_type = 'dead_neuron' if activation == 'relu' else 'vanishing_gradient'
    layer_breakdown = []
    total_hidden_units = 0
    affected_units = 0

    for layer_idx, z in enumerate(preactivations, start=1):
        if z.ndim != 2 or z.shape[0] == 0:
            continue

        if activation == 'relu':
            affected_fraction = (z <= 0).float().mean(dim=0)
        else:
            affected_fraction = (z.abs() >= saturation_threshold).float().mean(dim=0)

        affected_mask = affected_fraction >= sample_threshold
        layer_units = int(z.shape[1])
        layer_affected = int(affected_mask.sum().item())

        total_hidden_units += layer_units
        affected_units += layer_affected

        layer_breakdown.append({
            'layer': layer_idx,
            'units': layer_units,
            'affected_units': layer_affected,
            'affected_ratio': layer_affected / layer_units if layer_units else 0.0,
        })

    return {
        'activation': activation,
        'issue_type': issue_type,
        'total_hidden_units': total_hidden_units,
        'affected_units': affected_units,
        'affected_ratio': affected_units / total_hidden_units if total_hidden_units else 0.0,
        'sample_threshold': sample_threshold,
        'saturation_threshold': saturation_threshold,
        'layer_breakdown': layer_breakdown,
    }
    
def train_model(
        model : nn.Module,
        optimizer : torch.optim.Optimizer,
        criterion : nn.Module,
        X_train : torch.Tensor,
        y_train : torch.Tensor,
    X_val : torch.Tensor | None = None,
    y_val : torch.Tensor | None = None,
        num_epochs : int = 100,
        batch_size : int = 32,
    show_progress : bool = True,
    return_details : bool = False,
    ):

    model.train()
    dataset = torch.utils.data.TensorDataset(X_train, y_train)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    losses = []
    val_losses = []
    epoch_times = []
    cumulative_times = []
    total_elapsed = 0.0
    best_epoch = None
    best_train_loss = None
    best_val_loss = None

    if show_progress:
        epoch_bar = trange(
            num_epochs,
            desc="Epoch",
            dynamic_ncols=True,
            mininterval=0.5,
        )
    else:
        epoch_bar = range(num_epochs)

    for epoch in epoch_bar:
        epoch_start = time.perf_counter()
        epoch_loss = 0.0

        if show_progress:
            batch_bar = tqdm(
                dataloader,
                desc=f"Batch {epoch + 1}/{num_epochs}",
                leave=False,
                dynamic_ncols=True,
                mininterval=0.5,
            )
        else:
            batch_bar = dataloader

        for X_batch, y_batch in batch_bar:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * X_batch.size(0)

        epoch_loss /= len(X_train)
        losses.append(epoch_loss)

        val_loss = None
        if X_val is not None and y_val is not None:
            val_loss = evaluate_model(
                model=model,
                criterion=criterion,
                X_eval=X_val,
                y_eval=y_val,
                batch_size=batch_size,
            )
            val_losses.append(val_loss)
            if best_val_loss is None or val_loss < best_val_loss:
                best_epoch = epoch + 1
                best_train_loss = epoch_loss
                best_val_loss = val_loss

        epoch_elapsed = time.perf_counter() - epoch_start
        epoch_times.append(epoch_elapsed)
        total_elapsed += epoch_elapsed
        cumulative_times.append(total_elapsed)

        if show_progress:
            postfix = {"loss": f"{epoch_loss:.4f}"}
            if val_loss is not None:
                postfix["val_loss"] = f"{val_loss:.4f}"
            epoch_bar.set_postfix(**postfix)

    if return_details:
        return {
            'train_losses': losses,
            'val_losses': val_losses,
            'epoch_times': epoch_times,
            'cumulative_times': cumulative_times,
            'best_epoch': best_epoch,
            'best_train_loss': best_train_loss,
            'best_val_loss': best_val_loss,
        }

    return losses


def evaluate_model(
        model : nn.Module,
        criterion : nn.Module,
        X_eval : torch.Tensor,
        y_eval : torch.Tensor,
        batch_size : int = 32,
    ):

    model.eval()
    dataset = torch.utils.data.TensorDataset(X_eval, y_eval)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

    total_loss = 0.0

    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            total_loss += loss.item() * X_batch.size(0)

    return total_loss / len(X_eval)


def k_fold_cross_validate(
        X : torch.Tensor,
        y : torch.Tensor,
        model_factory = MultiLayerPerceptron,
        optimizer_factory = torch.optim.Adam,
        criterion : nn.Module = nn.MSELoss(),
        model_factory_args : tuple = (),
        model_factory_kwargs : dict | None = None,
        optimizer_factory_args : tuple = (),
        optimizer_factory_kwargs : dict | None = None,
        num_folds : int = 5,
        num_epochs : int = 100,
        batch_size : int = 32,
        random_state : int = 42,
        show_progress : bool = True,
    ):

    kf = KFold(n_splits=num_folds, shuffle=True, random_state=random_state)
    fold_results = []
    model_factory_kwargs = model_factory_kwargs or {}
    optimizer_factory_kwargs = optimizer_factory_kwargs or {}

    if show_progress:
        fold_bar = tqdm(
            list(kf.split(X)),
            desc="Fold",
            leave=False,
            dynamic_ncols=True,
            mininterval=0.5,
        )
    else:
        fold_bar = list(kf.split(X))

    for fold_idx, (train_idx, val_idx) in enumerate(fold_bar, start=1):
        model = model_factory(*model_factory_args, **model_factory_kwargs)
        optimizer = optimizer_factory(model.parameters(), *optimizer_factory_args, **optimizer_factory_kwargs)

        X_fold_train = X[train_idx]
        y_fold_train = y[train_idx]
        X_fold_val = X[val_idx]
        y_fold_val = y[val_idx]

        train_details = train_model(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            X_train=X_fold_train,
            y_train=y_fold_train,
            X_val=X_fold_val,
            y_val=y_fold_val,
            num_epochs=num_epochs,
            batch_size=batch_size,
            show_progress=False,
            return_details=True,
        )
        val_loss = train_details['val_losses'][-1]

        fold_results.append({
            'fold': fold_idx,
            'train_losses': train_details['train_losses'],
            'val_losses': train_details['val_losses'],
            'val_loss': val_loss,
            'best_epoch': train_details['best_epoch'],
            'best_train_loss': train_details['best_train_loss'],
            'best_val_loss': train_details['best_val_loss'],
            'epoch_times': train_details['epoch_times'],
            'cumulative_times': train_details['cumulative_times'],
        })

        if show_progress:
            fold_bar.set_postfix(val_loss=f"{val_loss:.4f}")

    return fold_results