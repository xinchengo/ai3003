import json
import pickle
from pathlib import Path


def parse_values(value):
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    return list(value)


def as_int(value):
    return int(value)


def as_float(value):
    return float(value)


def as_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "y", "on")
    return bool(value)


def device():
    import torch

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def count_tensor_dataset(path):
    import torch

    data_obj = torch.load(path, map_location="cpu")
    if isinstance(data_obj, tuple):
        return int(data_obj[0].size(0))
    return int(data_obj.size(0))


def model_config(model):
    return {
        "feature_dim": model.feature_dim,
        "head_hidden_dim": model.projection_head[0].out_features,
        "projection_dim": model.projection_head[-1].out_features,
        "head_use_batchnorm": any(
            layer.__class__.__name__ == "BatchNorm1d"
            for layer in model.projection_head
        ),
    }


def write_json(path, payload):
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def update_wandb_summary(wandb_run, payload):
    if wandb_run is None:
        return
    for key, value in payload.items():
        wandb_run.summary[key] = value


def read_json_if_exists(path, default):
    path = Path(path)
    if path.exists():
        with open(path, "r") as f:
            return json.load(f)
    return default


def extract_model_state(checkpoint):
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"]
    return checkpoint


def load_model_state(model, checkpoint, allow_classifier_mismatch=True):
    state_dict = extract_model_state(checkpoint)
    try:
        model.load_state_dict(state_dict)
    except RuntimeError as exc:
        if not allow_classifier_mismatch or "classifier" not in str(exc):
            raise
        state_dict = {
            key: value
            for key, value in state_dict.items()
            if not key.startswith("classifier.")
        }
        model.load_state_dict(state_dict, strict=False)


def save_training_state(path, model, optimizer, epoch, history, config):
    import torch

    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "epoch": epoch,
        "history": history,
        "config": config,
    }, path)


def save_linear_probe(path, classifier):
    with open(path, "wb") as f:
        pickle.dump(classifier, f)


def load_linear_probe(path):
    with open(path, "rb") as f:
        return pickle.load(f)
