from datetime import datetime
from pathlib import Path

import torch

from dataloader import (
    get_cifar10_classification_dataloader,
    get_cifar10_simclr_dataloader,
)
from model import SimCLRModel
from trainer import (
    evaluate_classifier,
    evaluate_logistic_regression_classifier,
    extract_encoder_features,
    fit_logistic_regression_classifier,
    nt_logistic_loss,
    nt_xent_loss,
    pretrain_simclr,
    train_end2end,
    triplet_loss,
)
from train_utils import (
    as_bool,
    as_float,
    as_int,
    count_tensor_dataset,
    device as get_device,
    load_linear_probe,
    load_model_state,
    model_config,
    read_json_if_exists,
    save_linear_probe,
    save_training_state,
    update_wandb_summary,
    write_json,
)


def _wandb_init(use_wandb, use_swanlab=False, swanlab_mode="cloud", **kwargs):
    if not use_wandb and not use_swanlab:
        return None, None
    if use_swanlab:
        try:
            import swanlab
        except ImportError as exc:
            raise ImportError("Install SwanLab with `pip install swanlab` to use --use-swanlab true.") from exc

        swanlab.sync_wandb(mode=swanlab_mode, wandb_run=use_wandb)
    try:
        import wandb
    except ImportError as exc:
        raise ImportError("Install Weights & Biases with `pip install wandb`; SwanLab sync_wandb also uses the wandb API.") from exc

    return wandb, wandb.init(**kwargs)


def _check_paths(*paths):
    for path in paths:
        if not Path(path).exists():
            raise FileNotFoundError(f"Missing dataset file: {path}")


def _build_simclr_model(encoder, head_hidden_dim=128, head_use_batchnorm=True, projection_dim=64):
    return SimCLRModel(
        encoder=encoder,
        head_hidden_dim=head_hidden_dim,
        head_use_batchnorm=head_use_batchnorm,
        projection_dim=projection_dim,
    )


def _build_contrastive_loss(loss_name, temperature=0.5, triplet_margin=1.0):
    loss_name = str(loss_name or "nt_xent").strip().lower().replace("-", "_")
    aliases = {
        "ntxent": "nt_xent",
        "info_nce": "nt_xent",
        "infonce": "nt_xent",
        "logistic": "nt_logistic",
        "contrastive": "nt_logistic",
        "logistic_contrastive": "nt_logistic",
    }
    loss_name = aliases.get(loss_name, loss_name)

    if loss_name == "nt_xent":
        return loss_name, lambda z1, z2: nt_xent_loss(z1, z2, temperature=temperature)
    if loss_name == "nt_logistic":
        return loss_name, lambda z1, z2: nt_logistic_loss(z1, z2, temperature=temperature)
    if loss_name == "triplet":
        return loss_name, lambda z1, z2: triplet_loss(z1, z2, margin=triplet_margin)
    raise ValueError(
        f"Unsupported loss_name: {loss_name}. "
        "Use one of: nt_xent, nt_logistic, triplet."
    )


def _head_config_from_checkpoint(checkpoint_dir, checkpoint=None):
    config = read_json_if_exists(Path(checkpoint_dir) / "config.json", {})
    if not config and isinstance(checkpoint, dict):
        config = checkpoint.get("config", {}) or {}
    return {
        "head_hidden_dim": config.get("head_hidden_dim", 128),
        "head_use_batchnorm": config.get("head_use_batchnorm", True),
        "projection_dim": config.get("projection_dim", 64),
    }


def run_simclr_train(
    ratio="r10",
    encoder="resnet18",
    pretrain_epochs=100,
    pretrain_batch_size=1024,
    probe_batch_size=256,
    pretrain_lr=1e-3,
    temperature=0.5,
    loss_name="nt_xent",
    triplet_margin=1.0,
    head_hidden_dim=128,
    head_use_batchnorm=True,
    projection_dim=64,
    use_blur=False,
    run_name="simclr",
    save_interval=0,
    mixed_precision=False,
    resume_checkpoint="",
    wandb_run_id="",
    wandb_resume="allow",
    data_root=None,
    results_root=None,
    use_wandb=True,
    use_swanlab=False,
    swanlab_mode="cloud",
    num_workers=8,
    prefetch_factor=4,
    backend="local",
):
    pretrain_epochs = as_int(pretrain_epochs)
    pretrain_batch_size = as_int(pretrain_batch_size)
    probe_batch_size = as_int(probe_batch_size)
    pretrain_lr = as_float(pretrain_lr)
    temperature = as_float(temperature)
    triplet_margin = as_float(triplet_margin)
    head_hidden_dim = as_int(head_hidden_dim)
    head_use_batchnorm = as_bool(head_use_batchnorm)
    projection_dim = as_int(projection_dim)
    use_blur = as_bool(use_blur)
    save_interval = as_int(save_interval)
    mixed_precision = as_bool(mixed_precision)
    resume_checkpoint = str(resume_checkpoint or "")
    wandb_run_id = str(wandb_run_id or "")
    wandb_resume = str(wandb_resume or "allow")
    use_wandb = as_bool(use_wandb)
    use_swanlab = as_bool(use_swanlab)
    swanlab_mode = str(swanlab_mode or "cloud")
    num_workers = as_int(num_workers)
    prefetch_factor = as_int(prefetch_factor)

    data_root = Path(data_root).expanduser().resolve()
    results_root = Path(results_root).expanduser().resolve()
    pretrain_path = data_root / ratio / "pretrain.pth"
    finetune_path = data_root / ratio / "finetune.pth"
    test_path = data_root / ratio / "test.pth"
    _check_paths(pretrain_path, finetune_path, test_path)

    if resume_checkpoint:
        checkpoint_dir = Path(resume_checkpoint).expanduser().resolve().parent
        old_config = read_json_if_exists(checkpoint_dir / "config.json", {})
        loss_name = old_config.get("loss_name", loss_name)
        triplet_margin = old_config.get("triplet_margin", triplet_margin)
        head_hidden_dim = old_config.get("head_hidden_dim", head_hidden_dim)
        head_use_batchnorm = old_config.get("head_use_batchnorm", head_use_batchnorm)
        projection_dim = old_config.get("projection_dim", projection_dim)
        if not wandb_run_id:
            wandb_run_id = old_config.get("wandb_run_id", "")
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_dir = results_root / "checkpoints" / f"{ratio}-{encoder}-{run_name}" / timestamp
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    triplet_margin = as_float(triplet_margin)
    head_hidden_dim = as_int(head_hidden_dim)
    head_use_batchnorm = as_bool(head_use_batchnorm)
    projection_dim = as_int(projection_dim)

    dev = get_device()
    print(f"Using device: {dev}")
    model = _build_simclr_model(
        encoder=encoder,
        head_hidden_dim=head_hidden_dim,
        head_use_batchnorm=head_use_batchnorm,
        projection_dim=projection_dim,
    ).to(dev)
    if resume_checkpoint:
        print(f"Loading checkpoint from {resume_checkpoint}")
        resume_state = torch.load(resume_checkpoint, map_location=dev)
        load_model_state(model, resume_state)
    else:
        resume_state = None

    existing_pretrain_history = read_json_if_exists(checkpoint_dir / "pretrain_history.json", [])
    start_epoch = 0
    if existing_pretrain_history:
        start_epoch = int(existing_pretrain_history[-1].get("epoch", len(existing_pretrain_history)))

    config = {
        "mode": "simclr",
        "backend": backend,
        "ratio": ratio,
        "encoder": encoder,
        "pretrain_epochs": pretrain_epochs,
        "pretrain_batch_size": pretrain_batch_size,
        "probe_batch_size": probe_batch_size,
        "pretrain_lr": pretrain_lr,
        "temperature": temperature,
        "loss_name": loss_name,
        "triplet_margin": triplet_margin,
        "use_blur": use_blur,
        "pretrain_optimizer": "AdamW",
        "linear_probe": "LogisticRegression",
        "run_name": run_name,
        "save_interval": save_interval,
        "mixed_precision": mixed_precision,
        "resume_checkpoint": resume_checkpoint,
        "wandb_resume": wandb_resume,
        "data_root": str(data_root),
        "results_root": str(results_root),
        "use_wandb": use_wandb,
        "use_swanlab": use_swanlab,
        "swanlab_mode": swanlab_mode,
        "num_workers": num_workers,
        "prefetch_factor": prefetch_factor,
        "device": str(dev),
        "checkpoint_dir": str(checkpoint_dir),
        "pretrain_path": str(pretrain_path),
        "finetune_path": str(finetune_path),
        "test_path": str(test_path),
        "pretrain_samples": count_tensor_dataset(pretrain_path),
        "finetune_samples": count_tensor_dataset(finetune_path),
        "test_samples": count_tensor_dataset(test_path),
        "resume_from_checkpoint": resume_checkpoint,
        "resume_start_epoch": start_epoch,
        "resume_additional_pretrain_epochs": pretrain_epochs if resume_checkpoint else 0,
        "wandb_run_id": wandb_run_id,
        **model_config(model),
    }
    write_json(checkpoint_dir / "config.json", config)

    wandb_kwargs = {
        "project": "ai3003-lab4",
        "config": config,
        "name": f"{ratio}-{encoder}-{run_name}",
    }
    if wandb_run_id:
        wandb_kwargs.update({"id": wandb_run_id, "resume": wandb_resume})
    wandb, wandb_run = _wandb_init(
        use_wandb,
        use_swanlab=use_swanlab,
        swanlab_mode=swanlab_mode,
        **wandb_kwargs,
    )
    if wandb_run is not None:
        config["wandb_run_id"] = wandb_run.id
        write_json(checkpoint_dir / "config.json", config)
    update_wandb_summary(wandb_run, {
        "checkpoint_dir": str(checkpoint_dir),
        "pretrain_samples": config["pretrain_samples"],
        "finetune_samples": config["finetune_samples"],
        "test_samples": config["test_samples"],
    })

    pretrain_loader = get_cifar10_simclr_dataloader(
        str(pretrain_path),
        batch_size=pretrain_batch_size,
        use_blur=use_blur,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
    )
    finetune_loader = get_cifar10_classification_dataloader(
        str(finetune_path),
        batch_size=probe_batch_size,
        train=False,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
    )
    test_loader = get_cifar10_classification_dataloader(
        str(test_path),
        batch_size=probe_batch_size,
        train=False,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
    )

    try:
        optimizer = torch.optim.AdamW(model.parameters(), lr=pretrain_lr)
        if (
            isinstance(resume_state, dict)
            and resume_state.get("optimizer_state_dict") is not None
        ):
            optimizer.load_state_dict(resume_state["optimizer_state_dict"])

        resolved_loss_name, loss_fn = _build_contrastive_loss(
            loss_name,
            temperature=temperature,
            triplet_margin=triplet_margin,
        )
        config["loss_name"] = resolved_loss_name
        write_json(checkpoint_dir / "config.json", config)
        pretrain_history = pretrain_simclr(
            model,
            pretrain_loader,
            optimizer,
            dev,
            loss_fn=loss_fn,
            num_epochs=pretrain_epochs,
            wandb_run=wandb_run,
            checkpoint_dir=checkpoint_dir,
            save_interval=save_interval,
            mixed_precision=mixed_precision,
            start_epoch=start_epoch,
            config=config,
        )
        if existing_pretrain_history:
            pretrain_history = existing_pretrain_history + pretrain_history

        pretrain_checkpoint = checkpoint_dir / "pretrain.pth"
        last_epoch = pretrain_history[-1]["epoch"] if pretrain_history else start_epoch
        save_training_state(pretrain_checkpoint, model, optimizer, last_epoch, pretrain_history, config)
        print(f"Saved pretrain checkpoint to {pretrain_checkpoint}")
        write_json(checkpoint_dir / "pretrain_history.json", pretrain_history)
        update_wandb_summary(wandb_run, {"pretrain_checkpoint": str(pretrain_checkpoint)})

        linear_probe, probe_record = fit_logistic_regression_classifier(
            model,
            finetune_loader,
            dev,
            test_eval_dataloader=test_loader,
            wandb_run=wandb_run,
        )
        linear_probe_path = checkpoint_dir / "linear_probe.pkl"
        save_linear_probe(linear_probe_path, linear_probe)
        print(f"Saved linear probe to {linear_probe_path}")

        final_checkpoint = checkpoint_dir / "final.pth"
        save_training_state(
            final_checkpoint,
            model,
            None,
            last_epoch,
            {"pretrain": pretrain_history, "linear_probe": probe_record},
            config,
        )
        print(f"Saved final checkpoint to {final_checkpoint}")
        write_json(checkpoint_dir / "linear_probe.json", probe_record)
        write_json(checkpoint_dir / "training_history.json", {
            "pretrain": pretrain_history,
            "linear_probe": probe_record,
        })

        test_metrics = probe_record["test"]
        if wandb_run is not None:
            wandb_run.log({
                "test/loss": test_metrics["loss"],
                "test/accuracy": test_metrics["accuracy"],
                "test/f1": test_metrics["f1"],
            })

        summary = {
            "pretrain_checkpoint": str(pretrain_checkpoint),
            "final_checkpoint": str(final_checkpoint),
            "linear_probe": str(linear_probe_path),
            "test": test_metrics,
            "config": config,
        }
        write_json(checkpoint_dir / "summary.json", summary)
        update_wandb_summary(wandb_run, {
            "final_checkpoint": str(final_checkpoint),
            "linear_probe": str(linear_probe_path),
            "test_accuracy": test_metrics["accuracy"],
            "test_f1": test_metrics["f1"],
            "test_loss": test_metrics["loss"],
            "summary_path": str(checkpoint_dir / "summary.json"),
        })
        return summary
    finally:
        if wandb is not None and wandb_run is not None:
            wandb.finish()


def run_end2end_train(
    ratio="r10",
    encoder="resnet18",
    num_epochs=100,
    batch_size=256,
    learning_rate=1e-3,
    weight_decay=1e-4,
    run_name="end2end",
    save_interval=0,
    data_root=None,
    results_root=None,
    use_wandb=True,
    use_swanlab=False,
    swanlab_mode="cloud",
    num_workers=8,
    prefetch_factor=4,
    backend="local",
):
    num_epochs = as_int(num_epochs)
    batch_size = as_int(batch_size)
    learning_rate = as_float(learning_rate)
    weight_decay = as_float(weight_decay)
    save_interval = as_int(save_interval)
    use_wandb = as_bool(use_wandb)
    use_swanlab = as_bool(use_swanlab)
    swanlab_mode = str(swanlab_mode or "cloud")
    num_workers = as_int(num_workers)
    prefetch_factor = as_int(prefetch_factor)
    data_root = Path(data_root).expanduser().resolve()
    results_root = Path(results_root).expanduser().resolve()

    finetune_path = data_root / ratio / "finetune.pth"
    test_path = data_root / ratio / "test.pth"
    _check_paths(finetune_path, test_path)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_dir = results_root / "checkpoints" / f"{ratio}-{encoder}-{run_name}" / timestamp
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    dev = get_device()
    print(f"Using device: {dev}")
    model = SimCLRModel(encoder=encoder).to(dev)
    config = {
        "mode": "end2end",
        "backend": backend,
        "ratio": ratio,
        "encoder": encoder,
        "num_epochs": num_epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "optimizer": "AdamW",
        "run_name": run_name,
        "save_interval": save_interval,
        "data_root": str(data_root),
        "results_root": str(results_root),
        "use_wandb": use_wandb,
        "use_swanlab": use_swanlab,
        "swanlab_mode": swanlab_mode,
        "num_workers": num_workers,
        "prefetch_factor": prefetch_factor,
        "device": str(dev),
        "checkpoint_dir": str(checkpoint_dir),
        "finetune_path": str(finetune_path),
        "test_path": str(test_path),
        "finetune_samples": count_tensor_dataset(finetune_path),
        "test_samples": count_tensor_dataset(test_path),
        **model_config(model),
    }
    write_json(checkpoint_dir / "config.json", config)

    wandb, wandb_run = _wandb_init(
        use_wandb,
        use_swanlab=use_swanlab,
        swanlab_mode=swanlab_mode,
        project="ai3003-lab4",
        config=config,
        name=f"{ratio}-{encoder}-{run_name}",
    )
    if wandb_run is not None:
        config["wandb_run_id"] = wandb_run.id
        write_json(checkpoint_dir / "config.json", config)
    update_wandb_summary(wandb_run, {
        "checkpoint_dir": str(checkpoint_dir),
        "finetune_samples": config["finetune_samples"],
        "test_samples": config["test_samples"],
    })

    train_loader = get_cifar10_classification_dataloader(
        str(finetune_path),
        batch_size=batch_size,
        train=True,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
    )
    test_loader = get_cifar10_classification_dataloader(
        str(test_path),
        batch_size=batch_size,
        train=False,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
    )

    try:
        history = train_end2end(
            model,
            train_loader,
            dev,
            num_epochs=num_epochs,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            eval_dataloader=test_loader,
            wandb_run=wandb_run,
            checkpoint_dir=checkpoint_dir,
            save_interval=save_interval,
        )

        final_checkpoint = checkpoint_dir / "final.pth"
        save_training_state(
            final_checkpoint,
            model,
            None,
            history[-1]["epoch"] if history else 0,
            history,
            config,
        )
        print(f"Saved final checkpoint to {final_checkpoint}")
        write_json(checkpoint_dir / "training_history.json", history)

        test_metrics = evaluate_classifier(model, test_loader, dev, "test")
        if wandb_run is not None:
            wandb_run.log({
                "test/loss": test_metrics["loss"],
                "test/accuracy": test_metrics["accuracy"],
                "test/f1": test_metrics["f1"],
            })

        summary = {
            "final_checkpoint": str(final_checkpoint),
            "test": test_metrics,
            "config": config,
        }
        write_json(checkpoint_dir / "summary.json", summary)
        update_wandb_summary(wandb_run, {
            "final_checkpoint": str(final_checkpoint),
            "test_accuracy": test_metrics["accuracy"],
            "test_f1": test_metrics["f1"],
            "test_loss": test_metrics["loss"],
            "summary_path": str(checkpoint_dir / "summary.json"),
        })
        return summary
    finally:
        if wandb is not None and wandb_run is not None:
            wandb.finish()


def run_resume_classifier(
    checkpoint_dir,
    ratio="",
    encoder="",
    probe_batch_size=0,
    data_root=None,
    use_wandb=True,
    use_swanlab=False,
    swanlab_mode="cloud",
    num_workers=8,
    prefetch_factor=4,
    backend="local",
):
    use_wandb = as_bool(use_wandb)
    use_swanlab = as_bool(use_swanlab)
    swanlab_mode = str(swanlab_mode or "cloud")
    num_workers = as_int(num_workers)
    prefetch_factor = as_int(prefetch_factor)
    data_root = Path(data_root).expanduser().resolve()
    checkpoint_dir = Path(checkpoint_dir).expanduser().resolve()
    config_path = checkpoint_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config.json in {checkpoint_dir}")

    config = read_json_if_exists(config_path, {})
    ratio = ratio or config["ratio"]
    encoder = encoder or config["encoder"]
    probe_batch_size = as_int(probe_batch_size or config.get("probe_batch_size", 256))

    pretrain_checkpoint = checkpoint_dir / "pretrain.pth"
    if not pretrain_checkpoint.exists():
        raise FileNotFoundError(f"Missing pretrain checkpoint: {pretrain_checkpoint}")

    finetune_path = data_root / ratio / "finetune.pth"
    test_path = data_root / ratio / "test.pth"
    _check_paths(finetune_path, test_path)

    config.update({
        "backend": backend,
        "resume_classifier": True,
        "resume_from": str(pretrain_checkpoint),
        "linear_probe": "LogisticRegression",
        "probe_batch_size": probe_batch_size,
        "finetune_path": str(finetune_path),
        "test_path": str(test_path),
        "data_root": str(data_root),
        "use_wandb": use_wandb,
        "use_swanlab": use_swanlab,
        "swanlab_mode": swanlab_mode,
        "num_workers": num_workers,
        "prefetch_factor": prefetch_factor,
    })

    dev = get_device()
    print(f"Using device: {dev}")
    checkpoint = torch.load(pretrain_checkpoint, map_location=dev)
    model = _build_simclr_model(
        encoder=encoder,
        **_head_config_from_checkpoint(checkpoint_dir, checkpoint),
    ).to(dev)
    load_model_state(model, checkpoint)

    wandb, wandb_run = _wandb_init(
        use_wandb,
        use_swanlab=use_swanlab,
        swanlab_mode=swanlab_mode,
        project="ai3003-lab4",
        config=config,
        name=f"{ratio}-{encoder}-{config.get('run_name', 'simclr')}-resume-classifier",
    )
    if wandb_run is not None:
        config["wandb_run_id"] = wandb_run.id
        write_json(checkpoint_dir / "config.json", config)
    update_wandb_summary(wandb_run, {
        "checkpoint_dir": str(checkpoint_dir),
        "pretrain_checkpoint": str(pretrain_checkpoint),
    })

    finetune_loader = get_cifar10_classification_dataloader(
        str(finetune_path),
        batch_size=probe_batch_size,
        train=False,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
    )
    test_loader = get_cifar10_classification_dataloader(
        str(test_path),
        batch_size=probe_batch_size,
        train=False,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
    )

    try:
        linear_probe, probe_record = fit_logistic_regression_classifier(
            model,
            finetune_loader,
            dev,
            test_eval_dataloader=test_loader,
            wandb_run=wandb_run,
        )
        linear_probe_path = checkpoint_dir / "linear_probe.pkl"
        save_linear_probe(linear_probe_path, linear_probe)
        print(f"Saved linear probe to {linear_probe_path}")

        final_checkpoint = checkpoint_dir / "final.pth"
        save_training_state(
            final_checkpoint,
            model,
            None,
            checkpoint.get("epoch", 0) if isinstance(checkpoint, dict) else 0,
            {"linear_probe": probe_record},
            config,
        )
        print(f"Saved final checkpoint to {final_checkpoint}")
        write_json(checkpoint_dir / "linear_probe.json", probe_record)

        pretrain_history = read_json_if_exists(checkpoint_dir / "pretrain_history.json", [])
        write_json(checkpoint_dir / "training_history.json", {
            "pretrain": pretrain_history,
            "linear_probe": probe_record,
        })

        test_metrics = probe_record["test"]
        if wandb_run is not None:
            wandb_run.log({
                "test/loss": test_metrics["loss"],
                "test/accuracy": test_metrics["accuracy"],
                "test/f1": test_metrics["f1"],
            })

        summary = {
            "pretrain_checkpoint": str(pretrain_checkpoint),
            "final_checkpoint": str(final_checkpoint),
            "linear_probe": str(linear_probe_path),
            "test": test_metrics,
            "config": config,
        }
        write_json(checkpoint_dir / "summary.json", summary)
        update_wandb_summary(wandb_run, {
            "final_checkpoint": str(final_checkpoint),
            "linear_probe": str(linear_probe_path),
            "test_accuracy": test_metrics["accuracy"],
            "test_f1": test_metrics["f1"],
            "test_loss": test_metrics["loss"],
            "summary_path": str(checkpoint_dir / "summary.json"),
        })
        return summary
    finally:
        if wandb is not None and wandb_run is not None:
            wandb.finish()


def run_eval(
    ratio="r10",
    encoder="resnet18",
    checkpoint_path="",
    batch_size=256,
    data_root=None,
    num_workers=8,
    prefetch_factor=4,
):
    if not checkpoint_path:
        raise ValueError("checkpoint_path is required for eval")

    data_root = Path(data_root).expanduser().resolve()
    batch_size = as_int(batch_size)
    num_workers = as_int(num_workers)
    prefetch_factor = as_int(prefetch_factor)
    dev = get_device()
    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    checkpoint = torch.load(checkpoint_path, map_location=dev)
    model = _build_simclr_model(
        encoder=encoder,
        **_head_config_from_checkpoint(checkpoint_path.parent, checkpoint),
    ).to(dev)
    load_model_state(
        model,
        checkpoint,
        allow_classifier_mismatch=False,
        allow_head_mismatch=True,
    )
    test_loader = get_cifar10_classification_dataloader(
        str(data_root / ratio / "test.pth"),
        batch_size=batch_size,
        train=False,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
    )
    linear_probe_path = checkpoint_path.parent / "linear_probe.pkl"
    if linear_probe_path.exists():
        linear_probe = load_linear_probe(linear_probe_path)
        features, labels = extract_encoder_features(model, test_loader, dev)
        return evaluate_logistic_regression_classifier(linear_probe, features, labels, "test")
    return evaluate_classifier(model, test_loader, dev, "test")
