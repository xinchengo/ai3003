import json
import os
from pathlib import Path

try:
    import modal
except ImportError:
    modal = None

from train_core import (
    run_end2end_train,
    run_eval,
    run_resume_classifier,
    run_simclr_train,
)
from train_utils import (
    as_bool,
    as_float,
    as_int,
    parse_values,
)


LAB_DIR = Path(__file__).parent
DEFAULT_DATA_ROOT = LAB_DIR / "data"
DEFAULT_RESULTS_ROOT = LAB_DIR / "results"


if modal is not None:
    app = modal.App("ai3003-lab4")
    data = modal.Volume.from_name("ai3003-lab4-data", create_if_missing=True)
    results = modal.Volume.from_name("ai3003-lab4-results", create_if_missing=True)
    image = (
        modal.Image.debian_slim()
        .pip_install("torch", "torchvision", "numpy", "scikit-learn", "wandb", "kagglehub")
        .workdir("/root")
        .add_local_file(LAB_DIR / "trainer.py", remote_path="/root/trainer.py")
        .add_local_file(LAB_DIR / "model.py", remote_path="/root/model.py")
        .add_local_file(LAB_DIR / "dataloader.py", remote_path="/root/dataloader.py")
        .add_local_file(LAB_DIR / "train_core.py", remote_path="/root/train_core.py")
        .add_local_file(LAB_DIR / "train_utils.py", remote_path="/root/train_utils.py")
    )
else:
    app = data = results = image = None


def _modal_function(**kwargs):
    if app is None:
        return lambda fn: fn
    return app.function(**kwargs)


def _local_entrypoint():
    if app is None:
        return lambda fn: fn
    return app.local_entrypoint()


def _require_modal():
    if modal is None:
        raise RuntimeError("Modal is not installed. Use --backend local or install modal.")


def _print(payload):
    print(json.dumps(payload, indent=2))


@_modal_function(
    image=image,
    gpu="L4",
    timeout=60 * 60 * 12,
    volumes={"/root/data": data, "/root/results": results} if modal is not None else {},
    secrets=[modal.Secret.from_name("wandb-secret")] if modal is not None else [],
)
def train(
    ratio="r10",
    encoder="resnet18",
    pretrain_epochs=100,
    pretrain_batch_size=1024,
    probe_batch_size=256,
    pretrain_lr=1e-3,
    temperature=0.5,
    use_blur=False,
    run_name="simclr",
    save_interval=0,
    mixed_precision=False,
    resume_checkpoint="",
    wandb_run_id="",
    wandb_resume="allow",
):
    os.chdir("/root")
    summary = run_simclr_train(
        ratio=ratio,
        encoder=encoder,
        pretrain_epochs=pretrain_epochs,
        pretrain_batch_size=pretrain_batch_size,
        probe_batch_size=probe_batch_size,
        pretrain_lr=pretrain_lr,
        temperature=temperature,
        use_blur=use_blur,
        run_name=run_name,
        save_interval=save_interval,
        mixed_precision=mixed_precision,
        resume_checkpoint=resume_checkpoint,
        wandb_run_id=wandb_run_id,
        wandb_resume=wandb_resume,
        data_root="/root/data",
        results_root="/root/results",
        use_wandb=True,
        backend="modal",
    )
    results.commit()
    return summary


@_modal_function(
    image=image,
    gpu="A100",
    timeout=60 * 60 * 6,
    volumes={"/root/data": data, "/root/results": results} if modal is not None else {},
    secrets=[modal.Secret.from_name("wandb-secret")] if modal is not None else [],
)
def train_baseline(
    ratio="r10",
    encoder="resnet18",
    num_epochs=100,
    batch_size=256,
    learning_rate=1e-3,
    weight_decay=1e-4,
    run_name="end2end",
    save_interval=0,
):
    os.chdir("/root")
    summary = run_end2end_train(
        ratio=ratio,
        encoder=encoder,
        num_epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        run_name=run_name,
        save_interval=save_interval,
        data_root="/root/data",
        results_root="/root/results",
        use_wandb=True,
        backend="modal",
    )
    results.commit()
    return summary


@_modal_function(
    image=image,
    gpu="L4",
    timeout=60 * 60 * 4,
    volumes={"/root/data": data, "/root/results": results} if modal is not None else {},
    secrets=[modal.Secret.from_name("wandb-secret")] if modal is not None else [],
)
def resume_classifier(checkpoint_dir, ratio="", encoder="", probe_batch_size=0):
    os.chdir("/root")
    summary = run_resume_classifier(
        checkpoint_dir=checkpoint_dir,
        ratio=ratio,
        encoder=encoder,
        probe_batch_size=probe_batch_size,
        data_root="/root/data",
        use_wandb=True,
        backend="modal",
    )
    results.commit()
    return summary


@_modal_function(
    image=image,
    gpu="L4",
    timeout=60 * 60 * 2,
    volumes={"/root/data": data, "/root/results": results} if modal is not None else {},
)
def eval(ratio="r10", encoder="resnet18", checkpoint_path="", batch_size=256):
    os.chdir("/root")
    return run_eval(
        ratio=ratio,
        encoder=encoder,
        checkpoint_path=checkpoint_path,
        batch_size=batch_size,
        data_root="/root/data",
    )


@_local_entrypoint()
def main(
    mode="simclr",
    backend="modal",
    ratio="r10",
    encoder="resnet18",
    checkpoint_path="",
    checkpoint_dir="",
    pretrain_epochs=100,
    pretrain_batch_size=1024,
    probe_batch_size=256,
    pretrain_lr=1e-3,
    end2end_epochs=100,
    end2end_batch_size=256,
    end2end_lr=1e-3,
    weight_decay=1e-4,
    temperature=0.5,
    use_blur=False,
    run_name="simclr",
    save_interval=0,
    mixed_precision=True,
    resume_checkpoint="",
    wandb_run_id="",
    wandb_resume="allow",
    data_root="",
    results_root="",
    use_wandb=True,
):
    ratios = parse_values(ratio)
    backend = str(backend or "modal")
    data_root = data_root or str(DEFAULT_DATA_ROOT)
    results_root = results_root or str(DEFAULT_RESULTS_ROOT)

    pretrain_epochs = as_int(pretrain_epochs)
    pretrain_batch_size = as_int(pretrain_batch_size)
    probe_batch_size = as_int(probe_batch_size)
    pretrain_lr = as_float(pretrain_lr)
    end2end_epochs = as_int(end2end_epochs)
    end2end_batch_size = as_int(end2end_batch_size)
    end2end_lr = as_float(end2end_lr)
    weight_decay = as_float(weight_decay)
    temperature = as_float(temperature)
    use_blur = as_bool(use_blur)
    save_interval = as_int(save_interval)
    mixed_precision = as_bool(mixed_precision)
    use_wandb = as_bool(use_wandb)

    if backend == "local":
        return _run_local(
            mode=mode,
            ratios=ratios,
            encoder=encoder,
            checkpoint_path=checkpoint_path,
            checkpoint_dir=checkpoint_dir,
            pretrain_epochs=pretrain_epochs,
            pretrain_batch_size=pretrain_batch_size,
            probe_batch_size=probe_batch_size,
            pretrain_lr=pretrain_lr,
            end2end_epochs=end2end_epochs,
            end2end_batch_size=end2end_batch_size,
            end2end_lr=end2end_lr,
            weight_decay=weight_decay,
            temperature=temperature,
            use_blur=use_blur,
            run_name=run_name,
            save_interval=save_interval,
            mixed_precision=mixed_precision,
            resume_checkpoint=resume_checkpoint,
            wandb_run_id=wandb_run_id,
            wandb_resume=wandb_resume,
            data_root=data_root,
            results_root=results_root,
            use_wandb=use_wandb,
        )

    _require_modal()
    return _run_modal(
        mode=mode,
        ratios=ratios,
        encoder=encoder,
        checkpoint_path=checkpoint_path,
        checkpoint_dir=checkpoint_dir,
        pretrain_epochs=pretrain_epochs,
        pretrain_batch_size=pretrain_batch_size,
        probe_batch_size=probe_batch_size,
        pretrain_lr=pretrain_lr,
        end2end_epochs=end2end_epochs,
        end2end_batch_size=end2end_batch_size,
        end2end_lr=end2end_lr,
        weight_decay=weight_decay,
        temperature=temperature,
        use_blur=use_blur,
        run_name=run_name,
        save_interval=save_interval,
        mixed_precision=mixed_precision,
        resume_checkpoint=resume_checkpoint,
        wandb_run_id=wandb_run_id,
        wandb_resume=wandb_resume,
    )


def _run_local(**kwargs):
    mode = kwargs["mode"]
    ratios = kwargs["ratios"]
    encoder = kwargs["encoder"]

    if kwargs["checkpoint_path"]:
        if len(ratios) != 1:
            raise ValueError("--checkpoint-path can only be used with one --ratio")
        metrics = run_eval(
            ratio=ratios[0],
            encoder=encoder,
            checkpoint_path=kwargs["checkpoint_path"],
            batch_size=kwargs["probe_batch_size"],
            data_root=kwargs["data_root"],
        )
        _print(metrics)
        return metrics

    if mode == "resume-classifier":
        if not kwargs["checkpoint_dir"]:
            raise ValueError("--checkpoint-dir is required for --mode resume-classifier")
        summary = run_resume_classifier(
            checkpoint_dir=kwargs["checkpoint_dir"],
            ratio=",".join(ratios) if ratios else "",
            encoder=encoder,
            probe_batch_size=kwargs["probe_batch_size"],
            data_root=kwargs["data_root"],
            use_wandb=kwargs["use_wandb"],
            backend="local",
        )
        _print(summary)
        return summary

    if mode == "end2end":
        summaries = {}
        run_name = kwargs["run_name"] if kwargs["run_name"] != "simclr" else "end2end"
        for ratio in ratios:
            summaries[ratio] = run_end2end_train(
                ratio=ratio,
                encoder=encoder,
                num_epochs=kwargs["end2end_epochs"],
                batch_size=kwargs["end2end_batch_size"],
                learning_rate=kwargs["end2end_lr"],
                weight_decay=kwargs["weight_decay"],
                run_name=run_name,
                save_interval=kwargs["save_interval"],
                data_root=kwargs["data_root"],
                results_root=kwargs["results_root"],
                use_wandb=kwargs["use_wandb"],
                backend="local",
            )
        payload = summaries if len(ratios) > 1 else summaries[ratios[0]]
        _print(payload)
        return payload

    if mode != "simclr":
        raise ValueError(f"Unknown mode: {mode}")
    if kwargs["resume_checkpoint"] and len(ratios) != 1:
        raise ValueError("--resume-checkpoint can only be used with one --ratio")

    summaries = {}
    for ratio in ratios:
        summaries[ratio] = run_simclr_train(
            ratio=ratio,
            encoder=encoder,
            pretrain_epochs=kwargs["pretrain_epochs"],
            pretrain_batch_size=kwargs["pretrain_batch_size"],
            probe_batch_size=kwargs["probe_batch_size"],
            pretrain_lr=kwargs["pretrain_lr"],
            temperature=kwargs["temperature"],
            use_blur=kwargs["use_blur"],
            run_name=kwargs["run_name"],
            save_interval=kwargs["save_interval"],
            mixed_precision=kwargs["mixed_precision"],
            resume_checkpoint=kwargs["resume_checkpoint"] if len(ratios) == 1 else "",
            wandb_run_id=kwargs["wandb_run_id"] if len(ratios) == 1 else "",
            wandb_resume=kwargs["wandb_resume"],
            data_root=kwargs["data_root"],
            results_root=kwargs["results_root"],
            use_wandb=kwargs["use_wandb"],
            backend="local",
        )
    payload = summaries if len(ratios) > 1 else summaries[ratios[0]]
    _print(payload)
    return payload


def _run_modal(**kwargs):
    mode = kwargs["mode"]
    ratios = kwargs["ratios"]
    encoder = kwargs["encoder"]

    if mode == "resume-classifier":
        if not kwargs["checkpoint_dir"]:
            raise ValueError("--checkpoint-dir is required for --mode resume-classifier")
        summary = resume_classifier.remote(
            kwargs["checkpoint_dir"],
            ",".join(ratios) if ratios else "",
            encoder,
            kwargs["probe_batch_size"],
        )
        _print(summary)
        return summary

    if kwargs["checkpoint_path"]:
        if len(ratios) != 1:
            raise ValueError("--checkpoint-path can only be used with one --ratio")
        metrics = eval.remote(ratios[0], encoder, kwargs["checkpoint_path"], kwargs["probe_batch_size"])
        _print(metrics)
        return metrics

    if mode == "end2end":
        run_names = [kwargs["run_name"] if kwargs["run_name"] != "simclr" else "end2end" for _ in ratios]
        if len(ratios) == 1:
            summary = train_baseline.remote(
                ratios[0],
                encoder,
                kwargs["end2end_epochs"],
                kwargs["end2end_batch_size"],
                kwargs["end2end_lr"],
                kwargs["weight_decay"],
                run_names[0],
                kwargs["save_interval"],
            )
            _print(summary)
            return summary

        summaries = {}
        for ratio, summary in zip(
            ratios,
            train_baseline.map(
                ratios,
                [encoder] * len(ratios),
                [kwargs["end2end_epochs"]] * len(ratios),
                [kwargs["end2end_batch_size"]] * len(ratios),
                [kwargs["end2end_lr"]] * len(ratios),
                [kwargs["weight_decay"]] * len(ratios),
                run_names,
                [kwargs["save_interval"]] * len(ratios),
            ),
        ):
            summaries[ratio] = summary
        _print(summaries)
        return summaries

    if mode != "simclr":
        raise ValueError(f"Unknown mode: {mode}")
    if kwargs["resume_checkpoint"] and len(ratios) != 1:
        raise ValueError("--resume-checkpoint can only be used with one --ratio")

    if len(ratios) == 1:
        summary = train.remote(
            ratios[0],
            encoder,
            kwargs["pretrain_epochs"],
            kwargs["pretrain_batch_size"],
            kwargs["probe_batch_size"],
            kwargs["pretrain_lr"],
            kwargs["temperature"],
            kwargs["use_blur"],
            kwargs["run_name"],
            kwargs["save_interval"],
            kwargs["mixed_precision"],
            kwargs["resume_checkpoint"],
            kwargs["wandb_run_id"],
            kwargs["wandb_resume"],
        )
        _print(summary)
        return summary

    summaries = {}
    for ratio, summary in zip(
        ratios,
        train.map(
            ratios,
            [encoder] * len(ratios),
            [kwargs["pretrain_epochs"]] * len(ratios),
            [kwargs["pretrain_batch_size"]] * len(ratios),
            [kwargs["probe_batch_size"]] * len(ratios),
            [kwargs["pretrain_lr"]] * len(ratios),
            [kwargs["temperature"]] * len(ratios),
            [kwargs["use_blur"]] * len(ratios),
            [kwargs["run_name"]] * len(ratios),
            [kwargs["save_interval"]] * len(ratios),
            [kwargs["mixed_precision"]] * len(ratios),
            [kwargs["resume_checkpoint"]] * len(ratios),
            [kwargs["wandb_run_id"]] * len(ratios),
            [kwargs["wandb_resume"]] * len(ratios),
        ),
    ):
        summaries[ratio] = summary
    _print(summaries)
    return summaries


def _build_arg_parser():
    import argparse

    parser = argparse.ArgumentParser(description="Train lab4 SimCLR locally, on a server, or on Modal.")
    parser.add_argument("--mode", default="simclr", choices=["simclr", "end2end", "resume-classifier"])
    parser.add_argument("--backend", default="local", choices=["local", "modal"])
    parser.add_argument("--ratio", default="r10")
    parser.add_argument("--encoder", default="resnet18")
    parser.add_argument("--checkpoint-path", default="")
    parser.add_argument("--checkpoint-dir", default="")
    parser.add_argument("--pretrain-epochs", default=100, type=int)
    parser.add_argument("--pretrain-batch-size", default=1024, type=int)
    parser.add_argument("--probe-batch-size", default=256, type=int)
    parser.add_argument("--pretrain-lr", default=1e-3, type=float)
    parser.add_argument("--end2end-epochs", default=100, type=int)
    parser.add_argument("--end2end-batch-size", default=256, type=int)
    parser.add_argument("--end2end-lr", default=1e-3, type=float)
    parser.add_argument("--weight-decay", default=1e-4, type=float)
    parser.add_argument("--temperature", default=0.5, type=float)
    parser.add_argument("--use-blur", default=False, type=as_bool)
    parser.add_argument("--run-name", default="simclr")
    parser.add_argument("--save-interval", default=0, type=int)
    parser.add_argument("--mixed-precision", default=True, type=as_bool)
    parser.add_argument("--resume-checkpoint", default="")
    parser.add_argument("--wandb-run-id", default="")
    parser.add_argument("--wandb-resume", default="allow")
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--results-root", default=str(DEFAULT_RESULTS_ROOT))
    parser.add_argument("--use-wandb", default=True, type=as_bool)
    return parser


if __name__ == "__main__":
    cli_args = _build_arg_parser().parse_args()
    main(**vars(cli_args))
