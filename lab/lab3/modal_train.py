import os
from pathlib import Path
import json
import shutil
from glob import glob

import modal


LAB_DIR = Path(__file__).parent

app = modal.App("ai3003-lab3")
data = modal.Volume.from_name("ai3003-lab3-data")
results = modal.Volume.from_name("ai3003-lab3-results")

image = (
    modal.Image.debian_slim()
    .pip_install("torch", "numpy", "tokenizers", "pandas", 
                 "scikit-learn", "wandb", "tqdm")
    .workdir("/root")
    .add_local_file(LAB_DIR / "config.json", remote_path="/root/config.json")
    .add_local_file(LAB_DIR / "config_utils.py", remote_path="/root/config_utils.py")
    .add_local_file(LAB_DIR / "trainer.py", remote_path="/root/trainer.py")
    .add_local_file(LAB_DIR / "model.py", remote_path="/root/model.py")
    .add_local_file(LAB_DIR / "tokenizer.py", remote_path="/root/tokenizer.py")
)

def _parse_datasets(eval_datasets):
    if isinstance(eval_datasets, str):
        return [item.strip() for item in eval_datasets.split(",") if item.strip()]
    return list(eval_datasets)


def _parse_configs(config):
    if isinstance(config, str):
        return [item.strip() for item in config.split(",") if item.strip()]
    return list(config)


def _save_best_checkpoint(checkpoint_dir, eval_results):
    best_name = max(
        eval_results,
        key=lambda name: (
            eval_results[name].get("val", {}).get("accuracy", float("-inf")),
            eval_results[name].get("val", {}).get("f1", float("-inf")),
        )
    )
    best_src = os.path.join(checkpoint_dir, best_name)
    best_dst = os.path.join(checkpoint_dir, "best.pth")
    if os.path.abspath(best_src) != os.path.abspath(best_dst):
        shutil.copyfile(best_src, best_dst)

    best_summary = {
        "best_checkpoint": best_name,
        "best_checkpoint_path": best_src,
        "saved_as": best_dst,
        "selection_metric": "val.accuracy",
        "metrics": eval_results[best_name],
    }
    with open(os.path.join(checkpoint_dir, "best_checkpoint.json"), "w") as f:
        json.dump(best_summary, f, indent=2)
    return best_summary


@app.function(
    image=image,
    gpu="L4",
    timeout=60 * 60 * 12,
    volumes={"/root/data": data, "/root/results": results},
    secrets=[modal.Secret.from_name("wandb-secret")],
)
def train(config="end2end-config1", run_eval=True, eval_datasets="train,val"):
    os.chdir("/root")

    from trainer import evaluate, train_pipeline

    checkpoint_path = train_pipeline(config)
    results.commit()

    eval_results = {}
    best_summary = None
    if run_eval:
        datasets = _parse_datasets(eval_datasets)
        # Find all cached checkpoints for this training run
        checkpoint_dir = os.path.dirname(checkpoint_path)
        all_checkpoints = sorted(glob(f"{checkpoint_dir}/*.pth"))
        
        # Evaluate each checkpoint
        for ckpt_path in all_checkpoints:
            ckpt_name = os.path.basename(ckpt_path)
            print(f"\nEvaluating {ckpt_name}...")
            metrics = evaluate(config, ckpt_path, datasets=datasets)
            eval_results[ckpt_name] = metrics
            for dataset, dataset_metrics in metrics.items():
                print(
                    f"  {dataset:5s} - Acc: {dataset_metrics['accuracy']:.4f}, "
                    f"F1: {dataset_metrics['f1']:.4f}"
                )

        with open(os.path.join(checkpoint_dir, "eval_results.json"), "w") as f:
            json.dump(eval_results, f, indent=2)
        if "val" in datasets and eval_results:
            best_summary_path = os.path.join(checkpoint_dir, "best_checkpoint.json")
            if os.path.exists(best_summary_path):
                with open(best_summary_path, "r") as f:
                    best_summary = json.load(f)
            else:
                best_summary = _save_best_checkpoint(checkpoint_dir, eval_results)
            print("\nBest checkpoint:")
            print(json.dumps(best_summary, indent=2))
        
        results.commit()

    return checkpoint_path, eval_results, best_summary


@app.function(
    image=image,
    gpu="L4",
    timeout=60 * 60 * 4,
    volumes={"/root/data": data, "/root/results": results},
)
def eval(config="end2end-config1", checkpoint_path="", eval_datasets="train,val"):
    os.chdir("/root")

    from trainer import evaluate

    datasets = _parse_datasets(eval_datasets)
    if os.path.isdir(checkpoint_path):
        eval_results = {}
        for ckpt_path in sorted(glob(f"{checkpoint_path}/*.pth")):
            ckpt_name = os.path.basename(ckpt_path)
            print(f"\nEvaluating {ckpt_name}...")
            eval_results[ckpt_name] = evaluate(
                config, ckpt_path, datasets=datasets)
        return eval_results

    return evaluate(config, checkpoint_path, datasets=datasets)


@app.local_entrypoint()
def main(
    config="end2end-config1",
    checkpoint_path="",
    run_eval=True,
    eval_datasets="train,val",
):
    configs = _parse_configs(config)
    if len(configs) > 1 and checkpoint_path:
        raise ValueError("--checkpoint-path can only be used with one --config")

    if checkpoint_path:
        metrics = eval.remote(configs[0], checkpoint_path, eval_datasets)
        print(f"Metrics: {json.dumps(metrics, indent=2)}")
        return

    if len(configs) == 1:
        checkpoint_path, eval_results, best_summary = train.remote(
            configs[0],
            run_eval,
            eval_datasets,
        )
        print(f"Checkpoint: {checkpoint_path}")
        if eval_results:
            print("\nEvaluation Results:")
            print(json.dumps(eval_results, indent=2))
        if best_summary:
            print("\nBest Checkpoint:")
            print(json.dumps(best_summary, indent=2))
        return

    results_by_config = {}
    run_eval_values = [run_eval] * len(configs)
    eval_dataset_values = [eval_datasets] * len(configs)
    for config_name, result in zip(
        configs,
        train.map(configs, run_eval_values, eval_dataset_values),
    ):
        checkpoint_path, eval_results, best_summary = result
        results_by_config[config_name] = {
            "checkpoint_path": checkpoint_path,
            "eval_results": eval_results,
            "best_summary": best_summary,
        }

    print(json.dumps(results_by_config, indent=2))
