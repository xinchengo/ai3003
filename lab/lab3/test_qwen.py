import json
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence

import modal


LAB_DIR = Path(__file__).parent

APP_NAME = "ai3003-lab3-qwen-vllm"
DEFAULT_MODEL = "Qwen/Qwen3.5-2B"
DEFAULT_TRAIN_CONFIG = "end2end-v14-rotary"
DEFAULT_SCALEDOWN_WINDOW_SECONDS = 20 * 60
SYSTEM_PROMPT = (
    "You will receive a clipped IMDB movie review as an input, and you should try to"
    "deduce whether the sentiment of the review is positive or negative. "
    "The review is truncated to the last 256 tokens"
    "Return exactly one lowercase word: positive or negative."
)

app = modal.App(APP_NAME)
data = modal.Volume.from_name("ai3003-lab3-data")
results = modal.Volume.from_name("ai3003-lab3-results")
hf_cache = modal.Volume.from_name("ai3003-lab3-hf-cache", create_if_missing=True)
vllm_cache = modal.Volume.from_name("ai3003-lab3-vllm-cache", create_if_missing=True)

image = (
    modal.Image.from_registry(
        "nvidia/cuda:13.2.1-cudnn-devel-ubuntu24.04",
        add_python="3.12",
    )
    .entrypoint([])
    .apt_install("ca-certificates")
    .uv_pip_install(
        "certifi",
        "huggingface-hub",
        "pandas",
        "scikit-learn",
        "tokenizers",
        "tqdm",
        "vllm",
    )
    .env(
        {
            "HF_HOME": "/root/.cache/huggingface",
            "HF_XET_HIGH_PERFORMANCE": "1",
            "REQUESTS_CA_BUNDLE": "/etc/ssl/certs/ca-certificates.crt",
            "SSL_CERT_FILE": "/etc/ssl/certs/ca-certificates.crt",
            "CURL_CA_BUNDLE": "/etc/ssl/certs/ca-certificates.crt",
        }
    )
    .workdir("/root")
    .add_local_file(LAB_DIR / "config.json", remote_path="/root/config.json")
    .add_local_file(LAB_DIR / "config_utils.py", remote_path="/root/config_utils.py")
    .add_local_file(LAB_DIR / "tokenizer.py", remote_path="/root/tokenizer.py")
)

VOLUMES = {
    "/root/data": data,
    "/root/results": results,
    "/root/.cache/huggingface": hf_cache,
    "/root/.cache/vllm": vllm_cache,
}


def _configure_ssl() -> None:
    try:
        import certifi

        cert_path = certifi.where()
    except ImportError:
        cert_path = "/etc/ssl/certs/ca-certificates.crt"

    if Path(cert_path).exists():
        os.environ["SSL_CERT_FILE"] = cert_path
        os.environ["REQUESTS_CA_BUNDLE"] = cert_path
        os.environ["CURL_CA_BUNDLE"] = cert_path


def _parse_label(text: str) -> str:
    match = re.search(r"\b(positive|negative)\b", text.lower())
    return match.group(1) if match else "unknown"


def _counts(y_true: Sequence[int], y_pred: Sequence[int]) -> Dict[str, int]:
    return {
        "tp": sum(t == 1 and p == 1 for t, p in zip(y_true, y_pred)),
        "tn": sum(t == 0 and p == 0 for t, p in zip(y_true, y_pred)),
        "fp": sum(t == 0 and p == 1 for t, p in zip(y_true, y_pred)),
        "fn": sum(t == 1 and p == 0 for t, p in zip(y_true, y_pred)),
    }


def _metrics(counts: Dict[str, int]) -> Dict[str, float]:
    tp, tn, fp, fn = counts["tp"], counts["tn"], counts["fp"], counts["fn"]
    total = tp + tn + fp + fn
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "accuracy": (tp + tn) / total if total else 0.0,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def _merge_metrics(parts: Sequence[Dict]) -> Dict:
    counts = {"tp": 0, "tn": 0, "fp": 0, "fn": 0}
    for part in parts:
        for key in counts:
            counts[key] += part["counts"][key]

    merged = dict(parts[0])
    merged.update(
        {
            "num_samples": sum(part["num_samples"] for part in parts),
            "num_shards": len(parts),
            "shard_id": None,
            "counts": counts,
            **_metrics(counts),
        }
    )
    return merged


def _load_eval_data(train_config_name: str, dataset: str, limit: int):
    import pandas as pd
    from config_utils import load_config
    from tokenizer import get_tokenizer

    config = load_config("config.json")
    train_config = config["train"][train_config_name]
    tokenizer_name = train_config["tokenizer"]
    tokenizer_config = config["preprocess"][tokenizer_name]
    clip_length = int(tokenizer_config["clip_length"])

    tokenizer = get_tokenizer(name=tokenizer_name, dataset="train")
    df = pd.read_csv(config["dataset"][dataset])
    if limit > 0:
        df = df.head(limit).copy()
    df = df.reset_index(names="source_index")

    reviews = []
    token_counts = []
    for review in df["review"].tolist():
        ids = tokenizer.encode(str(review)).ids
        clipped_ids = ids[-clip_length:]
        reviews.append(tokenizer.decode(clipped_ids).strip())
        token_counts.append(len(clipped_ids))

    return df, reviews, token_counts, tokenizer_name, clip_length


def _prompt(review: str) -> str:
    return (
        "/no_think Classify the sentiment of this movie review.\n"
        "Return only one word: positive or negative.\n\n"
        f"Review: {review}\n"
        "Sentiment:"
    )


def _evaluate_on_modal(
    runner,
    train_config: str,
    dataset: str,
    batch_size: int,
    limit: int,
    parallel: int,
):
    jobs = [
        runner.evaluate.spawn(
            train_config_name=train_config,
            dataset=dataset,
            batch_size=batch_size,
            limit=limit,
            shard_id=shard_id,
            num_shards=parallel,
        )
        for shard_id in range(parallel)
    ]
    parts = modal.FunctionCall.gather(*jobs)
    print(json.dumps(_merge_metrics(parts), indent=2))


class QwenVllm:
    gpu_name = ""

    @modal.enter()
    def start(self):
        start_time = time.time()
        _configure_ssl()

        import vllm

        self.llm = vllm.LLM(
            model=self.model_name,
            max_model_len=self.max_model_len,
            attention_backend="flashinfer",
            async_scheduling=True,
            generation_config="vllm",
            gpu_memory_utilization=0.90,
            trust_remote_code=True,
            disable_log_stats=True,
            enforce_eager=self.fast_start,
        )
        self.sampling_params = vllm.SamplingParams(
            temperature=0.0,
            max_tokens=8,
            stop=["\n"],
        )
        if self.warmup_on_start:
            self.llm.chat(
                [[{"role": "user", "content": "/no_think ready"}]],
                sampling_params=self.sampling_params,
                use_tqdm=False,
            )

        elapsed = time.time() - start_time
        print(
            f"[startup] model={self.model_name} gpu={self.gpu_name} "
            f"fast_start={self.fast_start} warmup_on_start={self.warmup_on_start} "
            f"seconds={elapsed:.2f}"
        )

    @modal.method()
    def evaluate(
        self,
        train_config_name: str = DEFAULT_TRAIN_CONFIG,
        dataset: str = "test",
        batch_size: int = 256,
        limit: int = 0,
        shard_id: int = 0,
        num_shards: int = 1,
    ) -> Dict:
        os.chdir("/root")

        import pandas as pd
        from tqdm import tqdm

        df, reviews, token_counts, tokenizer_name, clip_length = _load_eval_data(
            train_config_name,
            dataset,
            limit,
        )
        if num_shards > 1:
            df = df.iloc[shard_id::num_shards].reset_index(drop=True)
            reviews = reviews[shard_id::num_shards]
            token_counts = token_counts[shard_id::num_shards]

        predictions: List[str] = []
        raw_outputs: List[str] = []
        prompt_tokens = 0
        output_tokens = 0
        start_time = time.time()

        for start in tqdm(range(0, len(reviews), batch_size), desc=f"qwen {dataset}"):
            batch_reviews = reviews[start:start + batch_size]
            messages = [
                [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": _prompt(review)},
                ]
                for review in batch_reviews
            ]
            responses = self.llm.chat(
                messages,
                sampling_params=self.sampling_params,
                use_tqdm=False,
            )
            prompt_tokens += sum(len(response.prompt_token_ids) for response in responses)
            output_tokens += sum(len(response.outputs[0].token_ids) for response in responses)
            outputs = [response.outputs[0].text.strip() for response in responses]
            raw_outputs.extend(outputs)
            predictions.extend(_parse_label(output) for output in outputs)

        true_labels = df["sentiment"].tolist()
        y_true = [1 if label == "positive" else 0 for label in true_labels]
        y_pred = [1 if label == "positive" else 0 for label in predictions]
        counts = _counts(y_true, y_pred)
        elapsed = max(time.time() - start_time, 1e-9)

        metrics = {
            "model": self.model_name,
            "backend": "modal-vllm",
            "gpu": self.gpu_name,
            "dataset": dataset,
            "train_config": train_config_name,
            "tokenizer": tokenizer_name,
            "clip_length": clip_length,
            "clip_side": "tail",
            "batch_size": batch_size,
            "num_samples": len(df),
            "shard_id": shard_id,
            "num_shards": num_shards,
            "unknown_count": sum(label == "unknown" for label in predictions),
            "prompt_tokens": prompt_tokens,
            "output_tokens": output_tokens,
            "seconds": elapsed,
            "prompt_tokens_per_second": prompt_tokens / elapsed,
            "output_tokens_per_second": output_tokens / elapsed,
            "counts": counts,
            **_metrics(counts),
        }

        output_df = pd.DataFrame(
            {
                "source_index": df["source_index"].tolist(),
                "review": df["review"].tolist(),
                "qwen_review": reviews,
                "clip_token_count": token_counts,
                "true_label": true_labels,
                "pred_label": predictions,
                "raw_output": raw_outputs,
            }
        )
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_slug = self.model_name.replace("/", "__")
        out_dir = Path("/root/results/qwen_vllm") / model_slug / train_config_name
        out_dir.mkdir(parents=True, exist_ok=True)
        suffix = f"_shard{shard_id:02d}-of-{num_shards:02d}" if num_shards > 1 else ""
        output_df.to_csv(out_dir / f"{dataset}_{timestamp}{suffix}.csv", index=False)
        with open(out_dir / f"{dataset}_{timestamp}{suffix}.json", "w") as f:
            json.dump(metrics, f, indent=2)
        results.commit()

        print(json.dumps(metrics, indent=2))
        return metrics

    @modal.exit()
    def stop(self):
        del self.llm


@app.cls(
    image=image,
    gpu="H100",
    timeout=60 * 60 * 6,
    volumes=VOLUMES,
    scaledown_window=DEFAULT_SCALEDOWN_WINDOW_SECONDS,
)
class QwenH100(QwenVllm):
    gpu_name = "h100"
    model_name: str = modal.parameter(default=DEFAULT_MODEL)
    max_model_len: int = modal.parameter(default=2048)
    fast_start: bool = modal.parameter(default=True)
    warmup_on_start: bool = modal.parameter(default=False)


@app.cls(
    image=image,
    gpu="H200",
    timeout=60 * 60 * 6,
    volumes=VOLUMES,
    scaledown_window=DEFAULT_SCALEDOWN_WINDOW_SECONDS,
)
class QwenH200(QwenVllm):
    gpu_name = "h200"
    model_name: str = modal.parameter(default=DEFAULT_MODEL)
    max_model_len: int = modal.parameter(default=2048)
    fast_start: bool = modal.parameter(default=True)
    warmup_on_start: bool = modal.parameter(default=False)


@app.cls(
    image=image,
    gpu="B200",
    timeout=60 * 60 * 6,
    volumes=VOLUMES,
    scaledown_window=DEFAULT_SCALEDOWN_WINDOW_SECONDS,
)
class QwenB200(QwenVllm):
    gpu_name = "b200"
    model_name: str = modal.parameter(default=DEFAULT_MODEL)
    max_model_len: int = modal.parameter(default=2048)
    fast_start: bool = modal.parameter(default=True)
    warmup_on_start: bool = modal.parameter(default=False)


def _runner(
    gpu: str,
    model_name: str,
    max_model_len: int,
    fast_start: bool,
    warmup_on_start: bool,
):
    if gpu == "h100":
        return QwenH100(
            model_name=model_name,
            max_model_len=max_model_len,
            fast_start=fast_start,
            warmup_on_start=warmup_on_start,
        )
    if gpu == "h200":
        return QwenH200(
            model_name=model_name,
            max_model_len=max_model_len,
            fast_start=fast_start,
            warmup_on_start=warmup_on_start,
        )
    if gpu == "b200":
        return QwenB200(
            model_name=model_name,
            max_model_len=max_model_len,
            fast_start=fast_start,
            warmup_on_start=warmup_on_start,
        )
    raise ValueError("--gpu must be one of: h100, h200, b200")


@app.local_entrypoint()
def main(
    model_name: str = DEFAULT_MODEL,
    train_config: str = DEFAULT_TRAIN_CONFIG,
    dataset: str = "test",
    gpu: str = "h200",
    parallel: int = 1,
    batch_size: int = 256,
    limit: int = 0,
    max_model_len: int = 2048,
    fast_start: bool = True,
    warmup_on_start: bool = False,
):
    gpu = gpu.lower()
    parallel = max(1, parallel)
    runner = _runner(
        gpu,
        model_name,
        max_model_len,
        fast_start=fast_start,
        warmup_on_start=warmup_on_start,
    )

    _evaluate_on_modal(
        runner=runner,
        train_config=train_config,
        dataset=dataset,
        batch_size=batch_size,
        limit=limit,
        parallel=parallel,
    )
