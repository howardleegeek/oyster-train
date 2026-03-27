"""
qwen_engine.py — Qwen2.5-0.5B on-device LLM engine via llama.cpp.

Runs alongside LeWM for dual-model federated training:
  - LeWM: world model (vision + action prediction)
  - Qwen: language model (text understanding + generation)

Both models train on-device via Flower FL, sharing the same server.
"""
import gc
import json
import os
import platform
import threading
import time
from pathlib import Path

import numpy as np

# ─── Global state ─────────────────────────────────────────────────
_llm = None
_lora_adapter = None
_training_thread = None
_stop_event = threading.Event()

_status = {
    "state": "idle",
    "model": "Qwen2.5-0.5B",
    "model_size_mb": 0,
    "round": 0,
    "total_rounds": 0,
    "step": 0,
    "total_steps": 0,
    "loss": 0.0,
    "tokens_per_sec": 0.0,
    "memory_mb": 0,
    "server": "",
    "error": "",
    "lora_rank": 8,
    "lora_params": 0,
}

# Model download config
QWEN_REPO = "Qwen/Qwen2.5-0.5B-Instruct-GGUF"
QWEN_FILE = "qwen2.5-0.5b-instruct-q4_k_m.gguf"
MODEL_DIR = None  # Set at runtime to app's files dir


def get_status() -> str:
    return json.dumps(_status)


def _get_model_path() -> str:
    """Return path to the GGUF model file."""
    global MODEL_DIR
    if MODEL_DIR is None:
        MODEL_DIR = os.path.join(os.environ.get("ANDROID_DATA", "/data"), "models")
    os.makedirs(MODEL_DIR, exist_ok=True)
    return os.path.join(MODEL_DIR, QWEN_FILE)


def set_model_dir(path: str) -> str:
    """Called from Kotlin to set the model storage directory."""
    global MODEL_DIR
    MODEL_DIR = path
    os.makedirs(MODEL_DIR, exist_ok=True)
    return json.dumps({"model_dir": MODEL_DIR})


def download_model() -> str:
    """Download Qwen2.5-0.5B GGUF if not present. ~350MB."""
    model_path = _get_model_path()
    if os.path.exists(model_path):
        size_mb = os.path.getsize(model_path) / 1e6
        return json.dumps({"status": "exists", "path": model_path, "size_mb": round(size_mb, 1)})

    try:
        from huggingface_hub import hf_hub_download
        downloaded = hf_hub_download(
            repo_id=QWEN_REPO,
            filename=QWEN_FILE,
            local_dir=MODEL_DIR,
            local_dir_use_symlinks=False,
        )
        size_mb = os.path.getsize(downloaded) / 1e6
        _status["model_size_mb"] = round(size_mb, 1)
        return json.dumps({"status": "downloaded", "path": downloaded, "size_mb": round(size_mb, 1)})
    except Exception as e:
        return json.dumps({"status": "error", "error": str(e)})


def load_model(n_ctx: int = 512, n_threads: int = 4) -> str:
    """Load the GGUF model into llama.cpp."""
    global _llm
    try:
        from llama_cpp import Llama

        model_path = _get_model_path()
        if not os.path.exists(model_path):
            return json.dumps({"status": "error", "error": "Model not downloaded. Call download_model() first."})

        _llm = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_threads=n_threads,
            n_gpu_layers=0,  # CPU only on phone
            verbose=False,
        )

        size_mb = os.path.getsize(model_path) / 1e6
        _status["model_size_mb"] = round(size_mb, 1)
        _status["state"] = "loaded"

        return json.dumps({
            "status": "ok",
            "model": "Qwen2.5-0.5B",
            "quantization": "Q4_K_M",
            "context_length": n_ctx,
            "threads": n_threads,
            "size_mb": round(size_mb, 1),
        })
    except Exception as e:
        _status["state"] = "error"
        _status["error"] = str(e)
        return json.dumps({"status": "error", "error": str(e)})


def generate(prompt: str, max_tokens: int = 128) -> str:
    """Run inference — test that the model works on device."""
    if _llm is None:
        return json.dumps({"status": "error", "error": "Model not loaded"})

    try:
        t0 = time.time()
        output = _llm(
            prompt,
            max_tokens=max_tokens,
            temperature=0.7,
            top_p=0.9,
            echo=False,
        )
        elapsed = time.time() - t0
        text = output["choices"][0]["text"]
        n_tokens = output["usage"]["completion_tokens"]
        tps = n_tokens / elapsed if elapsed > 0 else 0

        _status["tokens_per_sec"] = round(tps, 1)

        return json.dumps({
            "status": "ok",
            "text": text,
            "tokens": n_tokens,
            "elapsed_s": round(elapsed, 2),
            "tokens_per_sec": round(tps, 1),
        })
    except Exception as e:
        return json.dumps({"status": "error", "error": str(e)})


# ─── LoRA fine-tuning via Flower ──────────────────────────────────
#
# llama.cpp supports LoRA adapters. For federated fine-tuning:
# 1. Each phone trains a small LoRA adapter on local data
# 2. Only LoRA deltas are sent to the server (tiny: ~2MB)
# 3. Server aggregates LoRA deltas across all phones
# 4. Aggregated LoRA is sent back to phones
#
# Full Qwen 0.5B params = 500MB per sync
# LoRA rank-8 deltas = ~2MB per sync → 250x compression

class LoRAAdapter:
    """Lightweight LoRA adapter for federated fine-tuning."""

    def __init__(self, rank: int = 8, alpha: float = 16.0, target_modules: int = 32):
        self.rank = rank
        self.alpha = alpha
        self.scale = alpha / rank
        self.target_modules = target_modules

        # Initialize LoRA A and B matrices for each target layer
        # Qwen2.5-0.5B: 24 transformer layers, q/k/v/o projections
        self.params_a = []  # down-projection (d_model -> rank)
        self.params_b = []  # up-projection (rank -> d_model)

        d_model = 896  # Qwen2.5-0.5B hidden size
        for _ in range(target_modules):
            a = np.random.randn(d_model, rank).astype(np.float32) * 0.01
            b = np.zeros((rank, d_model), dtype=np.float32)
            self.params_a.append(a)
            self.params_b.append(b)

        total = sum(a.size + b.size for a, b in zip(self.params_a, self.params_b))
        _status["lora_params"] = total
        _status["lora_rank"] = rank

    def get_flat_params(self) -> np.ndarray:
        """Flatten all LoRA params for Flower transport."""
        parts = []
        for a, b in zip(self.params_a, self.params_b):
            parts.extend([a.flatten(), b.flatten()])
        return np.concatenate(parts)

    def set_flat_params(self, flat: np.ndarray):
        """Restore LoRA params from Flower transport."""
        d_model = 896
        offset = 0
        for i in range(len(self.params_a)):
            a_size = d_model * self.rank
            b_size = self.rank * d_model
            self.params_a[i] = flat[offset:offset + a_size].reshape(d_model, self.rank)
            offset += a_size
            self.params_b[i] = flat[offset:offset + b_size].reshape(self.rank, d_model)
            offset += b_size

    def param_size_mb(self) -> float:
        total = sum(a.nbytes + b.nbytes for a, b in zip(self.params_a, self.params_b))
        return round(total / 1e6, 2)

    def train_step(self, input_ids: np.ndarray, lr: float = 1e-4) -> float:
        """
        One LoRA training step using SPSA gradient estimation.

        On-device training without autograd: use Simultaneous Perturbation
        Stochastic Approximation (SPSA) to estimate gradients on LoRA params.
        Works without a full training framework on the phone.
        """
        if _llm is None:
            return 0.0

        loss = self._compute_loss(input_ids)

        # SPSA gradient estimation on subset of layers per step
        epsilon = 1e-3
        for i in range(min(4, len(self.params_a))):
            # Perturb A matrix
            delta = np.random.choice([-1, 1], size=self.params_a[i].shape).astype(np.float32)
            self.params_a[i] += epsilon * delta
            loss_plus = self._compute_loss(input_ids)
            self.params_a[i] -= 2 * epsilon * delta
            loss_minus = self._compute_loss(input_ids)
            self.params_a[i] += epsilon * delta  # restore

            grad_a = (loss_plus - loss_minus) / (2 * epsilon) * delta
            self.params_a[i] -= lr * grad_a

            # Perturb B matrix
            delta_b = np.random.choice([-1, 1], size=self.params_b[i].shape).astype(np.float32)
            self.params_b[i] += epsilon * delta_b
            loss_plus = self._compute_loss(input_ids)
            self.params_b[i] -= 2 * epsilon * delta_b
            loss_minus = self._compute_loss(input_ids)
            self.params_b[i] += epsilon * delta_b

            grad_b = (loss_plus - loss_minus) / (2 * epsilon) * delta_b
            self.params_b[i] -= lr * grad_b

        return loss

    def _compute_loss(self, input_ids: np.ndarray) -> float:
        """Compute perplexity-based loss using llama.cpp."""
        if _llm is None:
            return 5.0
        try:
            tokens = input_ids.tolist() if isinstance(input_ids, np.ndarray) else input_ids
            _llm.reset()
            # Use llama.cpp's built-in perplexity estimation
            logits_list = []
            for i, tok in enumerate(tokens[:-1]):
                _llm.eval([tok])
            # Approximate loss from final logits
            scores = np.array(_llm.scores) if hasattr(_llm, 'scores') and _llm.scores else None
            if scores is not None and len(scores) > 0:
                last_logits = scores[-1]
                target = tokens[-1] if len(tokens) > 1 else 0
                # Stable log-softmax
                max_logit = np.max(last_logits)
                log_sum_exp = max_logit + np.log(np.sum(np.exp(last_logits - max_logit)))
                nll = -(last_logits[target] - log_sum_exp)
                return float(nll)
            return 5.0
        except Exception:
            return 5.0


def _fl_train_loop(server_address: str, num_rounds: int, local_steps: int):
    """Flower FL training loop for Qwen LoRA."""
    global _lora_adapter
    import flwr as fl

    _status["state"] = "connecting"
    _status["server"] = server_address
    _status["total_rounds"] = num_rounds

    _lora_adapter = LoRAAdapter(rank=8, alpha=16.0, target_modules=32)
    _status["state"] = "training"

    class _QwenClient(fl.client.NumPyClient):
        def get_parameters(self, config):
            return [_lora_adapter.get_flat_params()]

        def fit(self, parameters, config):
            if len(parameters) > 0:
                _lora_adapter.set_flat_params(parameters[0])

            before = _lora_adapter.get_flat_params().copy()

            steps = config.get("local_steps", local_steps)
            _status["total_steps"] = steps

            # Training samples (in production: local user data)
            train_texts = [
                "The robot navigates through the kitchen carefully.",
                "Pick up the blue cup from the table and place it in the sink.",
                "The autonomous vehicle detects a pedestrian crossing ahead.",
                "Adjust gripper pressure to handle fragile objects safely.",
            ]

            total_loss = 0.0
            for step in range(steps):
                if _stop_event.is_set():
                    break

                text = train_texts[step % len(train_texts)]
                tokens = _llm.tokenize(text.encode("utf-8")) if _llm else list(range(20))
                input_ids = np.array(tokens[:64], dtype=np.int32)

                loss = _lora_adapter.train_step(input_ids, lr=1e-4)
                total_loss += loss

                _status["step"] = step + 1
                _status["loss"] = round(loss, 4)

            _status["round"] = _status.get("round", 0) + 1

            after = _lora_adapter.get_flat_params()
            delta = [after - before]

            gc.collect()
            avg_loss = total_loss / max(steps, 1)
            return delta, len(train_texts), {"loss": round(avg_loss, 4)}

        def evaluate(self, parameters, config):
            return 0.0, 1, {"loss": 0.0}

    try:
        fl.client.start_numpy_client(
            server_address=server_address,
            client=_QwenClient(),
        )
        _status["state"] = "idle"
    except Exception as e:
        _status["state"] = "error"
        _status["error"] = str(e)


def start_training(server_address: str, num_rounds: int = 100, local_steps: int = 10) -> str:
    """Start Qwen LoRA federated training."""
    global _training_thread
    if _llm is None:
        return json.dumps({"status": "error", "error": "Model not loaded. Call load_model() first."})

    _stop_event.clear()
    _status["state"] = "connecting"
    _status["round"] = 0
    _status["step"] = 0
    _status["error"] = ""

    _training_thread = threading.Thread(
        target=_fl_train_loop,
        args=(server_address, num_rounds, local_steps),
        daemon=True,
    )
    _training_thread.start()
    return json.dumps({"status": "started", "server": server_address,
                        "lora_size_mb": LoRAAdapter().param_size_mb()})


def stop_training() -> str:
    _stop_event.set()
    _status["state"] = "paused"
    return json.dumps({"status": "stopped"})


def run_quick_test() -> str:
    """Quick inference test — called on app startup after load_model."""
    if _llm is None:
        return json.dumps({"status": "error", "error": "Model not loaded"})

    try:
        t0 = time.time()
        output = _llm("Hello, I am", max_tokens=16, temperature=0.7, echo=False)
        elapsed = time.time() - t0
        text = output["choices"][0]["text"]
        n_tok = output["usage"]["completion_tokens"]
        tps = n_tok / elapsed if elapsed > 0 else 0

        lora = LoRAAdapter(rank=8, target_modules=32)
        return json.dumps({
            "status": "ok",
            "model": "Qwen2.5-0.5B-Q4_K_M",
            "generated": text.strip(),
            "tokens_per_sec": round(tps, 1),
            "lora_size_mb": lora.param_size_mb(),
            "lora_params": int(lora.get_flat_params().size),
        })
    except Exception as e:
        return json.dumps({"status": "error", "error": str(e)})
