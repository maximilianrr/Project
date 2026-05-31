import io
import importlib.util
import pickle
import re
import sys
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.amp.autocast_mode import autocast


CURRENT_DIR = Path(__file__).resolve().parent
PACKAGE_DIR = CURRENT_DIR.parent
SRC_DIR = CURRENT_DIR.parents[2]
PROJECT_ROOT = CURRENT_DIR.parents[3]
WORKSPACE_ROOT = CURRENT_DIR.parents[4]
NANOCHAT_DIR = WORKSPACE_ROOT / "nanochat"


def _discover_nanochat_repo_root() -> Path | None:
    for ancestor in CURRENT_DIR.parents:
        candidate = ancestor / "nanochat"
        if (candidate / "nanochat" / "__init__.py").is_file():
            return candidate
    return None


NANOCHAT_REPO_ROOT = _discover_nanochat_repo_root()

for path in (CURRENT_DIR, PACKAGE_DIR, SRC_DIR, WORKSPACE_ROOT, NANOCHAT_DIR, NANOCHAT_REPO_ROOT):
    if path is None:
        continue
    if path.exists() and str(path) not in sys.path:
        sys.path.insert(0, str(path))


def _load_config():
    config_path = PACKAGE_DIR / "config.py"
    spec = importlib.util.spec_from_file_location("chatbot_config", config_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load config from {config_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


config = _load_config()

from model import NanoChat


def _ensure_rustbpe_importable() -> None:
    try:
        import rustbpe  # type: ignore # noqa: F401
    except ModuleNotFoundError:
        import types

        shim = types.ModuleType("rustbpe")

        class Tokenizer:
            pass

        shim.Tokenizer = Tokenizer  # type: ignore[attr-defined]
        sys.modules["rustbpe"] = shim


def load_tokenizer(tokenizer_path: str | None = None):
    tokenizer_file = Path(tokenizer_path or config.TOKENIZER_PKL)
    if not tokenizer_file.is_file():
        raise FileNotFoundError(f"Tokenizer not found at {tokenizer_file}")

    _ensure_rustbpe_importable()
    with tokenizer_file.open("rb") as file:
        return pickle.load(file)


def _candidate_checkpoint_paths() -> list[Path]:
    return [
        Path(config.BEST_MODEL_PTH),
        Path(config.CHECKPOINTS_DIR) / "best.pth",
        Path(config.CHECKPOINTS_DIR) / "latest.pth",
        Path(config.OUTPUT_DIR) / "best_model.pth",
        Path(config.OUTPUT_DIR) / "checkpoints" / "best_model.pth",
        Path(config.MODAL_CHECKPOINTS_DIR) / "best_model.pth",
    ]


def resolve_checkpoint_path(checkpoint_path: str | None = None) -> Path:
    if checkpoint_path is not None:
        resolved = Path(checkpoint_path)
        if resolved.is_file():
            return resolved
        raise FileNotFoundError(f"Model checkpoint not found at {resolved}")

    for candidate in _candidate_checkpoint_paths():
        if candidate.is_file():
            return candidate

    raise FileNotFoundError(
        "No model checkpoint found. Looked in: "
        + ", ".join(str(path) for path in _candidate_checkpoint_paths())
    )


def load_model(device=None, checkpoint_path: str | None = None):
    device = device or config.DEVICE
    checkpoint_file = resolve_checkpoint_path(checkpoint_path)
    checkpoint = torch.load(checkpoint_file, map_location=device)

    if isinstance(checkpoint, dict):
        for key in ("model", "state_dict", "model_state_dict", "model_state"):
            if key in checkpoint:
                state_dict = checkpoint[key]
                break
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    token_embedding_shape = state_dict["token_embedding_table.weight"].shape
    block_size, rope_half_dim = state_dict["blocks.0.self_att.cos"].shape
    n_layer = max(int(key.split(".")[1]) for key in state_dict if key.startswith("blocks.")) + 1
    n_embd = token_embedding_shape[1]
    head_dim = rope_half_dim * 2
    n_head = n_embd // head_dim

    inference_config = SimpleNamespace(
        N_EMB=n_embd,
        N_HEAD=n_head,
        N_LAYER=n_layer,
        BLOCK_SIZE=block_size,
        DROPOUT=getattr(config, "DROPOUT", 0.0),
        VOCAB_SIZE=token_embedding_shape[0],
    )

    model = NanoChat(inference_config).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def _tokens_to_text(token_ids: list[int], tokenizer) -> str:
    """Convert token IDs to user-facing text and strip special markers."""

    ignore_tokens = {
        "<|bos|>",
        "<|user_start|>",
        "<|user_end|>",
        "<|assistant_start|>",
        "<|assistant_end|>",
        "<|endoftext|>",
    }

    decoded = tokenizer.decode(token_ids)

    for special_token in ignore_tokens:
        decoded = decoded.replace(special_token, "")
    decoded = decoded.replace("|end|><", "")
    decoded = re.sub(r"<\|[^>]*\|>", "", decoded)
    decoded = " ".join(decoded.split()).strip()

    if not decoded or decoded.lower() in {"end", "assistant", "user"}:
        return "I'm not sure how to answer that right now."
    return decoded


def text_to_tokens(text: str, tokenizer) -> list[int]:
    bos = tokenizer.get_bos_token_id()
    user_start = tokenizer.encode_special("<|user_start|>")
    user_end = tokenizer.encode_special("<|user_end|>")
    assistant_start = tokenizer.encode_special("<|assistant_start|>")

    user_tokens = tokenizer.encode(text)

    return [bos, user_start] + user_tokens + [user_end, assistant_start]


def generate_output(user_input: str, model, device, tokenizer):
    use_amp = device.type == "cuda"
    token_list = text_to_tokens(user_input, tokenizer=tokenizer)

    text_tensor = torch.tensor([token_list], dtype=torch.long).to(device)
    end_token_id = tokenizer.encode_special("<|assistant_end|>")

    with torch.no_grad():
        with autocast(device_type=device.type, enabled=use_amp):
            with redirect_stdout(io.StringIO()):
                generated = model.generate(
                    input=text_tensor,
                    max_new_tokens=200,
                    stop_token_id=end_token_id,
                    temperature=0.6,
                    top_k=10,
                )

    prompt_length = text_tensor.shape[1]
    response_tokens = generated[0][prompt_length:].tolist()

    return _tokens_to_text(response_tokens, tokenizer)


def process_user_input(
    user_input: str,
    *,
    device=None,
    tokenizer=None,
    model=None,
    checkpoint_path: str | None = None,
):
    device = device or config.DEVICE
    tokenizer = tokenizer or load_tokenizer()
    model = model or load_model(device=device, checkpoint_path=checkpoint_path)
    return generate_output(user_input, model, device, tokenizer)
