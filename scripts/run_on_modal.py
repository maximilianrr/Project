import argparse
import os
import pickle
import shutil
import sys
from pathlib import Path

import modal

PROJECT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_DIR / "src"
LOCAL_DATA_DIR = PROJECT_DIR / "data"
LOCAL_REQUIREMENTS = PROJECT_DIR / "requirements.txt"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from chat_model import config

APP_NAME = "nanochat-training"
VOLUME_NAME = "nanochat-data"
REMOTE_PROJECT_ROOT = "/root/Project"
REMOTE_SRC_DIR = f"{REMOTE_PROJECT_ROOT}/src"
REMOTE_DATA_DIR = f"{REMOTE_PROJECT_ROOT}/data"
REMOTE_NANOCHAT_REPO = "/root/nanochat"

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

image = modal.Image.debian_slim()
if LOCAL_REQUIREMENTS.exists():
    image = image.pip_install_from_requirements(str(LOCAL_REQUIREMENTS))
else:
    image = image.pip_install("torch", "tqdm", "numpy")

HAS_MODAL_MOUNT = hasattr(modal, "Mount")

# Build runtime image and function kwargs in a way that works for both old and
# new Modal SDK versions. Older SDKs do not expose `modal.Mount`.
runtime_image = image.add_local_dir(str(SRC_DIR), remote_path=REMOTE_SRC_DIR)
if Path(config.NANOCHAT_DIR).exists():
    runtime_image = runtime_image.add_local_dir(config.NANOCHAT_DIR, remote_path=REMOTE_NANOCHAT_REPO)

sync_image = image
if LOCAL_DATA_DIR.exists():
    sync_image = sync_image.add_local_dir(str(LOCAL_DATA_DIR), remote_path="/root/local_data")

mounts = []
sync_mounts = []
if HAS_MODAL_MOUNT:
    mounts.append(modal.Mount.from_local_dir(str(SRC_DIR), remote_path=REMOTE_SRC_DIR, copy=True))
    sync_mounts.append(modal.Mount.from_local_dir(str(LOCAL_DATA_DIR), remote_path="/root/local_data", copy=True))
    if Path(config.NANOCHAT_DIR).exists():
        mounts.append(
            modal.Mount.from_local_dir(config.NANOCHAT_DIR, remote_path=REMOTE_NANOCHAT_REPO, copy=True)
        )


def _inject_paths() -> None:
    if REMOTE_SRC_DIR not in sys.path:
        sys.path.insert(0, REMOTE_SRC_DIR)
    if REMOTE_NANOCHAT_REPO not in sys.path:
        sys.path.insert(0, REMOTE_NANOCHAT_REPO)


def _load_tokenizer_only(init_tokenizer: bool = False):
    """Load only the tokenizer needed for pretraining, without loading finetune datasets."""
    if init_tokenizer:
        from chat_model.datasets import create_tokenizer

        create_tokenizer()

    try:
        import nanochat  # noqa: F401
    except ImportError:
        nanochat_path = os.path.join(config.PROJECT_ROOT, "..", "nanochat")
        if os.path.exists(nanochat_path) and nanochat_path not in sys.path:
            sys.path.insert(0, nanochat_path)

    with open(config.TOKENIZER_PKL, "rb") as file:
        return pickle.load(file)


def _pretrain_artifact_specs():
    specs = []
    for local_path, remote_path in [
        (LOCAL_DATA_DIR / "pretrain_splits", "/root/input/pretrain_splits"),
        (LOCAL_DATA_DIR / "processed" / "tokenized", "/root/input/processed/tokenized"),
        (LOCAL_DATA_DIR / "tokenizer_text.txt", "/root/input/tokenizer_text.txt"),
    ]:
        if local_path.exists():
            specs.append((local_path, remote_path))
    return specs


def _build_pretrain_sync_image():
    sync_image = image
    for local_path, remote_path in _pretrain_artifact_specs():
        if local_path.is_dir():
            sync_image = sync_image.add_local_dir(str(local_path), remote_path=remote_path)
        else:
            sync_image = sync_image.add_local_file(str(local_path), remote_path=remote_path)
    return sync_image


sync_function_kwargs = {
    "image": sync_image if not HAS_MODAL_MOUNT else image,
    "timeout": 3600 * 4,
    "volumes": {REMOTE_DATA_DIR: volume},
}
if HAS_MODAL_MOUNT:
    sync_function_kwargs["mounts"] = sync_mounts


@app.function(**sync_function_kwargs)
def sync_data_to_volume() -> None:
    """One-time sync: copy local Project/data into Modal Volume mounted at /root/Project/data."""
    target = Path(REMOTE_DATA_DIR)
    source = Path("/root/local_data")
    target.mkdir(parents=True, exist_ok=True)

    print(f"Syncing local data from {source} -> {target}")
    shutil.copytree(source, target, dirs_exist_ok=True)

    volume.commit()
    print("Data sync complete. Volume committed.")


pretrain_sync_function_kwargs = {
    "image": _build_pretrain_sync_image() if not HAS_MODAL_MOUNT else image,
    "timeout": 3600 * 2,
    "volumes": {REMOTE_DATA_DIR: volume},
}


@app.function(**pretrain_sync_function_kwargs)
def sync_pretrain_data_to_volume() -> None:
    """Sync only the artifacts required for pretraining into the Modal volume."""
    target_root = Path(REMOTE_DATA_DIR)
    target_root.mkdir(parents=True, exist_ok=True)

    for local_path, remote_path in _pretrain_artifact_specs():
        relative_target = Path(remote_path.replace("/root/input/", ""))
        source = Path(remote_path)
        destination = target_root / relative_target
        destination.parent.mkdir(parents=True, exist_ok=True)
        print(f"Syncing pretrain artifact {source} -> {destination}")

        if source.is_dir():
            shutil.copytree(source, destination, dirs_exist_ok=True)
        else:
            shutil.copy2(source, destination)

    volume.commit()
    print("Pretraining artifacts synced and volume committed.")


train_function_kwargs = {
    "image": runtime_image if not HAS_MODAL_MOUNT else image,
    "gpu": "A10G",
    "timeout": 3600 * 12,
    "volumes": {REMOTE_DATA_DIR: volume},
}
if HAS_MODAL_MOUNT:
    train_function_kwargs["mounts"] = mounts


@app.function(**train_function_kwargs)
def pretrain_remote(load_data: bool = False, init_tokenizer: bool = False) -> None:
    """Run only Stage-1 pretraining on Modal using data from persistent volume."""
    _inject_paths()

    from chat_model.training.train import (
        build_dataloaders,
        ensure_pretraining_splits,
        initialize_model_params,
        train,
    )

    tokenizer = _load_tokenizer_only(init_tokenizer=init_tokenizer)

    ensure_pretraining_splits(load_data=load_data)

    from chat_model.datasets import loader as dl

    pretrain_train_dataset = dl.build_dataset("train", tokenizer, data_dir=config.PRETRAIN_SPLITS_DIR)
    pretrain_val_dataset = dl.build_dataset("val", tokenizer, data_dir=config.PRETRAIN_SPLITS_DIR)

    pretrain_train_loader, pretrain_val_loader = build_dataloaders(
        pretrain_train_dataset,
        pretrain_val_dataset,
        config.STAGE1_BATCH_SIZE,
    )

    pretrain_model, criterion, optimizer, scheduler, scaler = initialize_model_params(
        tokenizer,
        pretrain_train_loader,
        config.STAGE1_EPOCHS,
        config.STAGE1_LEARNING_RATE,
        config.STAGE1_WARMUP_STEPS,
    )

    _, pretrain_best_val_loss = train(
        model=pretrain_model,
        train_loader=pretrain_train_loader,
        val_loader=pretrain_val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scaler=scaler,
        scheduler=scheduler,
        device=config.DEVICE,
        epochs=config.STAGE1_EPOCHS,
        patience=config.PATIENCE,
        vocab_size=config.VOCAB_SIZE,
        model_dir=config.PRE_TRAINED_DIR,
        best_checkpoint_path=config.BEST_PRETRAINED_PTH,
        resume_from_latest=False,
    )

    print(f"Pretraining complete. Best val loss: {pretrain_best_val_loss:.4f}")
    volume.commit()


@app.function(**train_function_kwargs)
def finetune_remote(load_data: bool = False, init_tokenizer: bool = False) -> None:
    """Run only Stage-2 fine-tuning on Modal using data from persistent volume."""
    _inject_paths()

    from chat_model.training.train import (
        build_dataloaders,
        initialize_model_params,
        load_model_weights,
        setup_training,
        train,
    )

    tokenizer, train_dataset, val_dataset = setup_training(
        load_data=load_data,
        init_tokenizer=init_tokenizer,
    )

    finetune_train_loader, finetune_val_loader = build_dataloaders(
        train_dataset,
        val_dataset,
        config.STAGE2_BATCH_SIZE,
    )

    finetune_model, criterion, optimizer, scheduler, scaler = initialize_model_params(
        tokenizer,
        finetune_train_loader,
        config.STAGE2_EPOCHS,
        config.STAGE2_LEARNING_RATE,
        config.STAGE2_WARMUP_STEPS,
    )

    pretrained_path = Path(config.BEST_PRETRAINED_PTH)
    if not pretrained_path.exists():
        raise FileNotFoundError(
            f"Missing pretrained checkpoint at {pretrained_path}. "
            "Run `--mode pretrain` first (or `--mode all`)."
        )

    load_model_weights(str(pretrained_path), finetune_model, config.DEVICE)

    freeze_blocks = max(0, len(finetune_model.blocks) - 2)

    _, best_val_loss = train(
        model=finetune_model,
        train_loader=finetune_train_loader,
        val_loader=finetune_val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scaler=scaler,
        scheduler=scheduler,
        device=config.DEVICE,
        epochs=config.STAGE2_EPOCHS,
        patience=config.PATIENCE,
        vocab_size=config.VOCAB_SIZE,
        model_dir=config.CHECKPOINTS_DIR,
        best_checkpoint_path=config.BEST_MODEL_PTH,
        resume_from_latest=False,
        freeze_epochs=2,
        freeze_blocks=freeze_blocks,
    )

    print(f"Fine-tuning complete. Best val loss: {best_val_loss:.4f}")
    volume.commit()


@app.local_entrypoint()
def run(mode: str = "pretrain", load_data: bool = False, init_tokenizer: bool = False) -> None:
    """
    mode:
      - sync      : Upload local Project/data to Modal volume once
      - pretrain  : Run Stage-1 pretraining from volume-backed data
            - finetune  : Run Stage-2 fine-tuning from volume-backed data
            - all       : Sync data, then run pretraining + fine-tuning

    Examples:
      modal run scripts/run_on_modal.py --mode sync
      modal run scripts/run_on_modal.py --mode pretrain
      modal run scripts/run_on_modal.py --mode finetune
      modal run scripts/run_on_modal.py --mode all
    """
    if mode not in {"sync", "pretrain", "finetune", "all"}:
        raise ValueError("mode must be one of: sync, pretrain, finetune, all")

    if mode in {"sync", "all"}:
        sync_data_to_volume.call()

    if mode in {"pretrain", "all"}:
        sync_pretrain_data_to_volume.call()
        pretrain_remote.call(load_data=load_data, init_tokenizer=init_tokenizer)

    if mode in {"finetune", "all"}:
        finetune_remote.call(load_data=load_data, init_tokenizer=init_tokenizer)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Modal sync / pretraining / fine-tuning")
    parser.add_argument("--mode", choices=["sync", "pretrain", "finetune", "all"], default="pretrain")
    parser.add_argument("--load-data", action="store_true", default=False)
    parser.add_argument("--init-tokenizer", action="store_true", default=False)
    args = parser.parse_args()

    run(mode=args.mode, load_data=args.load_data, init_tokenizer=args.init_tokenizer)
