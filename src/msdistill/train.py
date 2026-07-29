"""Training engine: teacher fine-tuning (stage 0) and student distillation (stage 1).

Two design decisions here are worth calling out.

Fold-matched teachers
---------------------
A separate teacher is fine-tuned for every cross-validation fold, on exactly that
fold's training patients. Reusing one teacher across folds would mean distilling
into fold *k* using a teacher that had already trained on fold *k*'s validation
patients, which inflates the apparent benefit of distillation on precisely the
cases used to measure it.

Co-trained ablation variants
----------------------------
All student variants for a fold are trained in a single pass over the data. Each
variant keeps its own parameters, optimizer and scheduler, but they share the
data loader and -- critically -- a single teacher forward pass per batch. The
teacher is roughly fifteen times more expensive than the student, so sharing it
turns a cost of `n_variants x (teacher + student)` into `teacher + n_variants x
student`, about a 4x saving for the seven-row grid.

The scientific benefit matters more than the speed: every variant sees the
identical sequence of batches with the identical augmentation, so differences
between rows cannot be attributed to data ordering. It also makes the resulting
per-patient scores properly paired for significance testing.
"""

from __future__ import annotations

import json
import math
import os
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import Config, Variant
from .data import PatientVolumeCache, make_dataloaders
from .losses import DistillationCriterion, HardLabelLoss
from .models.student import (PooledFeatureProjector, SpatialFeatureProjector,
                             build_student, count_parameters)
from .models.teacher import FrozenTeacher, build_teacher


# --------------------------------------------------------------------------- #
# Small utilities
# --------------------------------------------------------------------------- #

# Three resources are genuinely shared when folds run concurrently, and each
# needs its own lock:
#   * the global RNG, which `set_seed` mutates before every model construction
#   * the per-variant results JSON, which every fold appends to
#   * stdout, so interleaved progress lines stay readable
# Everything else -- the volume cache, the models, the optimizers -- is either
# read-only or owned by exactly one fold.
_RNG_LOCK = threading.Lock()
_RESULTS_LOCK = threading.Lock()
_PRINT_LOCK = threading.Lock()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def available_devices(limit: int = 0) -> List[torch.device]:
    """Devices to spread folds across.

    `MSDISTILL_FAKE_DEVICES=n` pretends there are n devices so the concurrent
    code path -- threads, locks, result merging -- can be exercised on a machine
    with one GPU or none. It exists because shipping untested concurrency is
    worse than a small testing seam; it has no effect unless the variable is set.
    """
    fake = os.environ.get("MSDISTILL_FAKE_DEVICES")
    if fake:
        base = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        devices = [base] * int(fake)
    elif torch.cuda.is_available():
        devices = [torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count())]
    else:
        devices = [torch.device("cpu")]
    return devices[:limit] if limit else devices


def plan_fold_devices(n_folds: int, cfg: Config) -> List[torch.device]:
    """Assign a device to each fold, round-robin over the requested width."""
    width = max(1, int(cfg.parallel_folds))
    devices = available_devices(width)
    if width > len(devices):
        _write(f"  requested parallel_folds={width} but only {len(devices)} device(s) "
               f"are visible; using {len(devices)}")
        width = len(devices)
    return [devices[i % len(devices)] for i in range(n_folds)]


def make_grad_scaler(enabled: bool):
    """`torch.amp.GradScaler` changed signature across releases; support both."""
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except TypeError:  # pragma: no cover - older torch
        return torch.cuda.amp.GradScaler(enabled=enabled)


def autocast_ctx(enabled: bool, device: torch.device):
    return torch.amp.autocast(device_type=device.type, enabled=enabled and device.type == "cuda")


def _tqdm(iterable, enabled: bool = True, **kwargs):
    if not enabled:
        return iterable
    try:
        from tqdm.auto import tqdm
        return tqdm(iterable, **kwargs)
    except ImportError:
        return iterable


def _write(msg: str) -> None:
    with _PRINT_LOCK:
        try:
            from tqdm.auto import tqdm
            tqdm.write(msg)
        except ImportError:
            print(msg)


class WarmupCosine:
    """Linear warmup then cosine decay, stepped once per epoch.

    Applied to every parameter group as a multiplicative factor on its own base
    learning rate, so the teacher's differential encoder/head rates keep their
    ratio throughout training.
    """

    def __init__(self, optimizer: torch.optim.Optimizer, warmup_epochs: int,
                 total_epochs: int, min_factor: float = 0.02):
        self.optimizer = optimizer
        self.warmup = max(int(warmup_epochs), 0)
        self.total = max(int(total_epochs), self.warmup + 1)
        self.min_factor = min_factor
        self.base_lrs = [g["lr"] for g in optimizer.param_groups]
        self.last_epoch = -1
        self.step(0)

    def factor(self, epoch: int) -> float:
        if epoch < self.warmup:
            return (epoch + 1) / float(self.warmup + 1)
        progress = (epoch - self.warmup) / max(1.0, self.total - self.warmup)
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.min_factor + (1.0 - self.min_factor) * cosine

    def step(self, epoch: int) -> None:
        self.last_epoch = epoch
        f = self.factor(epoch)
        for group, base in zip(self.optimizer.param_groups, self.base_lrs):
            group["lr"] = base * f

    def state_dict(self) -> Dict:
        return {"last_epoch": self.last_epoch, "base_lrs": self.base_lrs}

    def load_state_dict(self, state: Dict) -> None:
        self.base_lrs = state["base_lrs"]
        self.step(state["last_epoch"])


def cpu_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


# --------------------------------------------------------------------------- #
# Volume-level validation
# --------------------------------------------------------------------------- #

@torch.no_grad()
def predict_volume_probs(models: Sequence[nn.Module], cache: PatientVolumeCache, pid: str,
                         cfg: Config, device: torch.device,
                         batch_size: Optional[int] = None) -> np.ndarray:
    """Softmax probabilities for one patient volume, averaged over `models`.

    Averaging probabilities rather than logits is what "ensemble" means for the
    fold ensembles reported in the paper; logit averaging is not equivalent when
    the members disagree about scale.
    """
    batch_size = batch_size or cfg.batch_size
    images, _ = cache.get_volume(pid)              # (S, M, H, W)
    n_slices = images.shape[0]
    accum = np.zeros((n_slices, cfg.num_classes, *cfg.target_size), dtype=np.float32)

    for model in models:
        model.eval()
    for start in range(0, n_slices, batch_size):
        batch = torch.from_numpy(images[start:start + batch_size]).float().to(device)
        probs = None
        for model in models:
            out = model(batch)
            logits = out.logits if hasattr(out, "logits") else out
            p = F.softmax(logits.float(), dim=1)
            probs = p if probs is None else probs + p
        accum[start:start + batch_size] = (probs / len(models)).cpu().numpy()
    return accum


@torch.no_grad()
def predict_volume_labels(models: Sequence[nn.Module], cache: PatientVolumeCache, pid: str,
                          cfg: Config, device: torch.device) -> np.ndarray:
    """(H, W, S) integer label volume, matching the reference mask layout."""
    probs = predict_volume_probs(models, cache, pid, cfg, device)   # (S, C, H, W)
    labels = probs.argmax(axis=1)                                   # (S, H, W)
    return np.transpose(labels, (1, 2, 0))


@torch.no_grad()
def validation_dice(model: nn.Module, cache: PatientVolumeCache, val_ids: Sequence[str],
                    cfg: Config, device: torch.device) -> Tuple[float, Dict[str, float]]:
    """Mean per-patient foreground Dice over the validation patients.

    This is the model-selection criterion. It is computed the same way as the
    reported test metric -- per patient, over the whole volume, with classes
    absent from a patient excluded rather than scored as perfect. Selecting
    checkpoints on a per-batch average instead would optimize for a quantity the
    paper does not report.
    """
    from .metrics import dice_binary

    per_class: Dict[str, List[float]] = {name: [] for name in cfg.class_names}
    for pid in val_ids:
        pred = predict_volume_labels([model], cache, pid, cfg, device)
        ref = cache.masks[pid]
        for c, name in enumerate(cfg.class_names):
            per_class[name].append(dice_binary(pred == c, ref == c))

    means: Dict[str, float] = {}
    for name, values in per_class.items():
        arr = np.asarray(values, dtype=np.float64)
        defined = arr[~np.isnan(arr)]
        means[name] = float(defined.mean()) if defined.size else float("nan")

    fg = [means[cfg.class_names[i]] for i in cfg.foreground_classes]
    fg = [v for v in fg if not math.isnan(v)]
    return (float(np.mean(fg)) if fg else 0.0), means


# --------------------------------------------------------------------------- #
# Result bookkeeping with a configuration fingerprint
# --------------------------------------------------------------------------- #

def _results_path(cfg: Config, exp_name: str) -> str:
    return os.path.join(cfg.results_dir, f"{exp_name}_cv.json")


def load_cached_results(cfg: Config, exp_name: str) -> List[Dict]:
    """Load previously completed folds, but only if they came from this config.

    Silently reusing folds trained under different hyperparameters is how our
    earlier run ended up reporting byte-identical validation scores for two
    different training objectives. The fingerprint check makes that impossible:
    a mismatched cache is discarded rather than merged.
    """
    path = _results_path(cfg, exp_name)
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)
    if payload.get("fingerprint") != cfg.fingerprint():
        print(f"  [{exp_name}] cached results were produced by a different configuration "
              f"({payload.get('fingerprint')} != {cfg.fingerprint()}); ignoring them "
              f"and retraining.")
        return []
    return payload.get("folds", [])


def save_results(cfg: Config, exp_name: str, folds: List[Dict]) -> None:
    """Write atomically and under a lock.

    Concurrent folds append to the same per-variant file. The lock serialises
    the read-modify-write, and the rename makes a killed session leave either
    the old file or the new one, never a truncated one.
    """
    path = _results_path(cfg, exp_name)
    payload = {"fingerprint": cfg.fingerprint(), "exp_name": exp_name,
               "folds": sorted(folds, key=lambda r: r["fold"])}
    with _RESULTS_LOCK:
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, default=str)
        os.replace(tmp, path)


def checkpoint_path(cfg: Config, exp_name: str, fold: int, seed: Optional[int] = None) -> str:
    suffix = "" if seed is None else f"_seed{seed}"
    return os.path.join(cfg.checkpoint_dir, f"{exp_name}_fold{fold}{suffix}_best.pt")


# --------------------------------------------------------------------------- #
# Stage 0: teacher
# --------------------------------------------------------------------------- #

def train_teacher_fold(cfg: Config, fold: int, train_ids: Sequence[str], val_ids: Sequence[str],
                       cache: PatientVolumeCache, device: torch.device) -> Dict:
    set_seed(cfg.seed + fold)
    ckpt = checkpoint_path(cfg, "teacher", fold)
    train_loader, _ = make_dataloaders(train_ids, val_ids, cache, cfg, seed=cfg.seed + fold)

    model = build_teacher(cfg, pretrained=True).to(device)
    n_params = count_parameters(model)
    _write(f"\n=== Teacher (MiT-{cfg.teacher_variant.upper()}"
           f"{'+HR' if cfg.teacher_hr_decoder else ''}) | fold {fold} | "
           f"{n_params:,} params | train={len(train_ids)} val={len(val_ids)} ===")

    # Pretrained encoder gets the low rate; the freshly initialized decoder,
    # refinement head and classifier get ten times more.
    pretrained_params, fresh_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (pretrained_params if name.startswith("encoder.") else fresh_params).append(p)
    groups = [{"params": pretrained_params, "lr": cfg.teacher_lr}]
    if fresh_params:
        groups.append({"params": fresh_params, "lr": cfg.teacher_head_lr})

    optimizer = torch.optim.AdamW(groups, weight_decay=cfg.teacher_weight_decay)
    scheduler = WarmupCosine(optimizer, cfg.teacher_warmup_epochs, cfg.teacher_epochs)
    scaler = make_grad_scaler(cfg.use_amp and device.type == "cuda")
    criterion = HardLabelLoss(cfg).to(device)

    best_dice, best_state, patience = -1.0, None, 0
    history: Dict[str, List] = {"train_loss": [], "val_dice": [], "val_per_class": []}
    started = time.time()

    epochs = _tqdm(range(cfg.teacher_epochs), cfg.progress, desc=f"teacher f{fold}", unit="ep")
    for epoch in epochs:
        train_loader.dataset.set_epoch(epoch)
        scheduler.step(epoch)
        model.train()

        running, seen = 0.0, 0
        for images, masks in _tqdm(train_loader, cfg.progress, desc="  train", leave=False):
            images, masks = images.to(device, non_blocking=True), masks.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast_ctx(cfg.use_amp, device):
                loss = criterion(model(images), masks)
            scaler.scale(loss).backward()
            if cfg.grad_clip_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
            running += float(loss.detach()) * images.size(0)
            seen += images.size(0)

        train_loss = running / max(seen, 1)
        val_dice, per_class = validation_dice(model, cache, val_ids, cfg, device)
        history["train_loss"].append(train_loss)
        history["val_dice"].append(val_dice)
        history["val_per_class"].append(per_class)

        if val_dice > best_dice:
            best_dice, best_state, patience = val_dice, cpu_state_dict(model), 0
        else:
            patience += 1

        if hasattr(epochs, "set_postfix"):
            epochs.set_postfix(loss=f"{train_loss:.3f}", val=f"{val_dice:.3f}",
                               best=f"{best_dice:.3f}", pat=f"{patience}/{cfg.teacher_early_stop_patience}")
        if epoch % 5 == 0:
            _write("  ep {:>3} loss={:.4f} valDice={:.4f} | ".format(epoch, train_loss, val_dice)
                   + ", ".join(f"{k}={v:.3f}" for k, v in per_class.items()))

        elapsed_h = (time.time() - started) / 3600.0
        if elapsed_h >= cfg.max_hours_per_fold:
            _write(f"  wall-clock budget reached ({elapsed_h:.2f}h); stopping fold {fold}")
            break
        if patience >= cfg.teacher_early_stop_patience:
            _write(f"  early stop at epoch {epoch}")
            break

    if best_state is None:
        best_state = cpu_state_dict(model)
    torch.save(best_state, ckpt)
    _write(f"  teacher fold {fold}: best val Dice {best_dice:.4f} -> {ckpt}")

    del model, optimizer
    torch.cuda.empty_cache() if device.type == "cuda" else None
    return {"fold": fold, "exp_name": "teacher", "best_val_dice": best_dice,
            "checkpoint": ckpt, "history": history, "n_params": n_params}


def run_teacher_stage(cfg: Config, fold_splits: List[Dict[str, List[str]]],
                      cache: PatientVolumeCache, device: torch.device) -> List[Dict]:
    """Fine-tune one teacher per fold, optionally several folds at once.

    With `cfg.parallel_folds > 1` the pending folds are spread one per GPU. They
    share nothing but the read-only volume cache, so this scales close to
    linearly until it runs out of either folds or devices.
    """
    results = load_cached_results(cfg, "teacher")
    done = {r["fold"] for r in results if os.path.exists(r.get("checkpoint", ""))}
    pending = [(i, s) for i, s in enumerate(fold_splits) if i not in done]
    for fold in sorted(done):
        _write(f"  teacher fold {fold}: cached, skipping")

    if pending:
        fold_devices = plan_fold_devices(len(fold_splits), cfg)
        width = min(max(1, int(cfg.parallel_folds)), len(available_devices()), len(pending))

        def run_one(item):
            fold, split = item
            target = fold_devices[fold] if width > 1 else device
            if width > 1:
                _write(f"  teacher fold {fold} -> {target}")
            return train_teacher_fold(cfg, fold, split["train"], split["val"], cache, target)

        def record(result: Dict) -> None:
            """Persist as each fold lands, so an interrupted stage resumes."""
            results[:] = [x for x in results if x["fold"] != result["fold"]] + [result]
            save_results(cfg, "teacher", results)

        if width > 1:
            _write(f"\nrunning {len(pending)} teacher fold(s) across {width} device(s)")
            with ThreadPoolExecutor(max_workers=width) as pool:
                # `record` runs in this thread as each result is yielded, so the
                # results list is only ever mutated from one place.
                for result in pool.map(run_one, pending):
                    record(result)
        else:
            for item in pending:
                record(run_one(item))

    results = sorted(results, key=lambda r: r["fold"])
    scores = [r["best_val_dice"] for r in results]
    _write(f"\nTeacher CV: {np.mean(scores):.4f} +/- {np.std(scores):.4f}")
    return results


# --------------------------------------------------------------------------- #
# Stage 1: co-trained student variants
# --------------------------------------------------------------------------- #

@dataclass
class VariantState:
    variant: Variant
    model: nn.Module
    criterion: DistillationCriterion
    optimizer: torch.optim.Optimizer
    scheduler: WarmupCosine
    scaler: object
    spatial_projector: Optional[nn.Module] = None
    pooled_projector: Optional[nn.Module] = None
    best_dice: float = -1.0
    best_state: Optional[Dict] = None
    patience: int = 0
    stopped: bool = False
    history: Dict[str, List] = field(default_factory=lambda: {"train_loss": [], "val_dice": [],
                                                              "val_per_class": [], "components": []})


def _build_variant_state(cfg: Config, variant: Variant, teacher: Optional[FrozenTeacher],
                         device: torch.device, seed: int) -> VariantState:
    # Same seed for every variant so all students of a given width start from
    # identical weights; any divergence between rows is then caused by the
    # objective alone. Seeding and construction must be atomic: `set_seed`
    # touches the global RNG, so a concurrent fold interleaving between the two
    # would silently give one variant a different initialization.
    with _RNG_LOCK:
        set_seed(seed)
        model = build_student(cfg, variant.base_channels).to(device)
    criterion = DistillationCriterion(cfg, variant.terms).to(device)

    params = [{"params": list(model.parameters()), "lr": cfg.student_lr}]
    spatial_projector = pooled_projector = None
    if criterion.needs_spatial_feature:
        if teacher is None:
            raise ValueError(f"{variant.key} needs a teacher for spatial feature distillation")
        spatial_projector = SpatialFeatureProjector(model.feature_dim, teacher.feature_dim).to(device)
        params.append({"params": list(spatial_projector.parameters()), "lr": cfg.student_lr})
    if criterion.needs_pooled_feature:
        if teacher is None:
            raise ValueError(f"{variant.key} needs a teacher for pooled feature distillation")
        pooled_projector = PooledFeatureProjector(model.pooled_dim, teacher.pooled_dim).to(device)
        params.append({"params": list(pooled_projector.parameters()), "lr": cfg.student_lr})

    optimizer = torch.optim.AdamW(params, weight_decay=cfg.student_weight_decay)
    scheduler = WarmupCosine(optimizer, cfg.student_warmup_epochs, cfg.num_epochs)
    scaler = make_grad_scaler(cfg.use_amp and device.type == "cuda")
    return VariantState(variant=variant, model=model, criterion=criterion, optimizer=optimizer,
                        scheduler=scheduler, scaler=scaler, spatial_projector=spatial_projector,
                        pooled_projector=pooled_projector)


def _snapshot_path(cfg: Config, fold: int, seed: int) -> str:
    return os.path.join(cfg.checkpoint_dir, f"students_fold{fold}_seed{seed}_snapshot.pt")


def _save_snapshot(cfg: Config, fold: int, seed: int, epoch: int,
                   states: Dict[str, "VariantState"]) -> None:
    """Persist every variant's full training state mid-fold.

    A Kaggle session can be killed at any point. Without this, an interruption
    partway through a 60-epoch co-trained fold discards every variant's progress
    at once -- the cost of sharing one pass across the grid is that they also
    fail together.
    """
    payload = {
        "epoch": epoch,
        "fingerprint": cfg.fingerprint(),
        "variants": {
            key: {
                "model": s.model.state_dict(),
                "optimizer": s.optimizer.state_dict(),
                "scaler": s.scaler.state_dict(),
                "spatial_projector": (s.spatial_projector.state_dict()
                                      if s.spatial_projector is not None else None),
                "pooled_projector": (s.pooled_projector.state_dict()
                                     if s.pooled_projector is not None else None),
                "best_dice": s.best_dice,
                "best_state": s.best_state,
                "patience": s.patience,
                "stopped": s.stopped,
                "history": s.history,
            }
            for key, s in states.items()
        },
    }
    tmp = _snapshot_path(cfg, fold, seed) + ".tmp"
    torch.save(payload, tmp)
    os.replace(tmp, _snapshot_path(cfg, fold, seed))   # atomic; a killed write cannot corrupt


def _load_snapshot(cfg: Config, fold: int, seed: int,
                   states: Dict[str, "VariantState"]) -> int:
    """Restore mid-fold state. Returns the epoch to resume from (0 if none)."""
    path = _snapshot_path(cfg, fold, seed)
    if not os.path.exists(path):
        return 0
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("fingerprint") != cfg.fingerprint():
        _write(f"  snapshot for fold {fold} came from a different configuration; ignoring")
        return 0
    if set(payload.get("variants", {})) != set(states):
        _write(f"  snapshot for fold {fold} covers a different variant set; ignoring")
        return 0

    for key, saved in payload["variants"].items():
        s = states[key]
        s.model.load_state_dict(saved["model"])
        s.optimizer.load_state_dict(saved["optimizer"])
        s.scaler.load_state_dict(saved["scaler"])
        if saved["spatial_projector"] is not None and s.spatial_projector is not None:
            s.spatial_projector.load_state_dict(saved["spatial_projector"])
        if saved["pooled_projector"] is not None and s.pooled_projector is not None:
            s.pooled_projector.load_state_dict(saved["pooled_projector"])
        s.best_dice = saved["best_dice"]
        s.best_state = saved["best_state"]
        s.patience = saved["patience"]
        s.stopped = saved["stopped"]
        s.history = saved["history"]

    start = int(payload["epoch"]) + 1
    _write(f"  resuming fold {fold} from epoch {start}")
    return start


def _clear_snapshot(cfg: Config, fold: int, seed: int) -> None:
    path = _snapshot_path(cfg, fold, seed)
    if os.path.exists(path):
        os.remove(path)


def train_student_variants_fold(cfg: Config, fold: int, variants: Sequence[Variant],
                                train_ids: Sequence[str], val_ids: Sequence[str],
                                cache: PatientVolumeCache, teacher: Optional[FrozenTeacher],
                                device: torch.device, seed: Optional[int] = None) -> Dict[str, Dict]:
    """Train every ablation variant for one fold in a single shared pass."""
    seed = cfg.seed if seed is None else seed
    variants = list(variants)
    if any(v.needs_teacher for v in variants) and teacher is None:
        raise ValueError("at least one variant requires a teacher but none was provided")

    train_loader, _ = make_dataloaders(train_ids, val_ids, cache, cfg, seed=seed + fold)
    states = {v.key: _build_variant_state(cfg, v, teacher, device, seed + fold) for v in variants}
    start_epoch = _load_snapshot(cfg, fold, seed, states)

    needs_teacher_logits = any(s.criterion.uses_teacher for s in states.values())
    needs_teacher_feat = any(s.criterion.needs_spatial_feature or s.criterion.needs_pooled_feature
                             for s in states.values())

    _write(f"\n=== Students | fold {fold} | seed {seed} | {len(variants)} variants | "
           f"train={len(train_ids)} val={len(val_ids)} ===")
    _write("    " + ", ".join(v.key for v in variants))

    started = time.time()
    # A fold is complete when every variant has early-stopped or the epoch budget
    # is exhausted. Being cut short by the wall-clock guard is *not* completion,
    # and must not let `run_student_stage` skip the fold on the next run.
    completed = True
    epochs = _tqdm(range(start_epoch, cfg.num_epochs), cfg.progress,
                   desc=f"students f{fold}", unit="ep")
    for epoch in epochs:
        train_loader.dataset.set_epoch(epoch)
        active = [s for s in states.values() if not s.stopped]
        if not active:
            break
        for state in active:
            state.scheduler.step(epoch)
            state.criterion.set_epoch(epoch)
            state.model.train()

        running = {s.variant.key: 0.0 for s in active}
        components = {s.variant.key: {} for s in active}
        seen = 0

        for images, masks in _tqdm(train_loader, cfg.progress, desc="  train", leave=False):
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            teacher_logits = teacher_feat = teacher_pooled = None
            if needs_teacher_logits:
                with torch.no_grad(), autocast_ctx(cfg.use_amp, device):
                    if needs_teacher_feat:
                        out = teacher(images, return_features=True)
                        teacher_logits, teacher_feat, teacher_pooled = out.logits, out.feat, out.pooled
                    else:
                        teacher_logits = teacher(images)
                teacher_logits = teacher_logits.detach()
                if teacher_feat is not None:
                    teacher_feat = teacher_feat.detach()
                if teacher_pooled is not None:
                    teacher_pooled = teacher_pooled.detach()

            for state in active:
                state.optimizer.zero_grad(set_to_none=True)
                need_feats = state.criterion.needs_spatial_feature or state.criterion.needs_pooled_feature
                with autocast_ctx(cfg.use_amp, device):
                    if need_feats:
                        out = state.model(images, return_features=True)
                        s_logits, s_feat, s_pooled = out.logits, out.feat, out.pooled
                    else:
                        s_logits, s_feat, s_pooled = state.model(images), None, None

                    s_feat_proj = (state.spatial_projector(s_feat)
                                   if state.spatial_projector is not None else None)
                    s_pooled_proj = (state.pooled_projector(s_pooled)
                                     if state.pooled_projector is not None else None)

                    loss, parts = state.criterion(
                        s_logits, masks,
                        teacher_logits=teacher_logits if state.criterion.uses_teacher else None,
                        student_feat_projected=s_feat_proj, teacher_feat=teacher_feat,
                        student_pooled_projected=s_pooled_proj, teacher_pooled=teacher_pooled,
                    )

                state.scaler.scale(loss).backward()
                if cfg.grad_clip_norm > 0:
                    state.scaler.unscale_(state.optimizer)
                    torch.nn.utils.clip_grad_norm_(state.model.parameters(), cfg.grad_clip_norm)
                state.scaler.step(state.optimizer)
                state.scaler.update()

                running[state.variant.key] += float(loss.detach()) * images.size(0)
                for k, v in parts.items():
                    components[state.variant.key][k] = v
            seen += images.size(0)

        # ---- validation ------------------------------------------------
        for state in active:
            train_loss = running[state.variant.key] / max(seen, 1)
            val_dice, per_class = validation_dice(state.model, cache, val_ids, cfg, device)
            state.history["train_loss"].append(train_loss)
            state.history["val_dice"].append(val_dice)
            state.history["val_per_class"].append(per_class)
            state.history["components"].append(components[state.variant.key])

            if val_dice > state.best_dice:
                state.best_dice = val_dice
                state.best_state = cpu_state_dict(state.model)
                state.patience = 0
            else:
                state.patience += 1
                if state.patience >= cfg.early_stop_patience:
                    state.stopped = True
                    _write(f"  [{state.variant.key}] early stop at epoch {epoch} "
                           f"(best {state.best_dice:.4f})")

        if epoch % 5 == 0 or epoch == cfg.num_epochs - 1:
            summary = " | ".join(f"{s.variant.key}={s.history['val_dice'][-1]:.3f}" for s in active)
            _write(f"  ep {epoch:>3} valDice: {summary}")
        if hasattr(epochs, "set_postfix"):
            epochs.set_postfix(best=", ".join(f"{s.variant.key[:9]}:{s.best_dice:.3f}"
                                              for s in states.values()))

        if cfg.resume_every_epochs and (epoch + 1) % cfg.resume_every_epochs == 0:
            _save_snapshot(cfg, fold, seed, epoch, states)

        elapsed_h = (time.time() - started) / 3600.0
        if elapsed_h >= cfg.max_hours_per_fold:
            _save_snapshot(cfg, fold, seed, epoch, states)
            _write(f"  wall-clock budget reached ({elapsed_h:.2f}h) at epoch {epoch}; "
                   f"snapshot saved -- rerun the cell to continue this fold")
            completed = False
            break

    results: Dict[str, Dict] = {}
    for key, state in states.items():
        path = checkpoint_path(cfg, key, fold, seed if seed != cfg.seed else None)
        torch.save(state.best_state or cpu_state_dict(state.model), path)
        results[key] = {"fold": fold, "exp_name": key, "seed": seed,
                        "best_val_dice": state.best_dice, "checkpoint": path,
                        "history": state.history, "complete": completed}
        status = "" if completed else "  (INCOMPLETE -- will resume)"
        _write(f"  [{key}] fold {fold}: best val Dice {state.best_dice:.4f} -> {path}{status}")

    if completed:
        _clear_snapshot(cfg, fold, seed)
    del states
    torch.cuda.empty_cache() if device.type == "cuda" else None
    return results


def run_student_stage(cfg: Config, variants: Sequence[Variant],
                      fold_splits: List[Dict[str, List[str]]], cache: PatientVolumeCache,
                      teachers: Dict[int, FrozenTeacher], device: torch.device,
                      seed: Optional[int] = None) -> Dict[str, List[Dict]]:
    """Run every fold, returning {variant_key: [fold_result, ...]}."""
    seed = cfg.seed if seed is None else seed
    tag = "" if seed == cfg.seed else f"_seed{seed}"
    per_variant: Dict[str, List[Dict]] = {
        v.key: load_cached_results(cfg, v.key + tag) for v in variants
    }

    pending = []
    for fold, split in enumerate(fold_splits):
        # `complete` must be checked, not just checkpoint existence: a fold cut
        # short by the wall-clock guard writes its best-so-far checkpoints but is
        # not finished, and skipping it would silently report a truncated run.
        done = all(
            any(r["fold"] == fold and r.get("complete", True)
                and os.path.exists(r.get("checkpoint", ""))
                for r in per_variant[v.key])
            for v in variants
        )
        if done:
            _write(f"  students fold {fold}: cached, skipping")
        else:
            pending.append((fold, split))

    if pending:
        fold_devices = plan_fold_devices(len(fold_splits), cfg)
        width = min(max(1, int(cfg.parallel_folds)), len(available_devices()), len(pending))

        def run_one(item):
            fold, split = item
            target = fold_devices[fold] if width > 1 else device
            teacher = teachers.get(fold)
            if teacher is not None and width > 1:
                # Each fold owns its own teacher instance, so moving rather than
                # copying is safe and avoids a second 110 MB of weights.
                teacher = teacher.to(target)
                _write(f"  students fold {fold} -> {target}")
            return fold, train_student_variants_fold(
                cfg, fold, variants, split["train"], split["val"], cache,
                teacher, target, seed=seed,
            )

        def record(fold: int, fold_results: Dict[str, Dict]) -> None:
            for key, result in fold_results.items():
                per_variant[key] = [r for r in per_variant[key] if r["fold"] != fold]
                per_variant[key].append(result)
                save_results(cfg, key + tag, per_variant[key])

        if width > 1:
            _write(f"\nrunning {len(pending)} student fold(s) across {width} device(s); "
                   f"per-epoch logs from different folds will interleave")
            with ThreadPoolExecutor(max_workers=width) as pool:
                for fold, fold_results in pool.map(run_one, pending):
                    record(fold, fold_results)
        else:
            for item in pending:
                record(*run_one(item))

    for key, results in per_variant.items():
        scores = [r["best_val_dice"] for r in sorted(results, key=lambda r: r["fold"])]
        if scores:
            _write(f"  {key:<16} CV val Dice {np.mean(scores):.4f} +/- {np.std(scores):.4f}")
    return {k: sorted(v, key=lambda r: r["fold"]) for k, v in per_variant.items()}
