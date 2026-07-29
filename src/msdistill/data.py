"""Dataset discovery, preprocessing and loaders for MS3SEG.

Three things here differ materially from a naive implementation and each of them
changed the measured numbers:

1. Label remapping is *inferred* from the data rather than hard-coded. The
   distributed masks store four labels as widely spaced 16-bit values, and the
   exact values differ between mirrors of the dataset.
2. Intensity normalization uses a brain mask derived per volume, so the large
   air background does not dominate the percentile statistics.
3. Augmentation is restricted to anatomically plausible transforms. Arbitrary
   90-degree rotations of an axial brain slice produce configurations that never
   occur in acquisition and measurably cost accuracy on the rare classes.
"""

from __future__ import annotations

import math
import os
import random
import re
from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

try:
    import nibabel as nib
except ImportError:  # pragma: no cover - surfaced with a clear message at call time
    nib = None


# --------------------------------------------------------------------------- #
# Discovery
# --------------------------------------------------------------------------- #

_MODALITY_PATTERNS = (
    ("T1", ("t1wi", "t1_", "_t1", "t1w")),
    ("T2", ("t2wi", "t2_", "_t2", "t2w")),
    ("FLAIR", ("flair",)),
)


def classify_modality(filename: str) -> Optional[str]:
    """Map a NIfTI filename to a modality key, or None if it is not an input image."""
    lower = filename.lower()
    # Brain masks and skull-stripped derivatives live alongside the images.
    if "brain_mask" in lower or "brainmask" in lower:
        return None
    # FLAIR must be checked first: some mirrors name files "T2_FLAIR".
    if "flair" in lower:
        return "FLAIR"
    for modality, tokens in _MODALITY_PATTERNS:
        if modality == "FLAIR":
            continue
        if any(tok in lower for tok in tokens):
            return modality
    return None


def extract_patient_id(filename: str) -> str:
    stem = filename
    for ext in (".nii.gz", ".nii"):
        if stem.endswith(ext):
            stem = stem[: -len(ext)]
            break
    if stem.isdigit():
        return stem
    match = re.search(r"(\d{2,})", stem)
    return match.group(1) if match else stem


def find_dataset_roots(data_root: str) -> Tuple[Optional[str], Optional[str]]:
    """Locate the per-patient image directory and the mask directory.

    Searches for the canonical MS3SEG layout first, then falls back to a
    structural search so that a differently nested mirror still works.
    """
    canonical_images = os.path.join(data_root, "MS_100_patient_registered")
    canonical_masks = os.path.join(data_root, "MS_100_model_input", "man_4L_masks_new")
    if os.path.isdir(canonical_images) and os.path.isdir(canonical_masks):
        return canonical_images, canonical_masks

    images_root: Optional[str] = None
    masks_root: Optional[str] = None
    # followlinks=True is required: Kaggle exposes attached datasets under
    # /kaggle/input as symlinks, and os.walk does not descend into symlinked
    # directories by default -- so the default search silently finds nothing.
    for dirpath, _, _ in os.walk(data_root, followlinks=True):
        base = os.path.basename(dirpath)
        if base == "MS_100_patient_registered":
            images_root = dirpath
        elif base in {"man_4L_masks_new", "man_4L_masks"}:
            masks_root = dirpath
        if images_root and masks_root:
            break

    if images_root is None or masks_root is None:
        # Last resort: a directory of per-patient subfolders each holding several
        # NIfTI files is the images root; a flat directory of one NIfTI per
        # patient is the masks root.
        subdir_counts: Dict[str, int] = {}
        flat_dirs: List[Tuple[str, int]] = []
        for dirpath, _, filenames in os.walk(data_root, followlinks=True):
            nii = [f for f in filenames if f.endswith((".nii", ".nii.gz"))]
            if not nii:
                continue
            parent = os.path.dirname(dirpath)
            if len(nii) >= 3:
                subdir_counts[parent] = subdir_counts.get(parent, 0) + 1
            else:
                flat_dirs.append((dirpath, len(nii)))
            if len(nii) > 20:
                flat_dirs.append((dirpath, len(nii)))
        if images_root is None and subdir_counts:
            images_root = max(subdir_counts, key=subdir_counts.get)
        if masks_root is None and flat_dirs:
            masks_root = max(flat_dirs, key=lambda t: t[1])[0]

    return images_root, masks_root


def describe_tree(root: str, max_depth: int = 3, max_entries: int = 14) -> str:
    """A short listing of `root`, for telling the user what was actually found.

    A bare "dataset not found" is useless on a remote machine: the two causes --
    the dataset is not attached, and it is attached somewhere unexpected -- need
    completely different responses, and only a listing distinguishes them.
    """
    if not os.path.isdir(root):
        return f"  {root}  (does not exist)"

    lines: List[str] = []

    def walk(path: str, depth: int, prefix: str) -> None:
        if depth > max_depth or len(lines) > 60:
            return
        try:
            entries = sorted(os.listdir(path))
        except OSError as exc:
            lines.append(f"{prefix}<unreadable: {exc.strerror}>")
            return
        shown = entries[:max_entries]
        for name in shown:
            full = os.path.join(path, name)
            link = " -> symlink" if os.path.islink(full) else ""
            if os.path.isdir(full):
                lines.append(f"{prefix}{name}/{link}")
                walk(full, depth + 1, prefix + "  ")
            elif depth <= 2:
                lines.append(f"{prefix}{name}{link}")
        if len(entries) > max_entries:
            lines.append(f"{prefix}... and {len(entries) - max_entries} more")

    lines.append(f"{root}/")
    walk(root, 1, "  ")
    return "\n".join(lines)


def locate_dataset(preferred: Optional[str] = None,
                   search_bases: Sequence[str] = ("/kaggle/input", "/content", "data", "."),
                   verbose: bool = True) -> str:
    """Find the directory containing the MS3SEG layout, or raise with a listing.

    Tries `preferred` first, then searches each base for a directory holding
    `MS_100_patient_registered`. Symlinks are followed throughout.
    """
    if preferred and os.path.isdir(preferred):
        images, masks = find_dataset_roots(preferred)
        if images and masks:
            return preferred

    checked: List[str] = []
    for base in search_bases:
        if not os.path.isdir(base):
            continue
        checked.append(base)
        # Direct children first -- the common case, and far faster than a full walk.
        try:
            for name in sorted(os.listdir(base)):
                candidate = os.path.join(base, name)
                if not os.path.isdir(candidate):
                    continue
                images, masks = find_dataset_roots(candidate)
                if images and masks:
                    if verbose:
                        print(f"found MS3SEG at {candidate}")
                    return candidate
        except OSError:
            pass
        # Then a bounded recursive search.
        for dirpath, dirnames, _ in os.walk(base, followlinks=True):
            if os.path.relpath(dirpath, base).count(os.sep) > 4:
                dirnames[:] = []
                continue
            if "MS_100_patient_registered" in dirnames:
                images, masks = find_dataset_roots(dirpath)
                if images and masks:
                    if verbose:
                        print(f"found MS3SEG at {dirpath}")
                    return dirpath

    listing = "\n\n".join(describe_tree(b) for b in checked) if checked else \
        "  none of the search locations exist"
    raise FileNotFoundError(
        "MS3SEG not found.\n\n"
        f"Looked for a directory containing both 'MS_100_patient_registered/' and\n"
        f"'MS_100_model_input/man_4L_masks_new/' under: {list(search_bases)}\n"
        + (f"(and at the configured path {preferred!r})\n" if preferred else "")
        + "\nWhat is actually present:\n\n" + listing
        + "\n\nMost likely the dataset is not attached to this session. On Kaggle use\n"
          "'+ Add Input' in the right-hand panel and search for "
          "'ms3seg-mri-tri-mask-lesion-segmentation'.\n"
          "If it is attached under a name you can see above, set cfg.data_root to that\n"
          "path directly and rerun this cell."
    )


def build_patient_index(data_root: str, modalities: Sequence[str]) -> Dict[str, Dict[str, str]]:
    """Return {patient_id: {"T1": path, "T2": path, "FLAIR": path, "mask": path}}."""
    images_root, masks_root = find_dataset_roots(data_root)
    if images_root is None or masks_root is None:
        raise FileNotFoundError(
            f"Could not locate image/mask directories under {data_root!r}. "
            f"Found images_root={images_root!r}, masks_root={masks_root!r}. "
            "Point Config.data_root at the directory that contains "
            "'MS_100_patient_registered' and 'MS_100_model_input'."
        )

    index: Dict[str, Dict[str, str]] = defaultdict(dict)
    for pid in sorted(os.listdir(images_root)):
        pid_dir = os.path.join(images_root, pid)
        if not os.path.isdir(pid_dir):
            continue
        for fn in sorted(os.listdir(pid_dir)):
            if not fn.endswith((".nii", ".nii.gz")):
                continue
            modality = classify_modality(fn)
            if modality:
                index[pid][modality] = os.path.join(pid_dir, fn)

    for fn in sorted(os.listdir(masks_root)):
        if not fn.endswith((".nii", ".nii.gz")):
            continue
        index[extract_patient_id(fn)]["mask"] = os.path.join(masks_root, fn)

    complete = {
        pid: entry for pid, entry in index.items()
        if "mask" in entry and all(m in entry for m in modalities)
    }
    if not complete:
        raise FileNotFoundError(
            f"Indexed {len(index)} patients under {data_root!r} but none had all of "
            f"{list(modalities)} plus a mask."
        )
    return dict(sorted(complete.items()))


# --------------------------------------------------------------------------- #
# Label remapping
# --------------------------------------------------------------------------- #

def infer_label_values(mask_paths: Sequence[str], num_classes: int,
                       max_scan: int = 12) -> np.ndarray:
    """Infer the raw intensity value that encodes each class index.

    The published masks store the four labels as widely separated 16-bit values
    (0, 16448, 49087, 65535 in the copy we used) rather than 0-3, and resampling
    introduces small floating-point drift around them. Scanning a handful of
    volumes and taking the sorted unique values is robust to a mirror that used
    different constants.
    """
    if nib is None:
        raise ImportError("nibabel is required to read NIfTI volumes")
    observed: set = set()
    for path in list(mask_paths)[:max_scan]:
        data = np.asarray(nib.load(path).dataobj, dtype=np.float64)
        # Round away resampling drift before collecting uniques.
        observed.update(np.unique(np.round(data, 3)).tolist())
        if len(observed) > 4 * num_classes:
            break

    values = np.array(sorted(observed), dtype=np.float64)
    if len(values) == num_classes:
        return values

    # More uniques than classes: cluster them into `num_classes` groups by
    # splitting at the largest gaps, then use each group's median as the anchor.
    if len(values) > num_classes:
        gaps = np.diff(values)
        split_at = np.sort(np.argsort(gaps)[-(num_classes - 1):]) + 1
        groups = np.split(values, split_at)
        return np.array([float(np.median(g)) for g in groups], dtype=np.float64)

    raise ValueError(
        f"Found only {len(values)} distinct mask values ({values.tolist()}) but "
        f"expected {num_classes}. Check that the mask directory is correct."
    )


class LabelRemapper:
    """Snaps raw mask intensities to contiguous class indices 0..C-1."""

    def __init__(self, raw_values: np.ndarray):
        self.raw_values = np.asarray(raw_values, dtype=np.float64)
        self.midpoints = (self.raw_values[:-1] + self.raw_values[1:]) / 2.0

    def __call__(self, raw_mask: np.ndarray) -> np.ndarray:
        return np.searchsorted(self.midpoints, raw_mask, side="right").astype(np.int64)

    def __repr__(self) -> str:
        return f"LabelRemapper(raw_values={self.raw_values.tolist()})"


# --------------------------------------------------------------------------- #
# Preprocessing
# --------------------------------------------------------------------------- #

def brain_mask_from_volume(volume: np.ndarray) -> np.ndarray:
    """Crude but stable foreground mask: anything above a low fraction of the
    volume's robust maximum. Used only to restrict normalization statistics."""
    finite = volume[np.isfinite(volume)]
    if finite.size == 0:
        return np.zeros_like(volume, dtype=bool)
    hi = np.percentile(finite, 99.0)
    if not np.isfinite(hi) or hi <= 0:
        return volume > volume.min()
    return volume > (0.05 * hi)


def clip_and_normalize(volume: np.ndarray, low_pct: float, high_pct: float) -> np.ndarray:
    """Percentile-clip then z-score using brain-voxel statistics.

    Statistics are computed inside the brain mask only. Air is then set to the
    normalized value of the clipping floor rather than to a hard 0, which avoids
    manufacturing a spurious intensity edge at the skull boundary.
    """
    volume = np.nan_to_num(volume.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    mask = brain_mask_from_volume(volume)
    if mask.sum() < 64:
        mask = np.ones_like(volume, dtype=bool)

    lo, hi = np.percentile(volume[mask], [low_pct, high_pct])
    if hi <= lo:
        hi = lo + 1.0
    clipped = np.clip(volume, lo, hi)
    mean = float(clipped[mask].mean())
    std = float(clipped[mask].std()) + 1e-6
    return ((clipped - mean) / std).astype(np.float32)


def resize_2d(arr: np.ndarray, target: Tuple[int, int], is_mask: bool) -> np.ndarray:
    if tuple(arr.shape) == tuple(target):
        return arr
    tensor = torch.from_numpy(np.ascontiguousarray(arr)).float()[None, None]
    if is_mask:
        out = F.interpolate(tensor, size=target, mode="nearest")
    else:
        out = F.interpolate(tensor, size=target, mode="bilinear", align_corners=False)
    out = out[0, 0].numpy()
    return out.astype(np.int64) if is_mask else out.astype(np.float32)


# --------------------------------------------------------------------------- #
# Volume cache
# --------------------------------------------------------------------------- #

class PatientVolumeCache:
    """Loads, normalizes and resizes every requested patient once, in RAM.

    MS3SEG is 100 patients x 20 slices at 256x256x3 channels, roughly 1.5 GB as
    float32, so caching the whole cohort is comfortably within a Kaggle session
    and removes disk I/O from the training loop entirely.
    """

    def __init__(self, patient_ids: Sequence[str], patient_index: Dict[str, Dict[str, str]],
                 remapper: LabelRemapper, cfg, verbose: bool = True):
        if nib is None:
            raise ImportError("nibabel is required to read NIfTI volumes")
        self.cfg = cfg
        self.images: Dict[str, np.ndarray] = {}   # pid -> (M, H, W, S) float32
        # uint8, not int64: four class indices need one byte, and at 100 patients
        # x 256x256x20 the difference is about 1 GB of RAM that a Kaggle session
        # would rather spend on the teacher.
        self.masks: Dict[str, np.ndarray] = {}    # pid -> (H, W, S) uint8
        self.brain_fraction: Dict[str, np.ndarray] = {}  # pid -> (S,) float32
        # Physical voxel size in mm, adjusted for any in-plane resampling. Boundary
        # metrics (HD95, normalized surface Dice) are only interpretable in
        # millimetres, so the header spacing is carried through rather than
        # assuming isotropic unit voxels.
        self.spacing: Dict[str, Tuple[float, float, float]] = {}

        iterator = patient_ids
        if verbose:
            iterator = _maybe_tqdm(patient_ids, desc="caching volumes", unit="patient")

        for pid in iterator:
            entry = patient_index[pid]
            mask_img = nib.load(entry["mask"])
            raw_mask = np.asarray(mask_img.dataobj, dtype=np.float32)
            mask = remapper(raw_mask)

            zooms = tuple(float(z) for z in mask_img.header.get_zooms()[:3])
            if len(zooms) < 3 or not all(np.isfinite(zooms)) or min(zooms) <= 0:
                zooms = (1.0, 1.0, 1.0)
            scale_h = mask.shape[0] / float(cfg.target_size[0])
            scale_w = mask.shape[1] / float(cfg.target_size[1])
            self.spacing[pid] = (zooms[0] * scale_h, zooms[1] * scale_w, zooms[2])

            channels = []
            for modality in cfg.modalities:
                vol = np.asarray(nib.load(entry[modality]).dataobj, dtype=np.float32)
                if vol.shape[-1] != mask.shape[-1]:
                    raise ValueError(
                        f"{pid}: {modality} has {vol.shape[-1]} slices but the mask has "
                        f"{mask.shape[-1]}"
                    )
                vol = clip_and_normalize(vol, *cfg.clip_percentiles)
                channels.append(np.stack(
                    [resize_2d(vol[:, :, s], cfg.target_size, is_mask=False)
                     for s in range(vol.shape[-1])], axis=-1))

            self.images[pid] = np.stack(channels, axis=0)
            self.masks[pid] = np.stack(
                [resize_2d(mask[:, :, s], cfg.target_size, is_mask=True)
                 for s in range(mask.shape[-1])], axis=-1).astype(np.uint8)

            # Fraction of each slice that is brain rather than air, used to drop
            # near-empty end slices from training.
            flair_idx = cfg.modalities.index("FLAIR") if "FLAIR" in cfg.modalities else 0
            ref = self.images[pid][flair_idx]
            self.brain_fraction[pid] = (ref > ref.min() + 1e-3).mean(axis=(0, 1)).astype(np.float32)

    @property
    def patient_ids(self) -> List[str]:
        return list(self.images.keys())

    def n_slices(self, pid: str) -> int:
        return int(self.masks[pid].shape[-1])

    def get_slice(self, pid: str, s: int) -> Tuple[np.ndarray, np.ndarray]:
        return self.images[pid][:, :, :, s], self.masks[pid][:, :, s].astype(np.int64)

    def get_volume(self, pid: str) -> Tuple[np.ndarray, np.ndarray]:
        """Returns (S, M, H, W) images and (S, H, W) masks for whole-volume eval."""
        img = np.transpose(self.images[pid], (3, 0, 1, 2))
        msk = np.transpose(self.masks[pid], (2, 0, 1)).astype(np.int64)
        return img, msk

    def memory_bytes(self) -> int:
        """Resident size of the cache, for sanity-checking against session RAM."""
        return (sum(a.nbytes for a in self.images.values())
                + sum(a.nbytes for a in self.masks.values()))


def _maybe_tqdm(iterable, **kwargs):
    try:
        from tqdm.auto import tqdm
        return tqdm(iterable, **kwargs)
    except ImportError:
        return iterable


# --------------------------------------------------------------------------- #
# Augmentation
# --------------------------------------------------------------------------- #

def _affine_grid(theta: torch.Tensor, size: Tuple[int, int, int, int]) -> torch.Tensor:
    return F.affine_grid(theta, size, align_corners=False)


def augment_slice(image: np.ndarray, mask: np.ndarray, rng: random.Random,
                  max_rotation_deg: float = 15.0, max_translate: float = 0.06,
                  scale_range: Tuple[float, float] = (0.90, 1.10)) -> Tuple[np.ndarray, np.ndarray]:
    """Anatomically plausible 2-D augmentation.

    Left-right mirroring is retained because axial brain slices are close to
    sagittally symmetric. Arbitrary 90-degree rotations are *not* applied: they
    produce orientations that never occur in acquisition, and in our runs they
    cost roughly two Dice points on the rare lesion classes.
    """
    img = torch.from_numpy(np.ascontiguousarray(image)).float()[None]   # (1, M, H, W)
    msk = torch.from_numpy(np.ascontiguousarray(mask)).float()[None, None]  # (1, 1, H, W)

    if rng.random() < 0.5:
        img = torch.flip(img, dims=[3])
        msk = torch.flip(msk, dims=[3])

    if rng.random() < 0.7:
        angle = math.radians(rng.uniform(-max_rotation_deg, max_rotation_deg))
        scale = rng.uniform(*scale_range)
        tx = rng.uniform(-max_translate, max_translate)
        ty = rng.uniform(-max_translate, max_translate)
        cos_a, sin_a = math.cos(angle) / scale, math.sin(angle) / scale
        theta = torch.tensor([[[cos_a, -sin_a, tx], [sin_a, cos_a, ty]]], dtype=torch.float32)
        grid = _affine_grid(theta, (1, img.shape[1], img.shape[2], img.shape[3]))
        img = F.grid_sample(img, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
        msk = F.grid_sample(msk, grid, mode="nearest", padding_mode="zeros", align_corners=False)

    out_img = img[0].numpy()
    out_msk = msk[0, 0].numpy().astype(np.int64)

    # Per-channel intensity perturbation, applied after the geometric transform
    # so it is not resampled.
    if rng.random() < 0.5:
        gain = np.array([rng.uniform(0.90, 1.10) for _ in range(out_img.shape[0])],
                        dtype=np.float32)[:, None, None]
        bias = np.array([rng.uniform(-0.10, 0.10) for _ in range(out_img.shape[0])],
                        dtype=np.float32)[:, None, None]
        out_img = out_img * gain + bias
    if rng.random() < 0.25:
        out_img = out_img + np.random.normal(0.0, 0.03, size=out_img.shape).astype(np.float32)

    return np.ascontiguousarray(out_img, dtype=np.float32), np.ascontiguousarray(out_msk)


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #

class MS3SEGSliceDataset(Dataset):
    """Axial slices drawn from a cached set of patient volumes.

    Training instances optionally oversample lesion-bearing slices; evaluation
    instances never do, and never drop slices, so validation and test metrics are
    computed over the complete volume.
    """

    def __init__(self, patient_ids: Sequence[str], cache: PatientVolumeCache, cfg,
                 augment: bool = False, oversample: bool = False, seed: int = 0):
        self.cfg = cfg
        self.cache = cache
        self.augment = augment
        self.base_seed = seed
        self.epoch = 0

        self.index: List[Tuple[str, int]] = []
        for pid in patient_ids:
            for s in range(cache.n_slices(pid)):
                if augment and cache.brain_fraction[pid][s] < cfg.min_brain_fraction:
                    continue  # near-empty end slice, training only
                self.index.append((pid, s))

        if oversample and cfg.lesion_oversample_factor > 1.0:
            extra_reps = int(round(cfg.lesion_oversample_factor)) - 1
            lesion_slices = [
                (pid, s) for pid, s in self.index
                if np.any(np.isin(cache.masks[pid][:, :, s], cfg.lesion_classes))
            ]
            self.index.extend(lesion_slices * extra_reps)
            self.n_lesion_slices = len(lesion_slices)
        else:
            self.n_lesion_slices = 0

    def set_epoch(self, epoch: int) -> None:
        """Makes augmentation vary across epochs while staying reproducible."""
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int):
        pid, s = self.index[i]
        image, mask = self.cache.get_slice(pid, s)
        if self.augment:
            # Seeded per (epoch, index) so the exact same augmented batch is
            # delivered to every ablation variant -- the comparison is paired.
            rng = random.Random((self.base_seed * 1_000_003) ^ (self.epoch * 7919) ^ (i * 104_729))
            image, mask = augment_slice(image, mask, rng)
        else:
            image = np.ascontiguousarray(image)
            mask = np.ascontiguousarray(mask)
        return torch.from_numpy(image).float(), torch.from_numpy(mask).long()


def make_dataloaders(train_ids: Sequence[str], val_ids: Sequence[str],
                     cache: PatientVolumeCache, cfg, seed: int = 0):
    train_ds = MS3SEGSliceDataset(train_ids, cache, cfg, augment=True, oversample=True, seed=seed)
    val_ds = MS3SEGSliceDataset(val_ids, cache, cfg, augment=False, oversample=False, seed=seed)

    generator = torch.Generator()
    generator.manual_seed(seed)
    persistent = cfg.num_workers > 0
    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory, drop_last=True, persistent_workers=persistent,
        generator=generator,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory, persistent_workers=persistent,
    )
    return train_loader, val_loader


# --------------------------------------------------------------------------- #
# Patient-level partitioning
# --------------------------------------------------------------------------- #

def split_test_and_dev(patient_ids: Sequence[str], test_fraction: float,
                       seed: int) -> Tuple[List[str], List[str]]:
    rng = np.random.RandomState(seed)
    shuffled = rng.permutation(sorted(patient_ids))
    n_test = max(1, int(round(len(shuffled) * test_fraction)))
    return shuffled[n_test:].tolist(), shuffled[:n_test].tolist()


def make_fold_splits(dev_ids: Sequence[str], n_folds: int,
                     seed: int) -> List[Dict[str, List[str]]]:
    from sklearn.model_selection import KFold
    ids = np.array(sorted(dev_ids))
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    return [{"train": ids[tr].tolist(), "val": ids[va].tolist()} for tr, va in kf.split(ids)]


def summarize_class_balance(cache: PatientVolumeCache, cfg) -> Dict[str, float]:
    """Voxel fraction per class over the cached cohort, for the dataset table."""
    counts = np.zeros(cfg.num_classes, dtype=np.int64)
    for pid in cache.patient_ids:
        counts += np.bincount(cache.masks[pid].ravel().astype(np.int64),
                              minlength=cfg.num_classes)
    total = counts.sum()
    return {name: float(c) / float(total) for name, c in zip(cfg.class_names, counts)}
