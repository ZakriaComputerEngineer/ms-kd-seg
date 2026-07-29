"""Held-out test evaluation.

Every model -- teacher included -- is evaluated under exactly the same protocol:
the K fold checkpoints are ensembled by averaging softmax probabilities, and
metrics are computed per patient over the complete volume. Our earlier pipeline
compared a single best-fold teacher against three-fold student ensembles, which
handicapped the teacher by an amount that had nothing to do with distillation.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn

from .config import Config
from .data import PatientVolumeCache
from .metrics import CaseMetrics, evaluate_volume, global_dice, summarize
from .train import predict_volume_labels


@dataclass
class ModelEvaluation:
    name: str
    label: str
    n_params: int
    cases: List[CaseMetrics] = field(default_factory=list)
    summary: Dict = field(default_factory=dict)
    global_dice: Dict[str, float] = field(default_factory=dict)
    n_ensemble_members: int = 1
    objective_note: str = ""
    citation: str = ""
    # Accuracy of the *individually deployable* model, as mean and standard
    # deviation across the K single-fold checkpoints. Reported alongside the
    # ensemble so that the accuracy table and the efficiency table describe the
    # same artefact: a K-member ensemble costs K forward passes, and quoting its
    # accuracy next to a single model's latency would overstate the system.
    single_model_dice: Dict[str, float] = field(default_factory=dict)
    single_model_dice_std: Dict[str, float] = field(default_factory=dict)

    def dice(self, class_name: str) -> float:
        return self.summary.get("dice", {}).get(class_name, {}).get("mean", float("nan"))

    def std(self, metric: str, class_name: str) -> float:
        return self.summary.get(metric, {}).get(class_name, {}).get("std", float("nan"))

    def value(self, metric: str, class_name: str) -> float:
        return self.summary.get(metric, {}).get(class_name, {}).get("mean", float("nan"))

    def mean_foreground_dice(self) -> float:
        return self.summary.get("summary", {}).get("mean_foreground_dice", {}).get("mean", float("nan"))

    def to_dict(self) -> Dict:
        return {
            "name": self.name, "label": self.label, "n_params": self.n_params,
            "n_ensemble_members": self.n_ensemble_members,
            "objective_note": self.objective_note, "citation": self.citation,
            "summary": self.summary, "global_dice": self.global_dice,
            "single_model_dice": self.single_model_dice,
            "single_model_dice_std": self.single_model_dice_std,
            "cases": [c.to_dict() for c in self.cases],
        }


    @classmethod
    def from_dict(cls, payload: Dict) -> "ModelEvaluation":
        """Rebuild from `test_evaluations.json` so tables and figures can be
        regenerated without re-running inference."""
        ev = cls(
            name=payload["name"], label=payload["label"], n_params=payload["n_params"],
            cases=[CaseMetrics.from_dict(c) for c in payload.get("cases", [])],
            summary=payload.get("summary", {}),
            global_dice=payload.get("global_dice", {}),
            n_ensemble_members=payload.get("n_ensemble_members", 1),
        )
        ev.objective_note = payload.get("objective_note", "")
        ev.citation = payload.get("citation", "")
        ev.single_model_dice = payload.get("single_model_dice", {})
        ev.single_model_dice_std = payload.get("single_model_dice_std", {})
        return ev


def load_evaluations(path: str) -> Dict[str, ModelEvaluation]:
    """Load a saved `test_evaluations.json` back into evaluation objects."""
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)
    return {k: ModelEvaluation.from_dict(v) for k, v in payload["evaluations"].items()}


def load_ensemble(builder, checkpoints: Sequence[str], device: torch.device,
                  cfg: Config) -> List[nn.Module]:
    models = []
    for path in checkpoints:
        if not os.path.exists(path):
            raise FileNotFoundError(f"missing checkpoint {path}")
        model = builder(cfg).to(device)
        state = torch.load(path, map_location=device, weights_only=False)
        if isinstance(state, dict) and "model_state" in state:
            state = state["model_state"]
        model.load_state_dict(state)
        model.eval()
        models.append(model)
    if not models:
        raise ValueError("no checkpoints supplied for the ensemble")
    return models


@torch.no_grad()
def evaluate_ensemble(models: Sequence[nn.Module], cache: PatientVolumeCache,
                      test_ids: Sequence[str], cfg: Config, device: torch.device,
                      name: str, label: str, n_params: int,
                      compute_boundary: bool = True,
                      store_predictions_for: Optional[Sequence[str]] = None,
                      prediction_store: Optional[Dict[str, np.ndarray]] = None) -> ModelEvaluation:
    """Per-patient metrics for one model (or fold ensemble) on the test cohort."""
    from .train import _tqdm

    cases: List[CaseMetrics] = []
    predictions: List[np.ndarray] = []
    references: List[np.ndarray] = []

    for pid in _tqdm(list(test_ids), cfg.progress, desc=f"  eval {name}", leave=False):
        pred = predict_volume_labels(models, cache, pid, cfg, device)   # (H, W, S)
        ref = cache.masks[pid]
        cases.append(evaluate_volume(pred, ref, cfg, patient_id=pid,
                                     spacing=cache.spacing.get(pid, (1.0, 1.0, 1.0)),
                                     compute_boundary=compute_boundary))
        predictions.append(pred)
        references.append(ref)
        if store_predictions_for and pid in store_predictions_for and prediction_store is not None:
            prediction_store[f"{name}::{pid}"] = pred

    return ModelEvaluation(
        name=name, label=label, n_params=n_params, cases=cases,
        summary=summarize(cases, cfg),
        global_dice=global_dice(predictions, references, cfg),
        n_ensemble_members=len(models),
    )


def evaluate_with_single_model_stats(models: Sequence[nn.Module], cache: PatientVolumeCache,
                                     test_ids: Sequence[str], cfg: Config, device: torch.device,
                                     name: str, label: str, n_params: int,
                                     compute_boundary: bool = True,
                                     store_predictions_for: Optional[Sequence[str]] = None,
                                     prediction_store: Optional[Dict[str, np.ndarray]] = None
                                     ) -> ModelEvaluation:
    """Evaluate the fold ensemble, then each fold member on its own.

    The ensemble is the headline result because it is the standard
    cross-validation protocol and because every model in the comparison -- the
    teacher included -- uses it, keeping the comparison internally fair. The
    single-model statistics exist so that the deployment claim can be stated
    about the artefact that is actually deployed, which is one 0.49M network, not
    three of them.
    """
    ensemble = evaluate_ensemble(models, cache, test_ids, cfg, device, name, label, n_params,
                                 compute_boundary=compute_boundary,
                                 store_predictions_for=store_predictions_for,
                                 prediction_store=prediction_store)

    if len(models) > 1:
        per_fold: Dict[str, List[float]] = {c: [] for c in cfg.class_names}
        for i, model in enumerate(models):
            member = evaluate_ensemble([model], cache, test_ids, cfg, device,
                                       f"{name}_fold{i}", label, n_params,
                                       compute_boundary=False)
            for class_name in cfg.class_names:
                value = member.dice(class_name)
                if not np.isnan(value):
                    per_fold[class_name].append(value)
        for class_name, values in per_fold.items():
            if values:
                ensemble.single_model_dice[class_name] = float(np.mean(values))
                # nan, not 0.0: a single observation has no spread, and printing
                # "+/- 0.000" would claim a precision that was never measured.
                # `fmt_pm` renders nan as the mean alone.
                ensemble.single_model_dice_std[class_name] = (
                    float(np.std(values, ddof=1)) if len(values) > 1 else float("nan"))
    else:
        for class_name in cfg.class_names:
            value = ensemble.dice(class_name)
            if not np.isnan(value):
                ensemble.single_model_dice[class_name] = value
                ensemble.single_model_dice_std[class_name] = 0.0

    return ensemble


def save_evaluations(cfg: Config, evaluations: Dict[str, ModelEvaluation],
                     filename: str = "test_evaluations.json") -> str:
    path = os.path.join(cfg.results_dir, filename)
    payload = {
        "fingerprint": cfg.fingerprint(),
        "class_names": list(cfg.class_names),
        "evaluations": {k: v.to_dict() for k, v in evaluations.items()},
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)
    return path


def teacher_student_gap_recovery(teacher: ModelEvaluation, scratch: ModelEvaluation,
                                 variant: ModelEvaluation, class_name: str) -> float:
    """Fraction of the teacher-minus-scratch gap that a distilled variant closes.

    Undefined -- and returned as nan -- when the teacher does not actually beat
    the from-scratch student on this class, because "recovering X% of the gap" is
    meaningless when the gap is negative. Reporting a percentage there is a
    common way to make a failed distillation look successful.
    """
    t = teacher.dice(class_name)
    s = scratch.dice(class_name)
    v = variant.dice(class_name)
    if any(np.isnan(x) for x in (t, s, v)):
        return float("nan")
    gap = t - s
    if gap <= 1e-6:
        return float("nan")
    return float((v - s) / gap)
