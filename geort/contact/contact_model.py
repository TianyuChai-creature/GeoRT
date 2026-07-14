"""Train independent contact classifiers from automatically labeled D1 samples."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from geort.contact.auto_label_contacts import PAIR_NAMES


@dataclass(frozen=True, slots=True)
class ContactDataset:
    features: np.ndarray
    labels: np.ndarray
    held_out: np.ndarray
    pair_names: tuple[str, ...]
    pair_landmarks: np.ndarray
    frame_indices: np.ndarray


@dataclass(frozen=True, slots=True)
class FeatureScaler:
    mean: np.ndarray
    scale: np.ndarray

    def transform(self, features: np.ndarray) -> np.ndarray:
        return ((np.asarray(features) - self.mean) / self.scale).astype(np.float32)


@dataclass(frozen=True, slots=True)
class ContactTrainingArtifacts:
    checkpoint_path: Path
    report_path: Path


class ContactMLP(nn.Module):
    def __init__(self, hidden_dims: Sequence[int] = (64, 32)) -> None:
        super().__init__()
        dimensions = (6, *tuple(hidden_dims), 1)
        layers: list[nn.Module] = []
        for input_dim, output_dim in zip(dimensions[:-2], dimensions[1:-1], strict=True):
            layers.extend((nn.Linear(input_dim, output_dim), nn.ReLU()))
        layers.append(nn.Linear(dimensions[-2], dimensions[-1]))
        self.network = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features).squeeze(-1)


def load_contact_dataset(path: Path | str) -> ContactDataset:
    source = Path(path)
    with np.load(source, allow_pickle=False) as bundle:
        required = {"features", "labels", "held_out", "pair_names", "pair_landmarks", "frame_indices"}
        missing = sorted(required.difference(bundle.files))
        if missing:
            raise KeyError(f"contact label bundle is missing fields: {missing}")
        features = np.asarray(bundle["features"], dtype=np.float32)
        labels = np.asarray(bundle["labels"], dtype=np.int8)
        held_out = np.asarray(bundle["held_out"], dtype=bool)
        pair_names = tuple(str(name) for name in bundle["pair_names"].tolist())
        pair_landmarks = np.asarray(bundle["pair_landmarks"], dtype=np.int64)
        frame_indices = np.asarray(bundle["frame_indices"], dtype=np.int64)

    rows = len(features)
    if features.shape != (rows, 4, 6):
        raise ValueError(f"features must have shape [T, 4, 6], got {features.shape}")
    if labels.shape != (rows, 4) or held_out.shape != (rows, 4):
        raise ValueError("labels and held_out must have shape [T, 4]")
    if pair_names != PAIR_NAMES or pair_landmarks.shape != (4, 2):
        raise ValueError("contact pair ordering does not match the Step 7 contract")
    if frame_indices.shape != (rows,) or np.any(np.diff(frame_indices) <= 0):
        raise ValueError("frame_indices must be a strictly increasing [T] array")
    if not np.isfinite(features).all() or not np.isin(labels, (-1, 0, 1)).all():
        raise ValueError("contact features or labels are invalid")
    if np.any(held_out & (labels < 0)):
        raise ValueError("ambiguous samples cannot be part of held-out evaluation")
    return ContactDataset(features, labels, held_out, pair_names, pair_landmarks, frame_indices)


def fit_pair_scaler(dataset: ContactDataset, pair_index: int) -> FeatureScaler:
    training_mask = (dataset.labels[:, pair_index] >= 0) & ~dataset.held_out[:, pair_index]
    training_features = dataset.features[training_mask, pair_index]
    if not len(training_features):
        raise ValueError(f"{dataset.pair_names[pair_index]} has no clear training samples")
    mean = training_features.mean(axis=0, dtype=np.float64)
    scale = training_features.std(axis=0, dtype=np.float64)
    scale = np.maximum(scale, np.finfo(np.float32).eps)
    return FeatureScaler(mean=mean.astype(np.float32), scale=scale.astype(np.float32))


def create_pair_trainers(
    *,
    hidden_dims: Sequence[int] = (64, 32),
    learning_rate: float = 1e-4,
) -> tuple[dict[str, ContactMLP], dict[str, optim.Optimizer]]:
    models = {name: ContactMLP(hidden_dims) for name in PAIR_NAMES}
    optimizers = {
        name: optim.Adam(models[name].parameters(), lr=learning_rate)
        for name in PAIR_NAMES
    }
    return models, optimizers


def _binary_metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, float | int]:
    predictions = probabilities >= 0.5
    positives = labels == 1
    negatives = labels == 0
    true_positive = int(np.count_nonzero(predictions & positives))
    false_positive = int(np.count_nonzero(predictions & negatives))
    false_negative = int(np.count_nonzero(~predictions & positives))
    true_negative = int(np.count_nonzero(~predictions & negatives))
    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, np.finfo(float).eps)
    return {
        "positive_count": int(np.count_nonzero(positives)),
        "negative_count": int(np.count_nonzero(negatives)),
        "true_positive": true_positive,
        "true_negative": true_negative,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "accuracy": float((true_positive + true_negative) / len(labels)),
    }


def train_contact_models(
    *,
    labels_path: Path | str,
    output_dir: Path | str,
    epochs: int = 20,
    learning_rate: float = 1e-4,
    batch_size: int = 2048,
    hidden_dims: Sequence[int] = (64, 32),
    seed: int = 0,
    device: str = "cpu",
) -> ContactTrainingArtifacts:
    if epochs <= 0 or batch_size <= 0 or learning_rate <= 0.0:
        raise ValueError("epochs, batch_size, and learning_rate must be positive")
    torch.manual_seed(seed)
    np.random.seed(seed)
    target_device = torch.device(device)
    dataset = load_contact_dataset(labels_path)
    models, optimizers = create_pair_trainers(
        hidden_dims=hidden_dims,
        learning_rate=learning_rate,
    )
    pair_checkpoints: dict[str, dict] = {}
    pair_reports: list[dict] = []

    for pair_index, name in enumerate(PAIR_NAMES):
        scaler = fit_pair_scaler(dataset, pair_index)
        clear = dataset.labels[:, pair_index] >= 0
        training_mask = clear & ~dataset.held_out[:, pair_index]
        held_out_mask = clear & dataset.held_out[:, pair_index]
        training_labels = dataset.labels[training_mask, pair_index].astype(np.float32)
        held_out_labels = dataset.labels[held_out_mask, pair_index].astype(np.int8)
        if not np.any(training_labels == 1) or not np.any(training_labels == 0):
            raise ValueError(f"{name} training split must contain both classes")
        if not np.any(held_out_labels == 1) or not np.any(held_out_labels == 0):
            raise ValueError(f"{name} held-out split must contain both classes")

        training_features = scaler.transform(dataset.features[training_mask, pair_index])
        held_out_features = scaler.transform(dataset.features[held_out_mask, pair_index])
        tensor_dataset = TensorDataset(
            torch.from_numpy(training_features),
            torch.from_numpy(training_labels),
        )
        generator = torch.Generator().manual_seed(seed + pair_index)
        loader = DataLoader(
            tensor_dataset,
            batch_size=batch_size,
            shuffle=True,
            generator=generator,
        )
        model = models[name].to(target_device)
        optimizer = optimizers[name]
        negative_count = int(np.count_nonzero(training_labels == 0))
        positive_count = int(np.count_nonzero(training_labels == 1))
        positive_weight = negative_count / positive_count
        criterion = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor(positive_weight, device=target_device)
        )
        history: list[float] = []
        for _ in range(epochs):
            model.train()
            loss_sum = 0.0
            sample_count = 0
            for feature_batch, label_batch in loader:
                feature_batch = feature_batch.to(target_device)
                label_batch = label_batch.to(target_device)
                loss = criterion(model(feature_batch), label_batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_sum += float(loss.item()) * len(feature_batch)
                sample_count += len(feature_batch)
            history.append(loss_sum / sample_count)

        model.eval()
        with torch.no_grad():
            logits = model(torch.from_numpy(held_out_features).to(target_device))
            probabilities = torch.sigmoid(logits).cpu().numpy()
        metrics = _binary_metrics(held_out_labels, probabilities)
        pair_reports.append(
            {
                "name": name,
                "landmark_indices": dataset.pair_landmarks[pair_index].astype(int).tolist(),
                "training": {
                    "positive_count": positive_count,
                    "negative_count": negative_count,
                    "positive_weight": float(positive_weight),
                },
                "held_out": metrics,
                "history": {"train_bce": history},
            }
        )
        pair_checkpoints[name] = {
            "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
            "scaler_mean": torch.from_numpy(scaler.mean.copy()),
            "scaler_scale": torch.from_numpy(scaler.scale.copy()),
            "landmark_indices": torch.from_numpy(dataset.pair_landmarks[pair_index].copy()),
        }

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "contact_models.pth"
    report_path = output_dir / "training_report.json"
    if checkpoint_path.exists() or report_path.exists():
        raise FileExistsError(f"refusing to overwrite contact training output: {output_dir}")
    checkpoint = {
        "schema_version": 1,
        "pairs": pair_checkpoints,
        "hidden_dims": tuple(int(value) for value in hidden_dims),
    }
    report = {
        "schema_version": 1,
        "labels_path": str(Path(labels_path)),
        "epochs": int(epochs),
        "learning_rate": float(learning_rate),
        "batch_size": int(batch_size),
        "hidden_dims": [int(value) for value in hidden_dims],
        "seed": int(seed),
        "held_out_note": "F1 is archival and optimistic because D1 has no hover hard negatives.",
        "pairs": pair_reports,
    }
    torch.save(checkpoint, checkpoint_path)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return ContactTrainingArtifacts(checkpoint_path, report_path)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--hand-side", choices=("left", "right"), default="right")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--hidden-dims", type=int, nargs=2, default=(64, 32))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    return parser


def main(argv: list[str] | None = None) -> ContactTrainingArtifacts:
    args = build_arg_parser().parse_args(argv)
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = args.output_dir or Path("checkpoint") / f"contact_{args.hand_side}_{timestamp}_{args.tag}"
    artifacts = train_contact_models(
        labels_path=args.labels,
        output_dir=output_dir,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        hidden_dims=args.hidden_dims,
        seed=args.seed,
        device=device,
    )
    print(f"Contact checkpoint saved to {artifacts.checkpoint_path}")
    print(f"Training report saved to {artifacts.report_path}")
    return artifacts


if __name__ == "__main__":
    main()
