"""Train and evaluate the PyTorch concept bottleneck model."""

import argparse
import os
from typing import Dict, Optional, Sequence

import torch
from torch import nn
from torch.utils.data import DataLoader

from data import (
    CONCEPT_COLUMNS,
    CONCEPT_VALUES,
    DIAGNOSIS_VALUES,
    build_datasets,
)
from model import ConceptBottleneckModel


def compute_loss(
    concept_logits: Sequence[torch.Tensor],
    diagnosis_logits: torch.Tensor,
    concept_targets: torch.Tensor,
    diagnosis_targets: torch.Tensor,
    concept_weight: float,
) -> torch.Tensor:
    concept_loss = torch.stack(
        [
            nn.functional.cross_entropy(logits, concept_targets[:, index])
            for index, logits in enumerate(concept_logits)
        ]
    ).mean()
    diagnosis_loss = nn.functional.cross_entropy(diagnosis_logits, diagnosis_targets)
    return diagnosis_loss + concept_weight * concept_loss


def compute_concept_loss(
    concept_logits: Sequence[torch.Tensor],
    concept_targets: torch.Tensor,
) -> torch.Tensor:
    return torch.stack(
        [
            nn.functional.cross_entropy(logits, concept_targets[:, index])
            for index, logits in enumerate(concept_logits)
        ]
    ).mean()


def compute_diagnosis_loss(
    diagnosis_logits: torch.Tensor,
    diagnosis_targets: torch.Tensor,
) -> torch.Tensor:
    return nn.functional.cross_entropy(diagnosis_logits, diagnosis_targets)


def run_epoch(
    model: ConceptBottleneckModel,
    loader: DataLoader,
    device: torch.device,
    optimizer: Optional[torch.optim.Optimizer] = None,
    concept_weight: float = 1.0,
) -> Dict[str, float]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    diagnosis_correct = 0
    sample_count = 0
    for images, concepts, diagnoses in loader:
        images, concepts, diagnoses = (
            images.to(device),
            concepts.to(device),
            diagnoses.to(device),
        )
        with torch.set_grad_enabled(training):
            concept_logits, diagnosis_logits = model(images)
            loss = compute_loss(
                concept_logits,
                diagnosis_logits,
                concepts,
                diagnoses,
                concept_weight,
            )
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        diagnosis_correct += (
            diagnosis_logits.argmax(dim=1) == diagnoses
        ).sum().item()
        sample_count += batch_size

    return {
        "loss": total_loss / sample_count,
        "diagnosis_accuracy": diagnosis_correct / sample_count,
    }


def run_concept_epoch(
    model: ConceptBottleneckModel,
    loader: DataLoader,
    device: torch.device,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> Dict[str, float]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    sample_count = 0
    for images, concepts, _ in loader:
        images, concepts = images.to(device), concepts.to(device)
        with torch.set_grad_enabled(training):
            concept_logits, _ = model(images)
            loss = compute_concept_loss(concept_logits, concepts)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        sample_count += batch_size
    return {"concept_loss": total_loss / sample_count}


def run_independent_diagnosis_epoch(
    model: ConceptBottleneckModel,
    loader: DataLoader,
    device: torch.device,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> Dict[str, float]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    diagnosis_correct = 0
    sample_count = 0
    for images, _, diagnoses in loader:
        images, diagnoses = images.to(device), diagnoses.to(device)
        with torch.set_grad_enabled(training):
            _, diagnosis_logits = model(images, detach_concepts=True)
            loss = compute_diagnosis_loss(diagnosis_logits, diagnoses)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        diagnosis_correct += (
            diagnosis_logits.argmax(dim=1) == diagnoses
        ).sum().item()
        sample_count += batch_size
    return {
        "diagnosis_loss": total_loss / sample_count,
        "diagnosis_accuracy": diagnosis_correct / sample_count,
    }


def freeze_concept_branch(model: ConceptBottleneckModel) -> None:
    for parameter in model.feature_extractor.parameters():
        parameter.requires_grad = False
    for parameter in model.concept_heads.parameters():
        parameter.requires_grad = False


def save_checkpoint(
    path: str,
    model: ConceptBottleneckModel,
    image_size: int,
) -> None:
    torch.save(
        {
            "model": model.state_dict(),
            "concept_columns": CONCEPT_COLUMNS,
            "concept_values": CONCEPT_VALUES,
            "diagnosis_values": DIAGNOSIS_VALUES,
            "image_size": image_size,
        },
        path,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    default_root = os.path.join(os.path.dirname(__file__), "release_v0")
    parser.add_argument("--data-root", default=default_root)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--concept-weight", type=float, default=1.0)
    parser.add_argument(
        "--mode",
        choices=("joint", "sequential", "independent"),
        default="joint",
        help=(
            "joint trains both losses together; sequential pretrains concepts "
            "then freezes them; independent uses separate objectives/optimizers"
        ),
    )
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--output", default="concept_bottleneck.pt")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    datasets = build_datasets(args.data_root, args.image_size)
    loaders = {
        split: DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=split == "train",
            num_workers=0,
        )
        for split, dataset in datasets.items()
    }
    model = ConceptBottleneckModel(
        [len(CONCEPT_VALUES[column]) for column in CONCEPT_COLUMNS],
        len(DIAGNOSIS_VALUES),
    ).to(device)
    concept_parameters = list(model.feature_extractor.parameters())
    concept_parameters += list(model.concept_heads.parameters())
    concept_optimizer = torch.optim.Adam(concept_parameters, lr=args.learning_rate)
    diagnosis_optimizer = torch.optim.Adam(
        model.diagnosis_head.parameters(), lr=args.learning_rate
    )
    joint_optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    best_valid_accuracy = -1.0

    print(
        f"Using {device}; train={len(datasets['train'])}, "
        f"valid={len(datasets['valid'])}, test={len(datasets['test'])}"
    )
    if args.mode == "sequential":
        print("Sequential phase 1: training concept predictors.")
        for epoch in range(1, args.epochs + 1):
            train_metrics = run_concept_epoch(
                model, loaders["train"], device, concept_optimizer
            )
            print(
                f"concept epoch {epoch:03d} | "
                f"train concept_loss={train_metrics['concept_loss']:.4f}"
            )
        freeze_concept_branch(model)
        print("Sequential phase 2: training diagnosis head with frozen concepts.")

    for epoch in range(1, args.epochs + 1):
        if args.mode == "joint":
            train_metrics = run_epoch(
                model, loaders["train"], device, joint_optimizer, args.concept_weight
            )
        elif args.mode == "sequential":
            train_metrics = run_independent_diagnosis_epoch(
                model, loaders["train"], device, diagnosis_optimizer
            )
        else:
            concept_metrics = run_concept_epoch(
                model, loaders["train"], device, concept_optimizer
            )
            train_metrics = run_independent_diagnosis_epoch(
                model, loaders["train"], device, diagnosis_optimizer
            )
            train_metrics["concept_loss"] = concept_metrics["concept_loss"]

        valid_metrics = run_independent_diagnosis_epoch(
            model, loaders["valid"], device
        )
        train_loss = train_metrics["loss"] if args.mode == "joint" else train_metrics["diagnosis_loss"]
        print(
            f"epoch {epoch:03d} | "
            f"train loss={train_loss:.4f} "
            f"diag_acc={train_metrics['diagnosis_accuracy']:.3f} | "
            f"valid loss={valid_metrics['diagnosis_loss']:.4f} "
            f"diag_acc={valid_metrics['diagnosis_accuracy']:.3f}"
        )
        if valid_metrics["diagnosis_accuracy"] > best_valid_accuracy:
            best_valid_accuracy = valid_metrics["diagnosis_accuracy"]
            save_checkpoint(args.output, model, args.image_size)

    checkpoint = torch.load(args.output, map_location=device)
    model.load_state_dict(checkpoint["model"])
    test_metrics = run_independent_diagnosis_epoch(model, loaders["test"], device)
    print(
        f"test loss={test_metrics['diagnosis_loss']:.4f} "
        f"diag_acc={test_metrics['diagnosis_accuracy']:.3f}"
    )


if __name__ == "__main__":
    main()
