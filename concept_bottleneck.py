"""Backward-compatible entry point for the separated concept bottleneck project."""

from data import (
    CONCEPT_COLUMNS,
    CONCEPT_VALUES,
    DIAGNOSIS_VALUES,
    Derm7PtConceptDataset,
    build_datasets,
)
from model import ConceptBottleneckModel
from train import compute_loss, main, run_epoch


if __name__ == "__main__":
    main()
