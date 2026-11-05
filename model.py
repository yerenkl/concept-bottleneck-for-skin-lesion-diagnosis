"""Concept bottleneck neural network."""

from typing import List, Sequence, Tuple

import torch
from torch import nn


class ConceptBottleneckModel(nn.Module):
    def __init__(self, concept_sizes: Sequence[int], diagnosis_count: int) -> None:
        super().__init__()
        self.concept_sizes = list(concept_sizes)
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.concept_heads = nn.ModuleList(
            [nn.Linear(128, size) for size in self.concept_sizes]
        )
        self.diagnosis_head = nn.Linear(sum(self.concept_sizes), diagnosis_count)

    def forward(
        self, images: torch.Tensor
    ) -> Tuple[List[torch.Tensor], torch.Tensor]:
        features = self.feature_extractor(images).flatten(1)
        concept_logits = [head(features) for head in self.concept_heads]
        concept_probabilities = [
            torch.softmax(logits, dim=1) for logits in concept_logits
        ]
        diagnosis_logits = self.diagnosis_head(
            torch.cat(concept_probabilities, dim=1)
        )
        return concept_logits, diagnosis_logits
