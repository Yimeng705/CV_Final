"""
Real-world SLAM Dataset Loaders
================================
Supports standard benchmarks for evaluation:
- TUM RGB-D (fr1, fr2, fr3 sequences)
- EuRoC MAV (MH_01-05, V1_01-03, V2_01-03)
- Replica (room0-2, office0-4)

From 3DGS-Survey (Chen & Wang, 2026) method-002 evaluation framework.
"""

from .dataset_loader import (
    TUMDataset,
    EuRoCDataset,
    ReplicaDataset,
    create_dataloader,
    DATASET_CONFIGS
)