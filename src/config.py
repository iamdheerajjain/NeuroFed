import os
from dataclasses import dataclass, field
from typing import List, Optional
import yaml


@dataclass
class TrainConfig:
    data_root: str
    image_size: int = 224
    batch_size: int = 16
    num_workers: int = 2
    num_classes: int = 3
    class_names: List[str] = field(default_factory=lambda: ["ischemic", "hemorrhagic", "normal"])
    epochs: int = 10
    lr: float = 1e-3
    weight_decay: float = 1e-4
    early_stopping_patience: int = 5
    val_split: float = 0.2
    seed: int = 42
    checkpoint_dir: str = "checkpoints"
    use_pretrained: bool = False
    use_augmentation: bool = False


@dataclass
class FederatedConfig:
    server_address: str = "127.0.0.1:8080"
    num_clients: int = 3
    rounds: int = 5
    fraction_fit: float = 1.0
    fraction_eval: float = 1.0


@dataclass
class AppConfig:
    train: TrainConfig
    federated: FederatedConfig


def load_config(path: str) -> AppConfig:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    train = TrainConfig(**raw.get("train", {}))
    federated = FederatedConfig(**raw.get("federated", {}))
    return AppConfig(train=train, federated=federated)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)
