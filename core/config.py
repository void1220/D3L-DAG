"""Default D3L configuration."""
from dataclasses import dataclass, asdict

@dataclass
class D3LConfig:
    width: int = 256
    layers: int = 6
    heads: int = 8
    dropout: float = 0.05
    steps: int = 30
    noise_schedule: str = 'linear'
    kernel: str = 'absorbing'
    feature_dim: int = 17
    feature_mode: str = 'full'
    cross_attention: bool = False
    decoder: str = 'd3l'
    candidates: int = 256
    candidate_chunk: int = 64
    assignment_temperature: float = 0.0
    priority_temperature: float = 0.0
    resampling: bool = False
    resampling_rounds: int = 4
    restart_step: int = 15
    parents: int = 4
    ema: bool = True
    ema_decay: float = 0.99
    epochs: int = 800
    batch_size: int = 256
    learning_rate: float = 3e-05
    learning_rate_floor: float = 3e-06
    warmup_epochs: int = 10
    hold_epochs: int = 200
    schedule_epochs: int = 600
    permutation_augmentation: bool = True
    weight_decay: float = 0.01
    validation_every: int = 50
    validation_limit: int = 200
    patience: int = 3
    min_improvement: float = 0.001
    seed: int = 42
    threads: int = 4
    amp: bool = True

    def to_dict(self):
        return asdict(self)
