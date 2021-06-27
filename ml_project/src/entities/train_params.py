from dataclasses import dataclass, field
from typing import Optional


@dataclass()
class TrainingParams:
    """
    Dataclass for model parameters configuration.
    """
    model_type: str = field(default="RandomForestClassifier")
    random_state: int = field(default=42)
    max_depth: Optional[int] = None
    n_estimators: Optional[int] = 100
    solver: Optional[str] = None
