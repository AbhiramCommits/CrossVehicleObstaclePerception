"""CrossVehicleObstaclePerception: cross-vehicle obstacle detection with
transfer learning, adversarial domain adaptation and knowledge distillation."""
__version__ = "1.0.0"

from .config import (  # noqa: F401
    AdaptationConfig,
    BenchmarkConfig,
    Config,
    DistillationConfig,
    OpticalConfig,
    TrainingConfig,
)
from .data import (  # noqa: F401
    BDD100KDataset,
    CLASS_NAMES,
    DomainAdaptationDataset,
    KittiDataset,
    Obstacle,
    augment_image,
)
