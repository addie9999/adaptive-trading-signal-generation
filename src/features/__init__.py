"""Feature engineering: reusable, composable indicator generators.

Each feature *group* is a subclass of :class:`base.FeatureGenerator` and emits
a named block of columns. :class:`engineer.FeatureEngineer` composes them and
also builds the forward-return label used for supervised modelling.
"""

from src.features.base import FeatureGenerator
from src.features.engineer import FeatureEngineer

__all__ = ["FeatureGenerator", "FeatureEngineer"]
