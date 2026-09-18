"""StatArb ML exit & risk manager package (see SYSTEM_SPEC.md)."""

from src.features import FEATURE_NAMES, FEATURE_SCHEMA_VERSION, extract_feature_vector
from src.exit_manager import StatArbExitManager, TradeState
from src.config import load_strategy_config, exit_threshold

__all__ = [
    "FEATURE_NAMES",
    "FEATURE_SCHEMA_VERSION",
    "extract_feature_vector",
    "StatArbExitManager",
    "TradeState",
    "load_strategy_config",
    "exit_threshold",
]
