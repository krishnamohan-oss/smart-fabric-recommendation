"""Database package for Smart Fabric Recommendation System."""
from .feedback import (
    init_db,
    save_feedback,
    get_user_history,
    get_all_feedback,
    get_user_fabric_affinity,
    get_feedback_summary_stats,
    reset_user_history
)

__all__ = [
    "init_db",
    "save_feedback",
    "get_user_history",
    "get_all_feedback",
    "get_user_fabric_affinity",
    "get_feedback_summary_stats",
    "reset_user_history"
]
