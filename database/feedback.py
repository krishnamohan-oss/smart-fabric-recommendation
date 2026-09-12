"""
SQLite database operations for user feedback and personalization profile learning.
Stores interaction history, rating, selection decisions, and preference metrics.
"""

import os
import sqlite3
from datetime import datetime
from typing import Dict, List, Optional, Any
import pandas as pd

DEFAULT_DB_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DB_PATH = os.path.join(DEFAULT_DB_DIR, "fabrics_feedback.db")


def get_db_connection(db_path: Optional[str] = None) -> sqlite3.Connection:
    """Get a connection to the SQLite database."""
    target_path = db_path or os.getenv("DATABASE_PATH") or DEFAULT_DB_PATH
    os.makedirs(os.path.dirname(os.path.abspath(target_path)), exist_ok=True)
    conn = sqlite3.connect(target_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: Optional[str] = None) -> None:
    """Initialize database schema with user_feedback table."""
    conn = get_db_connection(db_path)
    try:
        with conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS user_feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    garment TEXT NOT NULL,
                    climate TEXT NOT NULL,
                    budget TEXT NOT NULL,
                    recommended_fabric TEXT NOT NULL,
                    selected_fabric TEXT NOT NULL,
                    rating INTEGER NOT NULL CHECK (rating BETWEEN 1 AND 5),
                    would_choose INTEGER NOT NULL CHECK (would_choose IN (0, 1)),
                    decision_factors TEXT,
                    user_notes TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_feedback_user ON user_feedback(user_id);"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_feedback_fabric ON user_feedback(selected_fabric);"
            )
    finally:
        conn.close()


def save_feedback(
    user_id: str,
    garment: str,
    climate: str,
    budget: str,
    recommended_fabric: str,
    selected_fabric: str,
    rating: int,
    would_choose: bool,
    decision_factors: str = "",
    user_notes: str = "",
    db_path: Optional[str] = None,
) -> int:
    """
    Save user feedback and choices into the database.
    Returns the newly inserted row ID.
    """
    init_db(db_path)
    conn = get_db_connection(db_path)
    try:
        with conn:
            cursor = conn.execute(
                """
                INSERT INTO user_feedback (
                    user_id, garment, climate, budget,
                    recommended_fabric, selected_fabric, rating,
                    would_choose, decision_factors, user_notes, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    user_id.strip() if user_id else "Guest",
                    garment,
                    climate,
                    budget,
                    recommended_fabric,
                    selected_fabric,
                    int(rating),
                    1 if would_choose else 0,
                    decision_factors,
                    user_notes,
                    datetime.now().isoformat(),
                ),
            )
            return cursor.lastrowid
    finally:
        conn.close()


def get_user_history(user_id: str, db_path: Optional[str] = None) -> pd.DataFrame:
    """Retrieve interaction history for a given user as a DataFrame."""
    init_db(db_path)
    conn = get_db_connection(db_path)
    try:
        query = """
            SELECT id, user_id, garment, climate, budget,
                   recommended_fabric, selected_fabric, rating,
                   would_choose, decision_factors, user_notes, created_at
            FROM user_feedback
            WHERE user_id = ?
            ORDER BY created_at DESC
        """
        df = pd.read_sql_query(query, conn, params=(user_id,))
        return df
    finally:
        conn.close()


def get_all_feedback(db_path: Optional[str] = None) -> pd.DataFrame:
    """Retrieve all feedback records across all users."""
    init_db(db_path)
    conn = get_db_connection(db_path)
    try:
        query = """
            SELECT id, user_id, garment, climate, budget,
                   recommended_fabric, selected_fabric, rating,
                   would_choose, decision_factors, user_notes, created_at
            FROM user_feedback
            ORDER BY created_at DESC
        """
        df = pd.read_sql_query(query, conn)
        return df
    finally:
        conn.close()


def get_user_fabric_affinity(user_id: str, db_path: Optional[str] = None) -> Dict[str, float]:
    """
    Calculate an affinity score (0.0 to 100.0) for fabrics chosen/rated by the user.
    Uses rating and whether they would choose the fabric:
      - 5-star rating + would choose gives highest affinity boost (up to 100)
      - 1-2 star ratings penalize affinity
    """
    df = get_user_history(user_id, db_path)
    if df.empty:
        return {}

    fabric_affinity: Dict[str, float] = {}
    grouped = df.groupby("selected_fabric")

    for fabric, group in grouped:
        avg_rating = group["rating"].mean()  # 1.0 - 5.0
        acceptance_rate = group["would_choose"].mean()  # 0.0 - 1.0
        total_interactions = len(group)

        # Baseline score mapped from rating (1=20, 5=100)
        rating_score = avg_rating * 20.0
        # Acceptance bonus: +10 if 100% accepted, -10 if rejected
        acceptance_adjustment = (acceptance_rate - 0.5) * 20.0
        # Frequency confidence weight
        confidence = min(1.0, total_interactions / 3.0)

        raw_score = (rating_score + acceptance_adjustment)
        clamped_score = max(0.0, min(100.0, raw_score))

        # Blend with neutral 50 based on confidence
        affinity = 50.0 * (1.0 - confidence) + clamped_score * confidence
        fabric_affinity[str(fabric)] = round(affinity, 2)

    return fabric_affinity


def get_feedback_summary_stats(db_path: Optional[str] = None) -> Dict[str, Any]:
    """Return aggregated metrics about global user interactions."""
    df = get_all_feedback(db_path)
    if df.empty:
        return {
            "total_reviews": 0,
            "avg_rating": 0.0,
            "acceptance_rate": 0.0,
            "top_selected_fabric": "None",
            "unique_users": 0,
        }

    return {
        "total_reviews": int(len(df)),
        "avg_rating": round(float(df["rating"].mean()), 2),
        "acceptance_rate": round(float(df["would_choose"].mean() * 100), 1),
        "top_selected_fabric": str(df["selected_fabric"].mode().iloc[0]) if not df["selected_fabric"].empty else "None",
        "unique_users": int(df["user_id"].nunique()),
    }


def reset_user_history(user_id: str, db_path: Optional[str] = None) -> int:
    """Delete all feedback records for a specific user. Returns rows deleted."""
    conn = get_db_connection(db_path)
    try:
        with conn:
            cursor = conn.execute("DELETE FROM user_feedback WHERE user_id = ?", (user_id,))
            return cursor.rowcount
    finally:
        conn.close()
