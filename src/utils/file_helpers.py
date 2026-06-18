"""Filesystem helpers for saving pipeline outputs."""

from __future__ import annotations

import json
import os
from typing import Any

import pandas as pd


def ensure_dir(path: str) -> str:
    """Create ``path`` (and parents) if needed; return it for chaining."""
    os.makedirs(path, exist_ok=True)
    return path


def save_dataframe(df: pd.DataFrame, path: str, index: bool = False) -> str:
    """Write a DataFrame to CSV, creating parent directories as needed."""
    ensure_dir(os.path.dirname(path))
    df.to_csv(path, index=index)
    return path


def save_json(obj: Any, path: str) -> str:
    """Write an object to pretty-printed JSON (NumPy scalars made native)."""
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, default=_to_native)
    return path


def save_text(text: str, path: str) -> str:
    """Write a plain-text file, creating parent directories as needed."""
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)
    return path


def _to_native(obj: Any) -> Any:
    """JSON fallback: convert NumPy scalars/arrays to built-in Python types."""
    if hasattr(obj, "item"):  # numpy scalar
        return obj.item()
    if hasattr(obj, "tolist"):  # numpy array
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")
