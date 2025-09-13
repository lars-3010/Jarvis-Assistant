"""
User configuration loader (YAML-based).

Loads `config/base.yaml` and `config/local.yaml` (if present), merges them,
and exposes helpers for components to read customization like property extraction
rules (frontmatter tag keys, inline tag prefixes, etc.).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from jarvis.utils.logging import setup_logging

logger = setup_logging(__name__)


_CONFIG_CACHE: dict[str, Any] | None = None


def _deep_merge(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    result = dict(a)
    for k, v in b.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def get_user_config(project_root: Path | None = None) -> dict[str, Any]:
    global _CONFIG_CACHE
    if _CONFIG_CACHE is not None:
        return _CONFIG_CACHE

    base: dict[str, Any] = {}
    local: dict[str, Any] = {}

    try:
        root = project_root or Path.cwd()
        base_path = root / "config" / "base.yaml"
        local_path = root / "config" / "local.yaml"

        if base_path.exists():
            with base_path.open("r", encoding="utf-8") as f:
                base = yaml.safe_load(f) or {}
        if local_path.exists():
            with local_path.open("r", encoding="utf-8") as f:
                local = yaml.safe_load(f) or {}
    except Exception as e:
        logger.warning(f"Failed to load user config: {e}")

    merged = _deep_merge(base, local)
    _CONFIG_CACHE = merged
    return merged


def get_property_extraction_config() -> dict[str, Any]:
    cfg = get_user_config()
    return cfg.get("property_extraction", {})

