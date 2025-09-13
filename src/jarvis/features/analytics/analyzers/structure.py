"""
Vault structure analyzer for organization pattern detection.
"""

import logging
import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..errors import AnalysisTimeoutError, InsufficientDataError
from ..models import (
    AnalyticsError,
    DepthMetrics,
    OrganizationMethod,
    OrganizationPattern,
)

logger = logging.getLogger(__name__)


@dataclass
class ContentCluster:
    name: str
    pattern: str
    paths: list[str]
    file_count: int
    coherence_score: float
    depth_range: tuple[int, int]
    keywords: list[str]


class VaultStructureAnalyzer:
    """Analyzes vault organizational patterns and structure."""

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}
        self.max_processing_time = self.config.get("max_processing_time_seconds", 15)

        self._para_indicators = {
            "folders": ["projects", "areas", "resources", "archive", "inbox"],
            "patterns": [
                r"^\d+\.\s*projects?",
                r"^\d+\.\s*areas?",
                r"^\d+\.\s*resources?",
                r"^\d+\.\s*archive",
                r"inbox|capture",
            ],
        }

        self._johnny_decimal_patterns = [
            r"^\d{2}[-\s]",
            r"^\d{2}\.\d{2}",
            r"^\d{1,2}[-\s]\d{1,2}[-\s]",
        ]

        self._zettelkasten_patterns = [
            r"^\d{12,14}",
            r"^\d{4}\.\d{2}\.\d{2}",
            r"^[A-Z]{1,3}\d{3,6}",
        ]

        logger.debug("VaultStructureAnalyzer initialized")

    async def detect_organization_method(self, files: list[Path]) -> OrganizationPattern:
        start_time = time.time()

        try:
            if len(files) < 5:
                raise InsufficientDataError("structure_analyzer", 5, len(files))

            folders = self._extract_folders(files)
            folder_names = [f.name.lower() for f in folders]

            scores = {
                OrganizationMethod.PARA: self._score_para_method(folder_names, files),
                OrganizationMethod.JOHNNY_DECIMAL: self._score_johnny_decimal(
                    folder_names, files
                ),
                OrganizationMethod.ZETTELKASTEN: self._score_zettelkasten(files),
                OrganizationMethod.TOPIC_BASED: self._score_topic_based(folder_names),
                OrganizationMethod.CHRONOLOGICAL: self._score_chronological(files),
            }

            if time.time() - start_time > self.max_processing_time:
                raise AnalysisTimeoutError("structure_analyzer", self.max_processing_time)

            best_method = max(scores.keys(), key=lambda k: scores[k]["score"])
            best_score_data = scores[best_method]

            high_scores = [
                method for method, data in scores.items() if data["score"] > 0.3
            ]

            if len(high_scores) > 1:
                method = OrganizationMethod.MIXED
                confidence = max(0.4, best_score_data["score"] * 0.8)
            elif best_score_data["score"] < 0.2:
                method = OrganizationMethod.UNKNOWN
                confidence = 0.1
            else:
                method = best_method
                confidence = best_score_data["score"]

            return OrganizationPattern(
                method=method,
                confidence=confidence,
                indicators=best_score_data["indicators"],
                folder_patterns=best_score_data["patterns"],
                exceptions=self._find_exceptions(files, method),
            )

        except Exception as e:
            if isinstance(e, (InsufficientDataError, AnalysisTimeoutError)):
                raise
            logger.error(f"Organization detection error: {e}")
            raise AnalyticsError(
                f"Failed to detect organization method: {e}",
                "structure_analyzer",
                "organization_detection",
            )

    async def calculate_depth_metrics(self, files: list[Path]) -> DepthMetrics:
        try:
            depth_counts = Counter()
            files_by_depth = defaultdict(list)

            for file_path in files:
                depth = len(file_path.parts) - 1
                depth_counts[depth] += 1
                files_by_depth[depth].append(str(file_path))

            if not depth_counts:
                raise InsufficientDataError("structure_analyzer", 1, 0)

            max_depth = max(depth_counts.keys())
            avg_depth = sum(d * c for d, c in depth_counts.items()) / sum(
                depth_counts.values()
            )

            depth_variety = len(depth_counts)
            depth_distribution_evenness = self._calculate_evenness(
                list(depth_counts.values())
            )

            complexity_score = min(
                1.0,
                (
                    (max_depth / 10.0) * 0.4
                    + (depth_variety / 8.0) * 0.3
                    + depth_distribution_evenness * 0.3
                ),
            )

            organization_score = self._calculate_organization_score(
                depth_counts, avg_depth, max_depth
            )

            return DepthMetrics(
                depth_distribution=dict(depth_counts),
                files_by_depth=dict(files_by_depth),
                complexity_score=complexity_score,
                organization_score=organization_score,
            )

        except Exception as e:
            if isinstance(e, InsufficientDataError):
                raise
            logger.error(f"Depth metrics calculation error: {e}")
            raise AnalyticsError(
                f"Failed to calculate depth metrics: {e}",
                "structure_analyzer",
                "depth_calculation",
            )

    # --- helpers ---
    def _extract_folders(self, files: list[Path]) -> list[Path]:
        return sorted({p.parent for p in files})

    def _score_para_method(self, folder_names: list[str], files: list[Path]) -> dict:
        indicators = []
        patterns = []
        score = 0.0
        folder_set = set(folder_names)
        if any(name in folder_set for name in self._para_indicators["folders"]):
            score += 0.3
            indicators.append("contains PARA folders")
        for pat in self._para_indicators["patterns"]:
            patterns.append(pat)
        return {"score": score, "indicators": indicators, "patterns": patterns}

    def _score_johnny_decimal(self, folder_names: list[str], files: list[Path]) -> dict:
        patterns = self._johnny_decimal_patterns
        match_count = sum(1 for f in folder_names if any(re.match(p, f) for p in patterns))
        score = min(1.0, match_count / max(1, len(folder_names)))
        return {"score": score, "indicators": ["johnny-decimal patterns"], "patterns": patterns}

    def _score_zettelkasten(self, files: list[Path]) -> dict:
        patterns = self._zettelkasten_patterns
        match_count = sum(1 for f in files if any(re.match(p, f.name) for p in patterns))
        score = min(1.0, match_count / max(1, len(files)))
        return {"score": score, "indicators": ["zettelkasten ids"], "patterns": patterns}

    def _score_topic_based(self, folder_names: list[str]) -> dict:
        common_topics = {"notes", "articles", "references", "research"}
        overlap = len(common_topics & set(folder_names))
        score = min(1.0, overlap / 5.0)
        return {"score": score, "indicators": ["topic folders"], "patterns": []}

    def _score_chronological(self, files: list[Path]) -> dict:
        patterns = [r"^\d{4}"]
        match_count = sum(1 for f in files if any(re.match(p, f.name) for p in patterns))
        score = min(1.0, match_count / max(1, len(files)))
        return {"score": score, "indicators": ["year folders"], "patterns": patterns}

    def _find_exceptions(self, files: list[Path], method: OrganizationMethod) -> list[str]:
        return []

    def _calculate_evenness(self, counts: list[int]) -> float:
        if not counts:
            return 0.0
        total = sum(counts)
        max_count = max(counts)
        return 1.0 - (max_count / max(1, total))

    def _calculate_organization_score(
        self, depth_counts: Counter, avg_depth: float, max_depth: int
    ) -> float:
        return max(0.0, min(1.0, 1.0 - (avg_depth / max(1.0, max_depth)) * 0.5))

