"""Import shim for structure analyzer (moved to features)."""

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
            for p in files:
                depth = len(p.parts) - 1
                depth_counts[depth] += 1
                files_by_depth[depth].append(str(p))

            total_files = len(files)
            if total_files == 0:
                raise InsufficientDataError("structure_analyzer", 1, 0)

            max_depth = max(depth_counts.keys(), default=0)
            # Simple heuristics for scores
            organization_score = min(1.0, 1.0 - (max_depth / 20.0))
            complexity_score = min(1.0, max_depth / 10.0)

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
                "depth_metrics",
            )

    # --- internal helpers ---
    def _extract_folders(self, files: list[Path]) -> list[Path]:
        return sorted({p.parent for p in files})

    def _score_para_method(self, folder_names: list[str], files: list[Path]) -> dict:
        indicators = []
        score = 0.0

        found_folders = set(folder_names)
        matches = [f for f in self._para_indicators["folders"] if f in found_folders]
        if matches:
            indicators.append(f"Found PARA folders: {', '.join(matches)}")
            score += 0.3

        pattern_matches = [
            p for p in self._para_indicators["patterns"] if any(re.search(p, f) for f in folder_names)
        ]
        if pattern_matches:
            indicators.append("Found PARA-style numbering patterns")
            score += 0.3

        project_like = sum(1 for f in folder_names if re.search(r"projects?|goals|outcomes", f))
        if project_like > 3:
            indicators.append("Multiple project-like folders detected")
            score += 0.2

        archive_like = sum(1 for f in folder_names if re.search(r"archive|old|completed", f))
        if archive_like > 1:
            indicators.append("Archive-like folders present")
            score += 0.2

        return {"score": min(1.0, score), "indicators": indicators, "patterns": pattern_matches}

    def _score_johnny_decimal(self, folder_names: list[str], files: list[Path]) -> dict:
        indicators = []
        pattern_matches = [p for p in self._johnny_decimal_patterns if any(re.search(p, f) for f in folder_names)]
        score = min(1.0, len(pattern_matches) * 0.4)
        if pattern_matches:
            indicators.append("Johnny Decimal numbering patterns detected")
        return {"score": score, "indicators": indicators, "patterns": pattern_matches}

    def _score_zettelkasten(self, files: list[Path]) -> dict:
        indicators = []
        score = 0.0
        # Look for common Zettelkasten patterns in filenames
        filenames = [p.name for p in files]
        pattern_matches = [p for p in self._zettelkasten_patterns if any(re.search(p, f) for f in filenames)]
        if pattern_matches:
            indicators.append("Zettelkasten-style identifiers present in filenames")
            score += 0.6
        return {"score": min(1.0, score), "indicators": indicators, "patterns": pattern_matches}

    def _score_topic_based(self, folder_names: list[str]) -> dict:
        indicators = []
        # Topic-based organization inferred by many distinct folder names
        diversity = len(set(folder_names))
        score = min(1.0, diversity / 50.0)
        if score > 0.3:
            indicators.append("High diversity of folder names suggests topic-based organization")
        return {"score": score, "indicators": indicators, "patterns": []}

    def _score_chronological(self, files: list[Path]) -> dict:
        indicators = []
        score = 0.0
        # Simple: look for date-like patterns in folders or filenames
        date_pattern = re.compile(r"\b(19|20)\d{2}([-/_.])?(0[1-9]|1[0-2])\b")
        matches = 0
        for p in files:
            if date_pattern.search(str(p)):
                matches += 1
        if matches > max(5, len(files) * 0.1):
            indicators.append("Many date-like paths detected")
            score = min(1.0, matches / max(10.0, len(files)))
        return {"score": score, "indicators": indicators, "patterns": [r"YYYY-MM"]}

    def _find_exceptions(self, files: list[Path], method: OrganizationMethod) -> list[str]:
        return []

    def _calculate_evenness(self, counts: list[int]) -> float:
        if not counts:
            return 0.0
        total = sum(counts)
        if total == 0:
            return 0.0
        target = total / len(counts)
        variance = sum((c - target) ** 2 for c in counts) / len(counts)
        return max(0.0, min(1.0, 1.0 - (variance / (target + 1e-6))))

    def _calculate_organization_score(
        self, folder_distribution: dict[str, int], depth_metrics: DepthMetrics
    ) -> float:
        evenness = self._calculate_evenness(list(folder_distribution.values()))
        depth_penalty = min(0.5, (max(depth_metrics.depth_distribution.keys(), default=0) / 20.0))
        return max(0.0, min(1.0, evenness * (1.0 - depth_penalty)))

