"""Import shim for quality analyzer (moved to features)."""

"""
Content quality analyzer for note assessment and improvement suggestions.

Analyzes individual notes and vault-wide quality patterns,
providing detailed scoring and actionable improvement recommendations.
"""

import logging
import re
import time
from pathlib import Path
from typing import Any

from jarvis.core.interfaces import IGraphDatabase, IVaultReader

from ..errors import ServiceUnavailableError
from ..models import (
    AnalyticsError,
    ConnectionMetrics,
    QualityGap,
    QualityLevel,
    QualityScore,
)

logger = logging.getLogger(__name__)


class ContentQualityAnalyzer:
    """Analyzes content quality and completeness across vault notes."""

    def __init__(
        self,
        vault_reader: IVaultReader | None = None,
        graph_db: IGraphDatabase | None = None,
        config: dict[str, Any] | None = None,
    ):
        self.vault_reader = vault_reader
        self.graph_db = graph_db
        self.config = config or {}

        self.max_processing_time = self.config.get("max_processing_time_seconds", 15)
        self.connection_weight = self.config.get("connection_weight", 0.3)
        self.freshness_weight = self.config.get("freshness_weight", 0.2)
        self.scoring_algorithm = self.config.get("scoring_algorithm", "comprehensive")

        self.header_pattern = re.compile(r'^#{1,6}\s+.+', re.MULTILINE)
        self.list_pattern = re.compile(r'^\s*[-*+]\s+.+|^\s*\d+\.\s+.+', re.MULTILINE)
        self.link_pattern = re.compile(r'\[\[([^\]]+)\]\]|\[([^\]]+)\]\([^\)]+\)')
        self.tag_pattern = re.compile(r'#[a-zA-Z][a-zA-Z0-9_/-]*')
        self.code_pattern = re.compile(r'```[\s\S]*?```|`[^`]+`')
        self.todo_pattern = re.compile(r'- \[ \]|TODO:|FIXME:|NOTE:')

        logger.debug("ContentQualityAnalyzer initialized")

    async def assess_note_quality(
        self,
        content: str,
        metadata: dict[str, Any],
        note_path: str = "",
    ) -> QualityScore:
        start_time = time.time()

        try:
            if not content or len(content.strip()) < 10:
                return self._create_minimal_quality_score(content, metadata, note_path)

            content_metrics = await self._analyze_content_structure(content)
            connection_metrics = await self._analyze_connections(content, note_path)
            freshness_score = self._calculate_freshness_score(metadata)

            completeness = self._calculate_completeness_score(content, content_metrics)
            structure = self._calculate_structure_score(content_metrics)
            connections = self._calculate_connections_score(connection_metrics)
            freshness = freshness_score

            if self.scoring_algorithm == "comprehensive":
                overall_score = self._calculate_comprehensive_score(
                    completeness, structure, connections, freshness
                )
            else:
                overall_score = (
                    completeness + structure + connections + freshness
                ) / 4

            quality_level = self._determine_quality_level(overall_score)

            suggestions = self._generate_improvement_suggestions(
                content,
                content_metrics,
                connection_metrics,
                completeness,
                structure,
                connections,
                freshness,
            )

            confidence = self._calculate_assessment_confidence(
                content, content_metrics, connection_metrics
            )

            if time.time() - start_time > self.max_processing_time:
                logger.warning(f"Quality assessment timeout for {note_path}")

            return QualityScore(
                overall_score=overall_score,
                level=quality_level,
                completeness=completeness,
                structure=structure,
                connections=connections,
                freshness=freshness,
                word_count=content_metrics["word_count"],
                link_count=content_metrics["link_count"],
                backlink_count=connection_metrics.inbound_links,
                last_modified=metadata.get("last_modified", 0.0),
                headers_count=content_metrics["headers_count"],
                list_items_count=content_metrics["list_items_count"],
                connection_metrics=connection_metrics,
                suggestions=suggestions,
                confidence=confidence,
                domain=self._infer_domain(content, note_path),
                tags=content_metrics["tags"],
            )

        except Exception as e:
            logger.error(f"Quality assessment error for {note_path}: {e}")
            raise AnalyticsError(
                f"Failed to assess note quality: {e}", "quality_analyzer", "note_assessment"
            )

    async def calculate_connection_density(self, note_path: str) -> ConnectionMetrics:
        try:
            if not self.graph_db or not self.graph_db.is_healthy:
                return self._create_minimal_connection_metrics()

            graph_data = self.graph_db.get_note_graph(note_path, depth=2)

            outbound_links = len(graph_data.get("outbound_links", []))
            inbound_links = len(graph_data.get("inbound_links", []))
            bidirectional = sum(
                1 for l in graph_data.get("relationships", []) if l.get("bidirectional")
            )
            broken = len(graph_data.get("broken_links", []))

            density = min(1.0, (outbound_links + inbound_links + bidirectional) / 50.0)

            return ConnectionMetrics(
                outbound_links=outbound_links,
                inbound_links=inbound_links,
                bidirectional_links=bidirectional,
                broken_links=broken,
                connection_density=density,
                hub_score=min(1.0, inbound_links / 25.0),
                authority_score=min(1.0, inbound_links / 25.0),
            )
        except Exception as e:
            logger.error(f"Connection density calculation error: {e}")
            return self._create_minimal_connection_metrics()

    # --- internal analysis helpers ---
    async def _analyze_content_structure(self, content: str) -> dict[str, Any]:
        headers = self.header_pattern.findall(content)
        list_items = self.list_pattern.findall(content)
        links = self.link_pattern.findall(content)
        tags = self.tag_pattern.findall(content)
        code_blocks = self.code_pattern.findall(content)
        todos = self.todo_pattern.findall(content)

        return {
            "headers_count": len(headers),
            "list_items_count": len(list_items),
            "link_count": len(links),
            "tags": [t.strip("#") for t in tags],
            "word_count": len(content.split()),
            "code_block_count": len(code_blocks),
            "todo_count": len(todos),
        }

    async def _analyze_connections(self, content: str, note_path: str) -> ConnectionMetrics:
        if not self.graph_db:
            return self._create_minimal_connection_metrics()
        return await self.calculate_connection_density(note_path)

    def _calculate_freshness_score(self, metadata: dict[str, Any]) -> float:
        last_modified = float(metadata.get("last_modified", 0.0))
        age_days = max(0.0, (time.time() - last_modified) / (60 * 60 * 24)) if last_modified else 365.0
        return max(0.0, 1.0 - min(1.0, age_days / 365.0))

    def _calculate_completeness_score(self, content: str, m: dict[str, Any]) -> float:
        base = min(1.0, len(content.split()) / 800.0)
        structure_bonus = min(0.2, (m["headers_count"] + m["list_items_count"]) / 50.0)
        return max(0.0, min(1.0, base + structure_bonus))

    def _calculate_structure_score(self, m: dict[str, Any]) -> float:
        structure = min(1.0, (m["headers_count"] * 0.02) + (m["list_items_count"] * 0.01))
        return structure

    def _calculate_connections_score(self, cm: ConnectionMetrics) -> float:
        base = min(1.0, cm.connection_density)
        return base

    def _calculate_comprehensive_score(self, c: float, s: float, l: float, f: float) -> float:
        return max(0.0, min(1.0, c * 0.4 + s * 0.2 + l * self.connection_weight + f * self.freshness_weight))

    def _determine_quality_level(self, overall: float) -> QualityLevel:
        if overall < 0.25:
            return QualityLevel.SEEDLING
        if overall < 0.5:
            return QualityLevel.GROWING
        if overall < 0.75:
            return QualityLevel.MATURE
        return QualityLevel.COMPREHENSIVE

    def _generate_improvement_suggestions(
        self,
        content: str,
        m: dict[str, Any],
        cm: ConnectionMetrics,
        completeness: float,
        structure: float,
        connections: float,
        freshness: float,
    ) -> list[str]:
        suggestions: list[str] = []
        if completeness < 0.6:
            suggestions.append("Add more detailed explanations and examples")
        if structure < 0.5:
            suggestions.append("Improve structure with headers and lists")
        if connections < 0.5:
            suggestions.append("Add links to related notes to improve connectivity")
        if freshness < 0.5:
            suggestions.append("Update the note to reflect current information")
        return suggestions

    def _calculate_assessment_confidence(
        self, content: str, m: dict[str, Any], cm: ConnectionMetrics
    ) -> float:
        return min(1.0, 0.5 + (m["headers_count"] + m["list_items_count"]) / 100.0)

    def _infer_domain(self, content: str, note_path: str) -> str | None:
        return None

    def _create_minimal_quality_score(
        self, content: str, metadata: dict[str, Any], note_path: str
    ) -> QualityScore:
        cm = self._create_minimal_connection_metrics()
        return QualityScore(
            overall_score=0.1,
            level=QualityLevel.SEEDLING,
            completeness=0.1,
            structure=0.1,
            connections=cm.connection_density,
            freshness=self._calculate_freshness_score(metadata),
            word_count=len(content.split()),
            link_count=0,
            backlink_count=0,
            last_modified=metadata.get("last_modified", 0.0),
            headers_count=0,
            list_items_count=0,
            connection_metrics=cm,
            suggestions=["Add content to this note"],
            confidence=0.5,
            domain=None,
            tags=[],
        )

    def _create_minimal_connection_metrics(self) -> ConnectionMetrics:
        return ConnectionMetrics(
            outbound_links=0,
            inbound_links=0,
            bidirectional_links=0,
            broken_links=0,
            connection_density=0.0,
            hub_score=0.0,
            authority_score=0.0,
        )

