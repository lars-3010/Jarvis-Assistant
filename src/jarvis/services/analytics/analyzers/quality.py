"""
Content quality analyzer for note assessment and improvement suggestions.

This module analyzes individual notes and vault-wide quality patterns,
providing detailed scoring and actionable improvement recommendations.
"""

import re
import time
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any
from collections import Counter, defaultdict
from dataclasses import asdict
import logging

from jarvis.services.analytics.models import (
    QualityScore, QualityLevel, ConnectionMetrics, QualityGap, 
    QualityTrend, AnalyticsError
)
from jarvis.services.analytics.errors import (
    InsufficientDataError, AnalysisTimeoutError, ServiceUnavailableError
)
from jarvis.core.interfaces import IGraphDatabase, IVaultReader


logger = logging.getLogger(__name__)


class ContentQualityAnalyzer:
    """
    Analyzes content quality and completeness across vault notes.
    
    Provides detailed quality scoring using multiple criteria:
    - Completeness: content depth and thoroughness
    - Structure: organization and formatting quality  
    - Connections: links to other notes
    - Freshness: recency and relevance
    """
    
    def __init__(
        self,
        vault_reader: Optional[IVaultReader] = None,
        graph_db: Optional[IGraphDatabase] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """Initialize the quality analyzer."""
        self.vault_reader = vault_reader
        self.graph_db = graph_db
        self.config = config or {}
        
        # Configuration
        self.max_processing_time = self.config.get("max_processing_time_seconds", 15)
        self.connection_weight = self.config.get("connection_weight", 0.3)
        self.freshness_weight = self.config.get("freshness_weight", 0.2)
        self.scoring_algorithm = self.config.get("scoring_algorithm", "comprehensive")
        
        # Quality thresholds
        self.quality_thresholds = {
            QualityLevel.SEEDLING: (0.0, 0.25),
            QualityLevel.GROWING: (0.25, 0.50),
            QualityLevel.MATURE: (0.50, 0.75),
            QualityLevel.COMPREHENSIVE: (0.75, 1.0)
        }
        
        # Content analysis patterns
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
        metadata: Dict[str, Any],
        note_path: str = ""
    ) -> QualityScore:
        """
        Assess quality of a single note using comprehensive criteria.
        
        Args:
            content: Note content as string
            metadata: Note metadata (modification time, etc.)
            note_path: Path to the note for context
            
        Returns:
            QualityScore with detailed assessment
        """
        start_time = time.time()
        
        try:
            if not content or len(content.strip()) < 10:
                return self._create_minimal_quality_score(content, metadata, note_path)
            
            # Analyze content components
            content_metrics = await self._analyze_content_structure(content)
            connection_metrics = await self._analyze_connections(content, note_path)
            freshness_score = self._calculate_freshness_score(metadata)
            
            # Calculate component scores
            completeness = self._calculate_completeness_score(content, content_metrics)
            structure = self._calculate_structure_score(content_metrics)
            connections = self._calculate_connections_score(connection_metrics)
            freshness = freshness_score
            
            # Calculate overall score using weighted average
            if self.scoring_algorithm == "comprehensive":
                overall_score = self._calculate_comprehensive_score(
                    completeness, structure, connections, freshness
                )
            else:
                overall_score = (completeness + structure + connections + freshness) / 4
            
            # Determine quality level
            quality_level = self._determine_quality_level(overall_score)
            
            # Generate improvement suggestions
            suggestions = self._generate_improvement_suggestions(
                content, content_metrics, connection_metrics, 
                completeness, structure, connections, freshness
            )
            
            # Calculate confidence based on content analysis reliability
            confidence = self._calculate_assessment_confidence(
                content, content_metrics, connection_metrics
            )
            
            # Check processing time
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
                tags=content_metrics["tags"]
            )
            
        except Exception as e:
            logger.error(f"Quality assessment error for {note_path}: {e}")
            raise AnalyticsError(f"Failed to assess note quality: {e}",
                               "quality_analyzer", "note_assessment")
    
    async def calculate_connection_density(self, note_path: str) -> ConnectionMetrics:
        """
        Calculate how well-connected a note is to others in the vault.
        
        Args:
            note_path: Path to the note
            
        Returns:
            ConnectionMetrics with detailed connection analysis
        """
        try:
            if not self.graph_db or not self.graph_db.is_healthy:
                logger.warning("Graph database unavailable for connection analysis")
                return self._create_minimal_connection_metrics()
            
            # Get graph data for the note
            graph_data = self.graph_db.get_note_graph(note_path, depth=2)
            
            # Extract connection information
            outbound_links = len(graph_data.get("outbound_links", []))
            inbound_links = len(graph_data.get("inbound_links", []))
            
            # Calculate bidirectional links
            outbound_targets = set(graph_data.get("outbound_links", []))
            inbound_sources = set(graph_data.get("inbound_links", []))
            bidirectional_links = len(outbound_targets.intersection(inbound_sources))
            
            # Identify broken links (if available)
            broken_links = len(graph_data.get("broken_links", []))
            
            # Calculate connection density (0.0 to 1.0)
            total_possible_connections = max(1, graph_data.get("total_notes", 1) - 1)
            total_connections = outbound_links + inbound_links
            connection_density = min(1.0, total_connections / total_possible_connections)
            
            # Calculate hub score (how central this note is)
            hub_score = min(1.0, outbound_links / max(1, total_possible_connections * 0.1))
            
            # Calculate authority score (how referenced this note is)
            authority_score = min(1.0, inbound_links / max(1, total_possible_connections * 0.05))
            
            return ConnectionMetrics(
                outbound_links=outbound_links,
                inbound_links=inbound_links,
                bidirectional_links=bidirectional_links,
                broken_links=broken_links,
                connection_density=connection_density,
                hub_score=hub_score,
                authority_score=authority_score
            )
            
        except Exception as e:
            logger.error(f"Connection density calculation error: {e}")
            return self._create_minimal_connection_metrics()
    
    async def identify_quality_gaps(self, vault_files: List[Path]) -> List[QualityGap]:
        """
        Identify notes or areas needing quality improvement.
        
        Args:
            vault_files: List of all vault files
            
        Returns:
            List of QualityGap objects with improvement opportunities
        """
        try:
            if not self.vault_reader:
                raise ServiceUnavailableError("vault_reader", "quality gap identification")
            
            quality_gaps = []
            
            # Sample files if too many (for performance)
            sample_files = vault_files
            if len(vault_files) > 100:
                import random
                sample_files = random.sample(vault_files, 100)
            
            # Analyze each file for quality gaps
            for file_path in sample_files:
                try:
                    content, metadata = self.vault_reader.read_file(str(file_path))
                    quality_score = await self.assess_note_quality(content, metadata, str(file_path))
                    
                    # Identify specific quality issues
                    issues = self._identify_specific_issues(content, quality_score)
                    
                    if issues:
                        gap = QualityGap(
                            note_path=str(file_path),
                            current_quality=quality_score.overall_score,
                            potential_quality=self._estimate_potential_quality(quality_score),
                            gap_type=self._determine_primary_gap_type(issues),
                            priority=self._calculate_gap_priority(quality_score, issues),
                            issues=issues,
                            suggestions=quality_score.suggestions,
                            estimated_effort=self._estimate_improvement_effort(issues),
                            domain=quality_score.domain,
                            related_notes=self._find_related_notes(content, file_path)
                        )
                        quality_gaps.append(gap)
                
                except Exception as e:
                    logger.warning(f"Failed to analyze {file_path}: {e}")
                    continue
            
            # Sort by priority and potential impact
            quality_gaps.sort(
                key=lambda g: (g.priority == "high", g.potential_quality - g.current_quality),
                reverse=True
            )
            
            return quality_gaps[:50]  # Return top 50 gaps
            
        except Exception as e:
            logger.error(f"Quality gap identification error: {e}")
            raise AnalyticsError(f"Failed to identify quality gaps: {e}",
                               "quality_analyzer", "gap_identification")
    
    async def _analyze_content_structure(self, content: str) -> Dict[str, Any]:
        """Analyze structural elements of note content."""
        # Word count
        words = re.findall(r'\b\w+\b', content)
        word_count = len(words)
        
        # Headers
        headers = self.header_pattern.findall(content)
        headers_count = len(headers)
        
        # Lists
        list_items = self.list_pattern.findall(content)
        list_items_count = len(list_items)
        
        # Links
        links = self.link_pattern.findall(content)
        link_count = len(links)
        
        # Tags
        tags = self.tag_pattern.findall(content)
        
        # Code blocks
        code_blocks = self.code_pattern.findall(content)
        code_blocks_count = len(code_blocks)
        
        # TODOs and action items
        todos = self.todo_pattern.findall(content)
        todos_count = len(todos)
        
        # Calculate reading time (average 200 words per minute)
        reading_time_minutes = max(1, word_count / 200)
        
        # Analyze content depth
        paragraph_count = len([p for p in content.split('\n\n') if p.strip()])
        avg_paragraph_length = word_count / max(1, paragraph_count)
        
        return {
            "word_count": word_count,
            "headers_count": headers_count,
            "list_items_count": list_items_count,
            "link_count": link_count,
            "tags": tags,
            "code_blocks_count": code_blocks_count,
            "todos_count": todos_count,
            "reading_time_minutes": reading_time_minutes,
            "paragraph_count": paragraph_count,
            "avg_paragraph_length": avg_paragraph_length,
            "headers": headers,
            "links": links
        }
    
    async def _analyze_connections(self, content: str, note_path: str) -> ConnectionMetrics:
        """Analyze note connections using available services."""
        if self.graph_db and self.graph_db.is_healthy:
            return await self.calculate_connection_density(note_path)
        else:
            # Fallback to content-based analysis
            links = self.link_pattern.findall(content)
            outbound_links = len(links)
            
            return ConnectionMetrics(
                outbound_links=outbound_links,
                inbound_links=0,  # Can't determine without graph
                bidirectional_links=0,
                broken_links=0,
                connection_density=min(0.1, outbound_links / 50),  # Rough estimate
                hub_score=min(0.1, outbound_links / 20),
                authority_score=0.0  # Can't determine without graph
            )
    
    def _calculate_completeness_score(self, content: str, metrics: Dict[str, Any]) -> float:
        """Calculate how complete and thorough the content is."""
        score = 0.0
        
        # Word count contribution (0.0 to 0.4)
        word_count = metrics["word_count"]
        if word_count >= 500:
            score += 0.4
        elif word_count >= 200:
            score += 0.3
        elif word_count >= 100:
            score += 0.2
        elif word_count >= 50:
            score += 0.1
        
        # Structure contribution (0.0 to 0.3)
        if metrics["headers_count"] >= 3:
            score += 0.15
        elif metrics["headers_count"] >= 1:
            score += 0.1
        
        if metrics["list_items_count"] >= 3:
            score += 0.1
        elif metrics["list_items_count"] >= 1:
            score += 0.05
        
        if metrics["paragraph_count"] >= 3:
            score += 0.05
        
        # Content depth contribution (0.0 to 0.3)
        avg_paragraph_length = metrics["avg_paragraph_length"]
        if avg_paragraph_length >= 50:
            score += 0.15
        elif avg_paragraph_length >= 25:
            score += 0.1
        elif avg_paragraph_length >= 15:
            score += 0.05
        
        # Additional content types
        if metrics["code_blocks_count"] > 0:
            score += 0.05
        if len(metrics["tags"]) >= 2:
            score += 0.05
        if metrics["reading_time_minutes"] >= 3:
            score += 0.05
        
        return min(1.0, score)
    
    def _calculate_structure_score(self, metrics: Dict[str, Any]) -> float:
        """Calculate how well-structured and organized the content is."""
        score = 0.0
        
        # Header structure (0.0 to 0.4)
        headers_count = metrics["headers_count"]
        if headers_count >= 5:
            score += 0.4
        elif headers_count >= 3:
            score += 0.3
        elif headers_count >= 1:
            score += 0.2
        
        # Check for hierarchical headers
        headers = metrics.get("headers", [])
        if len(headers) >= 2:
            # Simple check for hierarchy (# then ##, etc.)
            header_levels = [len(h.split()[0]) for h in headers if h.startswith('#')]
            if len(set(header_levels)) > 1:
                score += 0.1
        
        # List usage (0.0 to 0.2)
        if metrics["list_items_count"] >= 5:
            score += 0.2
        elif metrics["list_items_count"] >= 2:
            score += 0.15
        elif metrics["list_items_count"] >= 1:
            score += 0.1
        
        # Content organization (0.0 to 0.2)
        if metrics["paragraph_count"] >= 3:
            score += 0.1
        
        # Balanced paragraph length indicates good structure
        avg_length = metrics["avg_paragraph_length"]
        if 20 <= avg_length <= 100:
            score += 0.1
        elif 10 <= avg_length <= 150:
            score += 0.05
        
        # Tags for organization (0.0 to 0.1)
        tag_count = len(metrics["tags"])
        if tag_count >= 3:
            score += 0.1
        elif tag_count >= 1:
            score += 0.05
        
        # Action items show structure (0.0 to 0.1)
        if metrics["todos_count"] > 0:
            score += 0.1
        
        return min(1.0, score)
    
    def _calculate_connections_score(self, connection_metrics: ConnectionMetrics) -> float:
        """Calculate score based on note connections."""
        score = 0.0
        
        # Outbound links (0.0 to 0.4)
        outbound = connection_metrics.outbound_links
        if outbound >= 10:
            score += 0.4
        elif outbound >= 5:
            score += 0.3
        elif outbound >= 2:
            score += 0.2
        elif outbound >= 1:
            score += 0.1
        
        # Inbound links (0.0 to 0.3)
        inbound = connection_metrics.inbound_links
        if inbound >= 5:
            score += 0.3
        elif inbound >= 2:
            score += 0.2
        elif inbound >= 1:
            score += 0.1
        
        # Bidirectional links bonus (0.0 to 0.2)
        bidirectional = connection_metrics.bidirectional_links
        if bidirectional >= 3:
            score += 0.2
        elif bidirectional >= 1:
            score += 0.1
        
        # Connection density (0.0 to 0.1)
        score += connection_metrics.connection_density * 0.1
        
        # Penalty for broken links
        if connection_metrics.broken_links > 0:
            penalty = min(0.2, connection_metrics.broken_links * 0.05)
            score = max(0.0, score - penalty)
        
        return min(1.0, score)
    
    def _calculate_freshness_score(self, metadata: Dict[str, Any]) -> float:
        """Calculate freshness score based on modification time."""
        last_modified = metadata.get("last_modified", 0.0)
        if last_modified == 0.0:
            return 0.5  # Default for unknown modification time
        
        current_time = time.time()
        age_days = (current_time - last_modified) / (24 * 3600)
        
        # Freshness scoring
        if age_days <= 7:
            return 1.0
        elif age_days <= 30:
            return 0.8
        elif age_days <= 90:
            return 0.6
        elif age_days <= 365:
            return 0.4
        elif age_days <= 730:
            return 0.2
        else:
            return 0.1
    
    def _calculate_comprehensive_score(
        self, 
        completeness: float, 
        structure: float, 
        connections: float, 
        freshness: float
    ) -> float:
        """Calculate overall score using comprehensive weighting."""
        # Base weights
        base_weights = {
            "completeness": 0.4,
            "structure": 0.3,
            "connections": self.connection_weight,
            "freshness": self.freshness_weight
        }
        
        # Normalize weights to sum to 1.0
        total_weight = sum(base_weights.values())
        normalized_weights = {k: v / total_weight for k, v in base_weights.items()}
        
        # Calculate weighted score
        weighted_score = (
            completeness * normalized_weights["completeness"] +
            structure * normalized_weights["structure"] +
            connections * normalized_weights["connections"] +
            freshness * normalized_weights["freshness"]
        )
        
        return min(1.0, weighted_score)
    
    def _determine_quality_level(self, overall_score: float) -> QualityLevel:
        """Determine quality level based on overall score."""
        for level, (min_score, max_score) in self.quality_thresholds.items():
            if min_score <= overall_score <= max_score:
                return level
        return QualityLevel.SEEDLING
    
    def _generate_improvement_suggestions(
        self,
        content: str,
        content_metrics: Dict[str, Any],
        connection_metrics: ConnectionMetrics,
        completeness: float,
        structure: float,
        connections: float,
        freshness: float
    ) -> List[str]:
        """Generate specific improvement suggestions."""
        suggestions = []
        
        # Completeness suggestions
        if completeness < 0.5:
            if content_metrics["word_count"] < 100:
                suggestions.append("Expand content with more detailed explanations")
            if content_metrics["headers_count"] == 0:
                suggestions.append("Add headers to organize content structure")
            if content_metrics["paragraph_count"] < 3:
                suggestions.append("Break content into more paragraphs for readability")
        
        # Structure suggestions
        if structure < 0.5:
            if content_metrics["headers_count"] < 2:
                suggestions.append("Use headers to create clear content hierarchy")
            if content_metrics["list_items_count"] == 0:
                suggestions.append("Use bullet points or numbered lists for key information")
            if len(content_metrics["tags"]) == 0:
                suggestions.append("Add relevant tags for better categorization")
        
        # Connection suggestions
        if connections < 0.3:
            if connection_metrics.outbound_links == 0:
                suggestions.append("Link to related notes to create connections")
            elif connection_metrics.outbound_links < 3:
                suggestions.append("Add more links to related concepts and notes")
            if connection_metrics.broken_links > 0:
                suggestions.append("Fix broken links to maintain note connectivity")
        
        # Freshness suggestions
        if freshness < 0.4:
            suggestions.append("Review and update content to maintain relevance")
        
        # Content-specific suggestions
        if content_metrics["todos_count"] > 0:
            suggestions.append("Complete or organize TODO items")
        
        return suggestions[:5]  # Return top 5 suggestions
    
    def _calculate_assessment_confidence(
        self,
        content: str,
        content_metrics: Dict[str, Any],
        connection_metrics: ConnectionMetrics
    ) -> float:
        """Calculate confidence in the quality assessment."""
        confidence = 0.7  # Base confidence
        
        # Higher confidence for longer content
        if content_metrics["word_count"] >= 200:
            confidence += 0.1
        elif content_metrics["word_count"] < 50:
            confidence -= 0.2
        
        # Higher confidence when graph data is available
        if self.graph_db and self.graph_db.is_healthy:
            confidence += 0.1
        else:
            confidence -= 0.1
        
        # Lower confidence for very short or very long content
        if content_metrics["word_count"] < 20 or content_metrics["word_count"] > 5000:
            confidence -= 0.1
        
        return max(0.1, min(1.0, confidence))
    
    def _infer_domain(self, content: str, note_path: str) -> Optional[str]:
        """Infer the knowledge domain of a note."""
        # Simple domain inference based on path and content
        path_parts = Path(note_path).parts
        
        # Check path for domain indicators
        domain_indicators = {
            "project": ["projects", "project"],
            "research": ["research", "studies", "papers"],
            "technical": ["code", "tech", "development", "programming"],
            "personal": ["journal", "personal", "diary"],
            "reference": ["reference", "resources", "docs"],
            "meeting": ["meetings", "notes"],
            "learning": ["learning", "courses", "education"]
        }
        
        for domain, keywords in domain_indicators.items():
            for keyword in keywords:
                if any(keyword in part.lower() for part in path_parts):
                    return domain
        
        # Check content for domain indicators
        content_lower = content.lower()
        if any(word in content_lower for word in ["code", "function", "class", "algorithm"]):
            return "technical"
        elif any(word in content_lower for word in ["research", "study", "hypothesis"]):
            return "research"
        elif any(word in content_lower for word in ["meeting", "agenda", "action items"]):
            return "meeting"
        
        return None
    
    def _create_minimal_quality_score(
        self,
        content: str,
        metadata: Dict[str, Any],
        note_path: str
    ) -> QualityScore:
        """Create a minimal quality score for very short or empty content."""
        word_count = len(content.split()) if content else 0
        
        return QualityScore(
            overall_score=0.1,
            level=QualityLevel.SEEDLING,
            completeness=0.1,
            structure=0.1,
            connections=0.0,
            freshness=self._calculate_freshness_score(metadata),
            word_count=word_count,
            link_count=0,
            backlink_count=0,
            last_modified=metadata.get("last_modified", 0.0),
            headers_count=0,
            list_items_count=0,
            connection_metrics=self._create_minimal_connection_metrics(),
            suggestions=["Add substantial content to improve note quality"],
            confidence=0.9,
            domain=None,
            tags=[]
        )
    
    def _create_minimal_connection_metrics(self) -> ConnectionMetrics:
        """Create minimal connection metrics when graph is unavailable."""
        return ConnectionMetrics(
            outbound_links=0,
            inbound_links=0,
            bidirectional_links=0,
            broken_links=0,
            connection_density=0.0,
            hub_score=0.0,
            authority_score=0.0
        )
    
    def _identify_specific_issues(self, content: str, quality_score: QualityScore) -> List[str]:
        """Identify specific quality issues in a note."""
        issues = []
        
        if quality_score.word_count < 50:
            issues.append("Content too short")
        if quality_score.headers_count == 0:
            issues.append("Missing headers")
        if quality_score.connection_metrics.outbound_links == 0:
            issues.append("No outbound links")
        if quality_score.connection_metrics.broken_links > 0:
            issues.append("Broken links present")
        if quality_score.freshness < 0.3:
            issues.append("Content is stale")
        if not quality_score.tags:
            issues.append("No tags for categorization")
        
        return issues
    
    def _estimate_potential_quality(self, quality_score: QualityScore) -> float:
        """Estimate the potential quality if issues were addressed."""
        potential = quality_score.overall_score
        
        # Add potential improvements
        if quality_score.word_count < 200:
            potential += 0.2
        if quality_score.headers_count < 2:
            potential += 0.1
        if quality_score.connection_metrics.outbound_links < 3:
            potential += 0.15
        if not quality_score.tags:
            potential += 0.05
        
        return min(1.0, potential)
    
    def _determine_primary_gap_type(self, issues: List[str]) -> str:
        """Determine the primary type of quality gap."""
        if "Content too short" in issues:
            return "completeness"
        elif "Missing headers" in issues or "No tags for categorization" in issues:
            return "structure"
        elif "No outbound links" in issues or "Broken links present" in issues:
            return "connections"
        elif "Content is stale" in issues:
            return "freshness"
        else:
            return "general"
    
    def _calculate_gap_priority(self, quality_score: QualityScore, issues: List[str]) -> str:
        """Calculate priority level for addressing a quality gap."""
        if quality_score.overall_score < 0.2:
            return "high"
        elif quality_score.overall_score < 0.4:
            return "medium"
        else:
            return "low"
    
    def _estimate_improvement_effort(self, issues: List[str]) -> str:
        """Estimate effort required to address quality issues."""
        if len(issues) >= 4:
            return "2h"
        elif len(issues) >= 2:
            return "30min"
        else:
            return "10min"
    
    def _find_related_notes(self, content: str, file_path: Path) -> List[str]:
        """Find notes related to the current note."""
        # Extract linked notes from content
        links = self.link_pattern.findall(content)
        related_notes = []
        
        for link_match in links:
            # Handle both [[link]] and [text](link) formats
            link = link_match[0] if link_match[0] else link_match[1]
            if link and not link.startswith('http'):
                related_notes.append(link)
        
        return related_notes[:5]  # Return up to 5 related notes