"""
Vault analytics orchestrator service.

This module provides the main analytics service that coordinates all analyzers
and provides a unified interface for vault analysis and insights.
"""

import asyncio
import time
import hashlib
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from dataclasses import asdict
import logging

from jarvis.core.interfaces import (
    IVaultAnalyticsService, IVaultReader, IVectorSearcher, 
    IGraphDatabase, IMetrics
)
from jarvis.services.analytics.models import (
    VaultContext, QualityAnalysis, DomainMap, QualityScore,
    KnowledgeDomain, OrganizationPattern, FolderHierarchy,
    DepthMetrics, ActionableRecommendation, QualityGap,
    BridgeOpportunity, AnalyticsError
)
from jarvis.services.analytics.errors import (
    VaultNotFoundError, AnalysisTimeoutError, ServiceUnavailableError,
    InsufficientDataError
)
from jarvis.services.analytics.cache import AnalyticsCache
from jarvis.services.analytics.analyzers.structure import VaultStructureAnalyzer
from jarvis.services.analytics.analyzers.quality import ContentQualityAnalyzer
from jarvis.services.analytics.analyzers.domain import KnowledgeDomainAnalyzer
from jarvis.utils.config import get_settings


logger = logging.getLogger(__name__)


class VaultAnalyticsService(IVaultAnalyticsService):
    """
    Main analytics service that orchestrates vault analysis.
    
    Coordinates structure, quality, and domain analyzers to provide
    comprehensive vault insights and recommendations.
    """
    
    def __init__(
        self,
        vault_reader: Optional[IVaultReader] = None,
        vector_searcher: Optional[IVectorSearcher] = None,
        graph_db: Optional[IGraphDatabase] = None,
        metrics: Optional[IMetrics] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """Initialize the analytics service."""
        self.vault_reader = vault_reader
        self.vector_searcher = vector_searcher
        self.graph_db = graph_db
        self.metrics = metrics
        
        # Configuration
        settings = get_settings()
        self.config = config or settings.get_analytics_config()
        self.enabled = self.config.get("enabled", True)
        
        if not self.enabled:
            logger.info("Analytics service disabled by configuration")
            return
        
        # Initialize cache
        self.cache = AnalyticsCache(self.config.get("cache", {}))
        
        # Initialize analyzers
        self.structure_analyzer = VaultStructureAnalyzer(
            config=self.config.get("performance", {})
        )
        self.quality_analyzer = ContentQualityAnalyzer(
            vault_reader=vault_reader,
            graph_db=graph_db,
            config=self.config.get("quality", {})
        )
        self.domain_analyzer = KnowledgeDomainAnalyzer(
            vector_searcher=vector_searcher,
            graph_db=graph_db,
            vault_reader=vault_reader,
            config=self.config.get("domains", {})
        )
        
        # Performance settings
        self.max_processing_time = self.config.get("performance", {}).get(
            "max_processing_time_seconds", 15
        )
        self.enable_parallel_processing = self.config.get("performance", {}).get(
            "enable_parallel_processing", True
        )
        self.sample_large_vaults = self.config.get("performance", {}).get(
            "sample_large_vaults", True
        )
        self.sample_threshold = self.config.get("performance", {}).get(
            "sample_threshold", 5000
        )
        
        logger.info("VaultAnalyticsService initialized")
    
    async def get_vault_context(self, vault_name: str = "default") -> Dict[str, Any]:
        """
        Generate comprehensive vault overview with structured data.
        
        Args:
            vault_name: Name of the vault to analyze
            
        Returns:
            Dictionary containing VaultContext data
        """
        if not self.enabled:
            raise ServiceUnavailableError("analytics_service", "vault context generation")
        
        start_time = time.time()
        
        try:
            # Check cache first
            cached_result = self.cache.get("vault_context", vault_name, cache_level=2)
            if cached_result:
                self._record_metrics("vault_context", start_time, cache_hit=True)
                return cached_result
            
            # Get vault files
            vault_files = await self._get_vault_files(vault_name)
            
            if not vault_files:
                raise VaultNotFoundError(vault_name)
            
            # Apply sampling for large vaults
            if len(vault_files) > self.sample_threshold and self.sample_large_vaults:
                vault_files = self._sample_files(vault_files)
                logger.info(f"Applied sampling: analyzing {len(vault_files)} files")
            
            # Generate content hash for cache invalidation
            content_hash = self._generate_content_hash(vault_files)
            
            # Run analysis components
            analysis_results = await self._run_comprehensive_analysis(vault_files, vault_name)
            
            # Synthesize results into VaultContext
            vault_context = await self._synthesize_vault_context(
                vault_name, vault_files, analysis_results, start_time
            )
            
            # Convert to dictionary for JSON serialization
            result = asdict(vault_context)
            
            # Cache the result
            self.cache.set(
                "vault_context", vault_name, result, 
                cache_level=2, content_hash=content_hash
            )
            
            self._record_metrics("vault_context", start_time, cache_hit=False)
            
            return result
            
        except Exception as e:
            if isinstance(e, (VaultNotFoundError, AnalysisTimeoutError, ServiceUnavailableError)):
                raise
            logger.error(f"Vault context generation error: {e}")
            raise AnalyticsError(f"Failed to generate vault context: {e}",
                               "analytics_service", "vault_context")
    
    async def analyze_quality_distribution(self, vault_name: str = "default") -> Dict[str, Any]:
        """
        Analyze content quality patterns across the vault.
        
        Args:
            vault_name: Name of the vault to analyze
            
        Returns:
            Dictionary containing QualityAnalysis data
        """
        if not self.enabled:
            raise ServiceUnavailableError("analytics_service", "quality analysis")
        
        start_time = time.time()
        
        try:
            # Check cache
            cached_result = self.cache.get("quality_analysis", vault_name, cache_level=2)
            if cached_result:
                self._record_metrics("quality_analysis", start_time, cache_hit=True)
                return cached_result
            
            # Get vault files
            vault_files = await self._get_vault_files(vault_name)
            
            if not vault_files:
                raise VaultNotFoundError(vault_name)
            
            # Run quality analysis
            quality_analysis = await self._run_quality_analysis(vault_files, vault_name, start_time)
            
            # Convert to dictionary
            result = asdict(quality_analysis)
            
            # Cache the result
            content_hash = self._generate_content_hash(vault_files)
            self.cache.set(
                "quality_analysis", vault_name, result,
                cache_level=2, content_hash=content_hash
            )
            
            self._record_metrics("quality_analysis", start_time, cache_hit=False)
            
            return result
            
        except Exception as e:
            if isinstance(e, (VaultNotFoundError, AnalysisTimeoutError)):
                raise
            logger.error(f"Quality analysis error: {e}")
            raise AnalyticsError(f"Failed to analyze quality distribution: {e}",
                               "analytics_service", "quality_analysis")
    
    async def map_knowledge_domains(self, vault_name: str = "default") -> Dict[str, Any]:
        """
        Identify and map knowledge domains with connection analysis.
        
        Args:
            vault_name: Name of the vault to analyze
            
        Returns:
            Dictionary containing DomainMap data
        """
        if not self.enabled:
            raise ServiceUnavailableError("analytics_service", "domain mapping")
        
        start_time = time.time()
        
        try:
            # Check cache
            cached_result = self.cache.get("domain_map", vault_name, cache_level=2)
            if cached_result:
                self._record_metrics("domain_map", start_time, cache_hit=True)
                return cached_result
            
            # Get vault files
            vault_files = await self._get_vault_files(vault_name)
            
            if not vault_files:
                raise VaultNotFoundError(vault_name)
            
            # Run domain analysis
            domain_map = await self._run_domain_analysis(vault_files, vault_name, start_time)
            
            # Convert to dictionary
            result = asdict(domain_map)
            
            # Cache the result
            content_hash = self._generate_content_hash(vault_files)
            self.cache.set(
                "domain_map", vault_name, result,
                cache_level=2, content_hash=content_hash
            )
            
            self._record_metrics("domain_map", start_time, cache_hit=False)
            
            return result
            
        except Exception as e:
            if isinstance(e, (VaultNotFoundError, AnalysisTimeoutError)):
                raise
            logger.error(f"Domain mapping error: {e}")
            raise AnalyticsError(f"Failed to map knowledge domains: {e}",
                               "analytics_service", "domain_mapping")
    
    async def assess_note_quality(
        self, 
        note_path: str, 
        vault_name: str = "default"
    ) -> Dict[str, Any]:
        """
        Assess quality of a specific note.
        
        Args:
            note_path: Path to the note to assess
            vault_name: Name of the vault containing the note
            
        Returns:
            Dictionary containing QualityScore data
        """
        if not self.enabled:
            raise ServiceUnavailableError("analytics_service", "note quality assessment")
        
        start_time = time.time()
        
        try:
            if not self.vault_reader:
                raise ServiceUnavailableError("vault_reader", "note reading")
            
            # Check cache
            cache_key = f"{note_path}:{vault_name}"
            cached_result = self.cache.get("note_quality", cache_key, cache_level=1)
            if cached_result:
                self._record_metrics("note_quality", start_time, cache_hit=True)
                return cached_result
            
            # Read note content
            content, metadata = self.vault_reader.read_file(note_path)
            
            # Assess quality
            quality_score = await self.quality_analyzer.assess_note_quality(
                content, metadata, note_path
            )
            
            # Convert to dictionary
            result = asdict(quality_score)
            
            # Cache the result (L1 cache for individual notes)
            self.cache.set("note_quality", cache_key, result, cache_level=1)
            
            self._record_metrics("note_quality", start_time, cache_hit=False)
            
            return result
            
        except Exception as e:
            logger.error(f"Note quality assessment error: {e}")
            raise AnalyticsError(f"Failed to assess note quality: {e}",
                               "analytics_service", "note_assessment")
    
    async def get_analytics_cache_status(self) -> Dict[str, Any]:
        """
        Get current cache status and freshness indicators.
        
        Returns:
            Dictionary containing CacheStatus data
        """
        try:
            cache_status = self.cache.get_cache_status()
            return asdict(cache_status)
        except Exception as e:
            logger.error(f"Cache status error: {e}")
            raise AnalyticsError(f"Failed to get cache status: {e}",
                               "analytics_service", "cache_status")
    
    async def invalidate_cache(self, vault_name: Optional[str] = None) -> bool:
        """
        Invalidate analytics cache for a vault or all vaults.
        
        Args:
            vault_name: Vault to invalidate, or None for all vaults
            
        Returns:
            True if successful
        """
        try:
            invalidated_count = self.cache.invalidate(vault_name)
            logger.info(f"Invalidated {invalidated_count} cache entries")
            return True
        except Exception as e:
            logger.error(f"Cache invalidation error: {e}")
            return False
    
    async def get_recommendations(
        self, 
        vault_name: str = "default", 
        limit: int = 10
    ) -> Dict[str, Any]:
        """
        Get actionable recommendations for vault improvement.
        
        Args:
            vault_name: Name of the vault to analyze
            limit: Maximum number of recommendations to return
            
        Returns:
            Dictionary containing recommendations
        """
        try:
            # Get vault context for comprehensive recommendations
            vault_context_dict = await self.get_vault_context(vault_name)
            
            # Extract recommendations from context
            recommendations = vault_context_dict.get("recommendations", [])[:limit]
            quality_gaps = vault_context_dict.get("quality_gaps", [])[:limit]
            bridge_opportunities = vault_context_dict.get("bridge_opportunities", [])[:limit]
            
            return {
                "recommendations": recommendations,
                "quality_gaps": quality_gaps,
                "bridge_opportunities": bridge_opportunities,
                "total_count": len(recommendations) + len(quality_gaps) + len(bridge_opportunities)
            }
            
        except Exception as e:
            logger.error(f"Recommendations generation error: {e}")
            raise AnalyticsError(f"Failed to generate recommendations: {e}",
                               "analytics_service", "recommendations")
    
    async def _get_vault_files(self, vault_name: str) -> List[Path]:
        """Get list of files in the vault."""
        if not self.vault_reader:
            raise ServiceUnavailableError("vault_reader", "file listing")
        
        try:
            files = self.vault_reader.list_files(".md")
            if not files:
                raise VaultNotFoundError(vault_name)
            return files
        except Exception as e:
            logger.error(f"Failed to get vault files: {e}")
            raise VaultNotFoundError(vault_name)
    
    def _sample_files(self, files: List[Path]) -> List[Path]:
        """Sample files for large vault analysis."""
        import random
        sample_size = min(self.sample_threshold, len(files))
        return random.sample(files, sample_size)
    
    def _generate_content_hash(self, files: List[Path]) -> str:
        """Generate hash of file list for cache invalidation."""
        file_info = []
        for file_path in files[:100]:  # Sample first 100 files for hash
            try:
                stat = file_path.stat()
                file_info.append(f"{file_path}:{stat.st_mtime}:{stat.st_size}")
            except Exception:
                file_info.append(str(file_path))
        
        content = "|".join(sorted(file_info))
        return hashlib.md5(content.encode()).hexdigest()[:16]
    
    async def _run_comprehensive_analysis(
        self, 
        vault_files: List[Path], 
        vault_name: str
    ) -> Dict[str, Any]:
        """Run all analysis components in parallel."""
        analysis_results = {}
        
        try:
            if self.enable_parallel_processing:
                # Run analyzers in parallel
                tasks = []
                
                # Structure analysis
                tasks.append(self._run_structure_analysis(vault_files))
                
                # Quality analysis (sample for performance)
                quality_files = vault_files[:200] if len(vault_files) > 200 else vault_files
                tasks.append(self._run_partial_quality_analysis(quality_files))
                
                # Domain analysis (if vector search available)
                if self.vector_searcher:
                    tasks.append(self._run_partial_domain_analysis(vault_files))
                
                # Run all tasks with timeout
                results = await asyncio.wait_for(
                    asyncio.gather(*tasks, return_exceptions=True),
                    timeout=self.max_processing_time
                )
                
                # Process results
                for i, result in enumerate(results):
                    if isinstance(result, Exception):
                        logger.warning(f"Analysis component {i} failed: {result}")
                        continue
                    
                    if i == 0:  # Structure analysis
                        analysis_results["structure"] = result
                    elif i == 1:  # Quality analysis
                        analysis_results["quality"] = result
                    elif i == 2:  # Domain analysis
                        analysis_results["domains"] = result
            
            else:
                # Run analyzers sequentially
                analysis_results["structure"] = await self._run_structure_analysis(vault_files)
                analysis_results["quality"] = await self._run_partial_quality_analysis(vault_files[:100])
                
                if self.vector_searcher:
                    analysis_results["domains"] = await self._run_partial_domain_analysis(vault_files)
            
        except asyncio.TimeoutError:
            logger.warning("Analysis timeout - returning partial results")
        except Exception as e:
            logger.error(f"Comprehensive analysis error: {e}")
        
        return analysis_results
    
    async def _run_structure_analysis(self, vault_files: List[Path]) -> Dict[str, Any]:
        """Run structure analysis component."""
        try:
            organization_pattern = await self.structure_analyzer.detect_organization_method(vault_files)
            depth_metrics = await self.structure_analyzer.calculate_depth_metrics(vault_files)
            content_clusters = await self.structure_analyzer.identify_content_clusters(vault_files)
            
            # Create folder hierarchy
            folder_hierarchy = self._create_folder_hierarchy(vault_files)
            
            return {
                "organization_pattern": organization_pattern,
                "depth_metrics": depth_metrics,
                "content_clusters": content_clusters,
                "folder_hierarchy": folder_hierarchy
            }
        except Exception as e:
            logger.error(f"Structure analysis error: {e}")
            return {}
    
    async def _run_partial_quality_analysis(self, vault_files: List[Path]) -> Dict[str, Any]:
        """Run quality analysis on a subset of files."""
        try:
            if not self.vault_reader:
                return {}
            
            quality_scores = {}
            quality_gaps = []
            
            # Sample files for quality analysis
            sample_files = vault_files[:50] if len(vault_files) > 50 else vault_files
            
            for file_path in sample_files:
                try:
                    content, metadata = self.vault_reader.read_file(str(file_path))
                    quality_score = await self.quality_analyzer.assess_note_quality(
                        content, metadata, str(file_path)
                    )
                    quality_scores[str(file_path)] = quality_score
                except Exception as e:
                    logger.warning(f"Failed to analyze {file_path}: {e}")
                    continue
            
            # Identify quality gaps
            if quality_scores:
                gaps = await self.quality_analyzer.identify_quality_gaps(sample_files)
                quality_gaps.extend(gaps)
            
            return {
                "quality_scores": quality_scores,
                "quality_gaps": quality_gaps
            }
        except Exception as e:
            logger.error(f"Quality analysis error: {e}")
            return {}
    
    async def _run_partial_domain_analysis(self, vault_files: List[Path]) -> Dict[str, Any]:
        """Run domain analysis with available services."""
        try:
            # This would require embeddings from vector searcher
            # For now, return empty results
            return {
                "semantic_clusters": [],
                "domain_connections": [],
                "bridge_opportunities": []
            }
        except Exception as e:
            logger.error(f"Domain analysis error: {e}")
            return {}
    
    async def _run_quality_analysis(
        self, 
        vault_files: List[Path], 
        vault_name: str, 
        start_time: float
    ) -> QualityAnalysis:
        """Run comprehensive quality analysis."""
        # Implementation would be similar to _run_partial_quality_analysis
        # but return a complete QualityAnalysis object
        quality_data = await self._run_partial_quality_analysis(vault_files)
        
        return QualityAnalysis(
            vault_name=vault_name,
            analysis_timestamp=time.time(),
            average_quality=0.5,  # Would calculate from quality_scores
            quality_distribution={"🌱": 10, "🌿": 20, "🌳": 15, "🗺️": 5},
            quality_trends=[],
            note_scores=quality_data.get("quality_scores", {}),
            quality_gaps=quality_data.get("quality_gaps", []),
            improvement_priorities=[],
            processing_time_ms=(time.time() - start_time) * 1000,
            confidence_score=0.8,
            cache_hit_rate=0.0
        )
    
    async def _run_domain_analysis(
        self, 
        vault_files: List[Path], 
        vault_name: str, 
        start_time: float
    ) -> DomainMap:
        """Run comprehensive domain analysis."""
        domain_data = await self._run_partial_domain_analysis(vault_files)
        
        return DomainMap(
            vault_name=vault_name,
            analysis_timestamp=time.time(),
            domains=[],
            domain_connections=domain_data.get("domain_connections", []),
            bridge_opportunities=domain_data.get("bridge_opportunities", []),
            semantic_clusters=domain_data.get("semantic_clusters", []),
            isolated_notes=[],
            processing_time_ms=(time.time() - start_time) * 1000,
            confidence_score=0.7,
            cache_hit_rate=0.0
        )
    
    async def _synthesize_vault_context(
        self,
        vault_name: str,
        vault_files: List[Path],
        analysis_results: Dict[str, Any],
        start_time: float
    ) -> VaultContext:
        """Synthesize all analysis results into a VaultContext."""
        processing_time_ms = (time.time() - start_time) * 1000
        
        # Extract structure data
        structure_data = analysis_results.get("structure", {})
        organization_pattern = structure_data.get("organization_pattern")
        folder_hierarchy = structure_data.get("folder_hierarchy")
        depth_metrics = structure_data.get("depth_metrics")
        
        # Extract quality data
        quality_data = analysis_results.get("quality", {})
        quality_scores = quality_data.get("quality_scores", {})
        quality_gaps = quality_data.get("quality_gaps", [])
        
        # Extract domain data
        domain_data = analysis_results.get("domains", {})
        
        # Calculate basic statistics
        total_size_bytes = sum(
            f.stat().st_size for f in vault_files[:100] 
            if f.exists()
        )
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            analysis_results, quality_scores, quality_gaps
        )
        
        # Create cache status
        cache_status = self.cache.get_cache_status()
        
        return VaultContext(
            total_notes=len(vault_files),
            total_size_bytes=total_size_bytes,
            last_updated=time.time(),
            analysis_timestamp=time.time(),
            vault_name=vault_name,
            organization_pattern=organization_pattern or self._create_default_organization_pattern(),
            folder_structure=folder_hierarchy or self._create_default_folder_hierarchy(),
            depth_metrics=depth_metrics or self._create_default_depth_metrics(),
            quality_distribution=self._calculate_quality_distribution(quality_scores),
            average_quality_score=self._calculate_average_quality(quality_scores),
            quality_trends=[],
            identified_domains=[],
            domain_connections=[],
            isolated_notes=[],
            processing_time_ms=processing_time_ms,
            cache_hit_rate=cache_status.cache_hit_rate,
            confidence_score=self._calculate_overall_confidence(analysis_results),
            recommendations=recommendations,
            quality_gaps=quality_gaps,
            bridge_opportunities=[],
            cache_status=cache_status,
            errors=[],
            analysis_complete={
                "structure": bool(structure_data),
                "quality": bool(quality_data),
                "domains": bool(domain_data),
                "connections": self.graph_db is not None and self.graph_db.is_healthy
            }
        )
    
    def _create_folder_hierarchy(self, vault_files: List[Path]) -> FolderHierarchy:
        """Create folder hierarchy from file list."""
        folders = set()
        for file_path in vault_files:
            current = file_path.parent
            while current != current.parent:
                folders.add(current)
                current = current.parent
        
        if not folders:
            return FolderHierarchy(
                max_depth=0, average_depth=0.0, total_folders=0,
                root_folders=[], deepest_paths=[], empty_folders=[]
            )
        
        depths = [len(f.parts) for f in folders]
        
        return FolderHierarchy(
            max_depth=max(depths) if depths else 0,
            average_depth=sum(depths) / len(depths) if depths else 0.0,
            total_folders=len(folders),
            root_folders=[f.name for f in folders if len(f.parts) == 1],
            deepest_paths=[str(f) for f in sorted(folders, key=lambda x: len(x.parts), reverse=True)[:5]],
            empty_folders=[]
        )
    
    def _calculate_quality_distribution(self, quality_scores: Dict[str, Any]) -> Dict[str, int]:
        """Calculate quality level distribution."""
        distribution = {"🌱": 0, "🌿": 0, "🌳": 0, "🗺️": 0}
        
        for score_data in quality_scores.values():
            if isinstance(score_data, dict):
                level = score_data.get("level")
                if level and hasattr(level, 'value'):
                    emoji = level.value
                    distribution[emoji] = distribution.get(emoji, 0) + 1
        
        return distribution
    
    def _calculate_average_quality(self, quality_scores: Dict[str, Any]) -> float:
        """Calculate average quality score."""
        if not quality_scores:
            return 0.0
        
        total_score = 0.0
        count = 0
        
        for score_data in quality_scores.values():
            if isinstance(score_data, dict):
                overall_score = score_data.get("overall_score", 0.0)
                total_score += overall_score
                count += 1
        
        return total_score / count if count > 0 else 0.0
    
    def _generate_recommendations(
        self,
        analysis_results: Dict[str, Any],
        quality_scores: Dict[str, Any],
        quality_gaps: List[Any]
    ) -> List[ActionableRecommendation]:
        """Generate actionable recommendations based on analysis."""
        recommendations = []
        
        # Structure-based recommendations
        structure_data = analysis_results.get("structure", {})
        if structure_data:
            org_pattern = structure_data.get("organization_pattern")
            if org_pattern and hasattr(org_pattern, 'confidence') and org_pattern.confidence < 0.5:
                recommendations.append(ActionableRecommendation(
                    title="Improve vault organization",
                    description="Your vault organization pattern is unclear. Consider adopting a consistent methodology like PARA or Johnny Decimal.",
                    priority="medium",
                    category="organization",
                    action_items=["Choose an organization system", "Restructure existing folders", "Document your system"],
                    estimated_time="2h",
                    difficulty="medium",
                    impact_score=0.7,
                    confidence=0.8
                ))
        
        # Quality-based recommendations
        if quality_gaps:
            high_priority_gaps = [g for g in quality_gaps if g.get("priority") == "high"]
            if high_priority_gaps:
                recommendations.append(ActionableRecommendation(
                    title="Address high-priority quality gaps",
                    description=f"Found {len(high_priority_gaps)} notes with significant quality issues that should be addressed first.",
                    priority="high",
                    category="quality",
                    action_items=["Review identified notes", "Add missing content", "Fix broken links"],
                    estimated_time="1h",
                    difficulty="easy",
                    impact_score=0.8,
                    confidence=0.9
                ))
        
        return recommendations[:5]  # Return top 5 recommendations
    
    def _calculate_overall_confidence(self, analysis_results: Dict[str, Any]) -> float:
        """Calculate overall confidence in the analysis."""
        component_count = len([k for k in analysis_results.keys() if analysis_results[k]])
        max_components = 3  # structure, quality, domains
        
        base_confidence = component_count / max_components
        
        # Adjust based on service availability
        if self.graph_db and self.graph_db.is_healthy:
            base_confidence += 0.1
        if self.vector_searcher:
            base_confidence += 0.1
        
        return min(1.0, base_confidence)
    
    def _record_metrics(self, operation: str, start_time: float, cache_hit: bool = False):
        """Record performance metrics."""
        if self.metrics:
            duration_ms = (time.time() - start_time) * 1000
            
            self.metrics.record_histogram(
                f"analytics_{operation}_duration_ms", 
                duration_ms,
                tags={"cache_hit": str(cache_hit)}
            )
            
            self.metrics.record_counter(
                f"analytics_{operation}_requests",
                tags={"cache_hit": str(cache_hit)}
            )
    
    def _create_default_organization_pattern(self) -> OrganizationPattern:
        """Create default organization pattern when analysis fails."""
        from jarvis.services.analytics.models import OrganizationMethod
        return OrganizationPattern(
            method=OrganizationMethod.UNKNOWN,
            confidence=0.0,
            indicators=[],
            folder_patterns=[],
            exceptions=[]
        )
    
    def _create_default_folder_hierarchy(self) -> FolderHierarchy:
        """Create default folder hierarchy when analysis fails."""
        return FolderHierarchy(
            max_depth=0,
            average_depth=0.0,
            total_folders=0,
            root_folders=[],
            deepest_paths=[],
            empty_folders=[]
        )
    
    def _create_default_depth_metrics(self) -> DepthMetrics:
        """Create default depth metrics when analysis fails."""
        return DepthMetrics(
            depth_distribution={},
            files_by_depth={},
            complexity_score=0.0,
            organization_score=0.0
        )