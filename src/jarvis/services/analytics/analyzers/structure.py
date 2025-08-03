"""
Vault structure analyzer for organization pattern detection.

This module analyzes vault organizational patterns, folder hierarchy,
and content structure to provide insights into vault organization quality.
"""

import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any
from collections import defaultdict, Counter
from dataclasses import dataclass
import logging

from jarvis.services.analytics.models import (
    OrganizationPattern, OrganizationMethod, FolderHierarchy, 
    DepthMetrics, AnalyticsError
)
from jarvis.services.analytics.errors import InsufficientDataError, AnalysisTimeoutError


logger = logging.getLogger(__name__)


@dataclass
class ContentCluster:
    """A cluster of content based on path patterns."""
    name: str
    pattern: str
    paths: List[str]
    file_count: int
    coherence_score: float  # 0.0-1.0
    depth_range: Tuple[int, int]
    keywords: List[str]


class VaultStructureAnalyzer:
    """
    Analyzes vault organizational patterns and structure.
    
    Detects common organization methodologies and provides
    insights into vault structure quality and consistency.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the structure analyzer."""
        self.config = config or {}
        self.max_processing_time = self.config.get("max_processing_time_seconds", 15)
        
        # Organization pattern indicators
        self._para_indicators = {
            "folders": ["projects", "areas", "resources", "archive", "inbox"],
            "patterns": [
                r"^\d+\.\s*projects?",
                r"^\d+\.\s*areas?",
                r"^\d+\.\s*resources?",
                r"^\d+\.\s*archive",
                r"inbox|capture"
            ]
        }
        
        self._johnny_decimal_patterns = [
            r"^\d{2}[-\s]",  # 10- or 10 
            r"^\d{2}\.\d{2}",  # 10.01
            r"^\d{1,2}[-\s]\d{1,2}[-\s]",  # 1-2- or 1 2 
        ]
        
        self._zettelkasten_patterns = [
            r"^\d{12,14}",  # 202301011200 (timestamp)
            r"^\d{4}\.\d{2}\.\d{2}",  # 2023.01.01
            r"^[A-Z]{1,3}\d{3,6}",  # A001, BIO001
        ]
        
        logger.debug("VaultStructureAnalyzer initialized")
    
    async def detect_organization_method(self, files: List[Path]) -> OrganizationPattern:
        """
        Detect the primary organization method used in the vault.
        
        Args:
            files: List of file paths in the vault
            
        Returns:
            OrganizationPattern with detected method and confidence
        """
        start_time = time.time()
        
        try:
            if len(files) < 5:
                raise InsufficientDataError("structure_analyzer", 5, len(files))
            
            # Extract folder structure
            folders = self._extract_folders(files)
            folder_names = [f.name.lower() for f in folders]
            
            # Score each organization method
            scores = {
                OrganizationMethod.PARA: self._score_para_method(folder_names, files),
                OrganizationMethod.JOHNNY_DECIMAL: self._score_johnny_decimal(folder_names, files),
                OrganizationMethod.ZETTELKASTEN: self._score_zettelkasten(files),
                OrganizationMethod.TOPIC_BASED: self._score_topic_based(folder_names),
                OrganizationMethod.CHRONOLOGICAL: self._score_chronological(files),
            }
            
            # Check processing time
            if time.time() - start_time > self.max_processing_time:
                raise AnalysisTimeoutError("structure_analyzer", self.max_processing_time)
            
            # Find the highest scoring method
            best_method = max(scores.keys(), key=lambda k: scores[k]["score"])
            best_score_data = scores[best_method]
            
            # Determine if it's mixed or unknown
            high_scores = [method for method, data in scores.items() 
                          if data["score"] > 0.3]
            
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
                exceptions=self._find_exceptions(files, method)
            )
            
        except Exception as e:
            if isinstance(e, (InsufficientDataError, AnalysisTimeoutError)):
                raise
            logger.error(f"Organization detection error: {e}")
            raise AnalyticsError(f"Failed to detect organization method: {e}", 
                               "structure_analyzer", "organization_detection")
    
    async def calculate_depth_metrics(self, files: List[Path]) -> DepthMetrics:
        """
        Calculate folder depth distribution and complexity metrics.
        
        Args:
            files: List of file paths in the vault
            
        Returns:
            DepthMetrics with depth analysis
        """
        try:
            # Calculate depth for each file
            depth_counts = Counter()
            files_by_depth = defaultdict(list)
            
            for file_path in files:
                # Count path components (depth)
                depth = len(file_path.parts) - 1  # Subtract 1 for the filename
                depth_counts[depth] += 1
                files_by_depth[depth].append(str(file_path))
            
            if not depth_counts:
                raise InsufficientDataError("structure_analyzer", 1, 0)
            
            # Calculate complexity score
            max_depth = max(depth_counts.keys())
            avg_depth = sum(d * c for d, c in depth_counts.items()) / sum(depth_counts.values())
            
            # Complexity is higher when files are spread across many depths
            depth_variety = len(depth_counts)
            depth_distribution_evenness = self._calculate_evenness(list(depth_counts.values()))
            
            # Normalize complexity (0.0 = very simple, 1.0 = very complex)
            complexity_score = min(1.0, (
                (max_depth / 10.0) * 0.4 +  # Deep nesting contributes to complexity
                (depth_variety / 8.0) * 0.3 +  # Many different depths
                depth_distribution_evenness * 0.3  # Even distribution across depths
            ))
            
            # Organization score (inverse of complexity, with adjustments)
            organization_score = self._calculate_organization_score(
                depth_counts, avg_depth, max_depth
            )
            
            return DepthMetrics(
                depth_distribution=dict(depth_counts),
                files_by_depth=dict(files_by_depth),
                complexity_score=complexity_score,
                organization_score=organization_score
            )
            
        except Exception as e:
            if isinstance(e, InsufficientDataError):
                raise
            logger.error(f"Depth metrics calculation error: {e}")
            raise AnalyticsError(f"Failed to calculate depth metrics: {e}",
                               "structure_analyzer", "depth_calculation")
    
    async def identify_content_clusters(self, files: List[Path]) -> List[ContentCluster]:
        """
        Identify natural content groupings based on path patterns.
        
        Args:
            files: List of file paths in the vault
            
        Returns:
            List of ContentCluster objects
        """
        try:
            clusters = []
            
            # Group files by their parent directories
            dir_groups = defaultdict(list)
            for file_path in files:
                parent_dir = file_path.parent
                dir_groups[parent_dir].append(file_path)
            
            # Analyze each directory group
            for parent_dir, dir_files in dir_groups.items():
                if len(dir_files) < 2:  # Skip single-file directories
                    continue
                
                cluster = self._analyze_directory_cluster(parent_dir, dir_files)
                if cluster:
                    clusters.append(cluster)
            
            # Sort clusters by coherence score (descending)
            clusters.sort(key=lambda c: c.coherence_score, reverse=True)
            
            return clusters[:20]  # Return top 20 clusters
            
        except Exception as e:
            logger.error(f"Content clustering error: {e}")
            raise AnalyticsError(f"Failed to identify content clusters: {e}",
                               "structure_analyzer", "content_clustering")
    
    def _extract_folders(self, files: List[Path]) -> Set[Path]:
        """Extract unique folder paths from file list."""
        folders = set()
        for file_path in files:
            current = file_path.parent
            while current != current.parent:  # Stop at root
                folders.add(current)
                current = current.parent
        return folders
    
    def _score_para_method(self, folder_names: List[str], files: List[Path]) -> Dict[str, Any]:
        """Score how well the vault matches PARA methodology."""
        score = 0.0
        indicators = []
        patterns = []
        
        # Check for PARA folder names
        para_matches = 0
        for folder in self._para_indicators["folders"]:
            if any(folder in name for name in folder_names):
                para_matches += 1
                indicators.append(f"Found '{folder}' folder")
        
        # Check for numbered PARA structure
        pattern_matches = 0
        for pattern in self._para_indicators["patterns"]:
            if any(re.search(pattern, name, re.IGNORECASE) for name in folder_names):
                pattern_matches += 1
                patterns.append(pattern)
                indicators.append(f"Found PARA pattern: {pattern}")
        
        # Calculate score
        if para_matches >= 3:  # At least 3 of the main PARA folders
            score += 0.6
        elif para_matches >= 2:
            score += 0.4
        elif para_matches >= 1:
            score += 0.2
        
        if pattern_matches > 0:
            score += min(0.4, pattern_matches * 0.2)
        
        return {
            "score": min(1.0, score),
            "indicators": indicators,
            "patterns": patterns
        }
    
    def _score_johnny_decimal(self, folder_names: List[str], files: List[Path]) -> Dict[str, Any]:
        """Score how well the vault matches Johnny Decimal system."""
        score = 0.0
        indicators = []
        patterns = []
        
        # Check for Johnny Decimal patterns in folder names
        matching_patterns = 0
        for pattern in self._johnny_decimal_patterns:
            matches = [name for name in folder_names 
                      if re.search(pattern, name, re.IGNORECASE)]
            if matches:
                matching_patterns += 1
                patterns.append(pattern)
                indicators.extend([f"Found Johnny Decimal pattern in '{m}'" for m in matches[:3]])
        
        # Check file names too
        file_names = [f.name for f in files]
        file_matches = 0
        for pattern in self._johnny_decimal_patterns:
            file_pattern_matches = [name for name in file_names 
                                  if re.search(pattern, name, re.IGNORECASE)]
            if file_pattern_matches:
                file_matches += 1
        
        # Calculate score
        if matching_patterns >= 2:
            score += 0.7
        elif matching_patterns >= 1:  
            score += 0.4
        
        if file_matches > 0:
            score += min(0.3, file_matches * 0.1)
        
        return {
            "score": min(1.0, score),
            "indicators": indicators,
            "patterns": patterns
        }
    
    def _score_zettelkasten(self, files: List[Path]) -> Dict[str, Any]:
        """Score how well the vault matches Zettelkasten method."""
        score = 0.0
        indicators = []
        patterns = []
        
        file_names = [f.name for f in files]
        
        # Check for Zettelkasten patterns in file names
        matching_files = 0
        for pattern in self._zettelkasten_patterns:
            matches = [name for name in file_names 
                      if re.search(pattern, name, re.IGNORECASE)]
            if matches:
                matching_files += len(matches)
                patterns.append(pattern)
                indicators.append(f"Found {len(matches)} files with Zettelkasten pattern")
        
        # Score based on percentage of files following pattern
        if len(files) > 0:
            pattern_percentage = matching_files / len(files)
            if pattern_percentage > 0.5:
                score = 0.8
            elif pattern_percentage > 0.3:
                score = 0.6
            elif pattern_percentage > 0.1:
                score = 0.3
            elif matching_files > 0:
                score = 0.1
        
        return {
            "score": score,
            "indicators": indicators,
            "patterns": patterns
        }
    
    def _score_topic_based(self, folder_names: List[str]) -> Dict[str, Any]:
        """Score how well the vault is organized by topics."""
        score = 0.0
        indicators = []
        patterns = []
        
        # Look for topic-like folder names (not dates or numbers)
        topic_folders = []
        for name in folder_names:
            # Skip folders that look like dates, numbers, or system folders
            if (not re.search(r'^\d+', name) and 
                not re.search(r'\d{4}[-/]\d{1,2}', name) and
                not name.startswith('.') and
                len(name) > 2):
                topic_folders.append(name)
        
        if len(folder_names) > 0:
            topic_ratio = len(topic_folders) / len(folder_names)
            if topic_ratio > 0.7:
                score = 0.6
                indicators.append(f"High ratio of topic-based folders ({topic_ratio:.1%})")
            elif topic_ratio > 0.5:
                score = 0.4
                indicators.append(f"Moderate topic-based organization ({topic_ratio:.1%})")
            elif topic_ratio > 0.3:
                score = 0.2
                indicators.append(f"Some topic-based folders ({topic_ratio:.1%})")
        
        return {
            "score": score,
            "indicators": indicators,
            "patterns": patterns
        }
    
    def _score_chronological(self, files: List[Path]) -> Dict[str, Any]:
        """Score how well the vault is organized chronologically."""
        score = 0.0
        indicators = []
        patterns = []
        
        # Look for date patterns in folder names and file names
        date_patterns = [
            r'\d{4}[-/]\d{1,2}[-/]\d{1,2}',  # YYYY-MM-DD or YYYY/MM/DD
            r'\d{4}[-/]\d{1,2}',  # YYYY-MM or YYYY/MM
            r'\d{4}',  # Just year
        ]
        
        all_paths = [str(f) for f in files]
        
        date_matches = 0
        for pattern in date_patterns:
            matches = [path for path in all_paths 
                      if re.search(pattern, path)]
            if matches:
                date_matches += len(matches)
                patterns.append(pattern)
        
        if len(files) > 0:
            date_percentage = date_matches / len(files)
            if date_percentage > 0.4:
                score = 0.7
                indicators.append(f"High chronological organization ({date_percentage:.1%})")
            elif date_percentage > 0.2:
                score = 0.4
                indicators.append(f"Moderate chronological patterns ({date_percentage:.1%})")
            elif date_percentage > 0.1:
                score = 0.2
                indicators.append(f"Some chronological patterns ({date_percentage:.1%})")
        
        return {
            "score": score,
            "indicators": indicators,
            "patterns": patterns
        }
    
    def _find_exceptions(self, files: List[Path], method: OrganizationMethod) -> List[str]:
        """Find files/folders that don't fit the detected organization method."""
        exceptions = []
        
        # This is a simplified implementation
        # In practice, this would analyze files that don't match the expected patterns
        
        if method == OrganizationMethod.PARA:
            # Look for files not in PARA structure
            for file_path in files[:10]:  # Sample first 10 files
                path_str = str(file_path).lower()
                if not any(para_folder in path_str 
                          for para_folder in self._para_indicators["folders"]):
                    exceptions.append(str(file_path))
        
        return exceptions[:5]  # Return max 5 exceptions
    
    def _calculate_evenness(self, values: List[int]) -> float:
        """Calculate how evenly distributed the values are (Shannon evenness)."""
        if not values or sum(values) == 0:
            return 0.0
        
        total = sum(values)
        proportions = [v / total for v in values if v > 0]
        
        # Shannon entropy
        entropy = -sum(p * (p.bit_length() - 1) for p in proportions if p > 0)
        max_entropy = (len(proportions).bit_length() - 1) if len(proportions) > 1 else 1
        
        return entropy / max_entropy if max_entropy > 0 else 0.0
    
    def _calculate_organization_score(self, depth_counts: Counter, 
                                    avg_depth: float, max_depth: int) -> float:
        """Calculate how well-organized the vault structure is."""
        # Good organization characteristics:
        # - Reasonable depth (not too shallow, not too deep)
        # - Consistent depth usage
        # - Not too many different depths
        
        # Optimal depth range is 2-4 levels
        depth_penalty = 0.0
        if avg_depth < 1.5:
            depth_penalty = 0.2  # Too shallow
        elif avg_depth > 5.0:
            depth_penalty = 0.3  # Too deep
        elif max_depth > 8:
            depth_penalty = 0.2  # Some very deep paths
        
        # Consistency bonus (most files at similar depths)
        evenness = self._calculate_evenness(list(depth_counts.values()))
        consistency_bonus = evenness * 0.3
        
        # Base organization score
        base_score = 0.7
        
        return max(0.0, min(1.0, base_score - depth_penalty + consistency_bonus))
    
    def _analyze_directory_cluster(self, parent_dir: Path, files: List[Path]) -> Optional[ContentCluster]:
        """Analyze a directory to create a content cluster."""
        if len(files) < 2:
            return None
        
        # Extract keywords from directory name and file names
        dir_name = parent_dir.name.lower()
        file_names = [f.stem.lower() for f in files]
        
        # Simple keyword extraction (split on common separators)
        all_words = []
        for name in [dir_name] + file_names:
            words = re.split(r'[-_\s\.]+', name)
            all_words.extend([w for w in words if len(w) > 2])
        
        # Get most common words as keywords
        word_counts = Counter(all_words)
        keywords = [word for word, count in word_counts.most_common(5)]
        
        # Calculate coherence based on shared keywords
        coherence_score = min(1.0, len(keywords) / max(3, len(files) * 0.5))
        
        # Calculate depth range
        depths = [len(f.parts) for f in files]
        depth_range = (min(depths), max(depths))
        
        return ContentCluster(
            name=parent_dir.name,
            pattern=f"Files in {parent_dir}",
            paths=[str(f) for f in files],
            file_count=len(files),
            coherence_score=coherence_score,
            depth_range=depth_range,
            keywords=keywords
        )