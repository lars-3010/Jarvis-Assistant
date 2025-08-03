"""
Knowledge domain analyzer for semantic clustering and relationship mapping.

This module analyzes knowledge domains within the vault, identifying semantic clusters,
cross-domain connections, and opportunities for bridging related but unlinked domains.
"""

import time
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any
from collections import defaultdict, Counter
from dataclasses import asdict
import logging
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA

from jarvis.services.analytics.models import (
    KnowledgeDomain, SemanticCluster, DomainConnection, BridgeOpportunity,
    AnalyticsError
)
from jarvis.services.analytics.errors import (
    InsufficientDataError, AnalysisTimeoutError, ServiceUnavailableError,
    ModelError
)
from jarvis.core.interfaces import IVectorSearcher, IGraphDatabase, IVaultReader


logger = logging.getLogger(__name__)


class KnowledgeDomainAnalyzer:
    """
    Analyzes knowledge domains and their relationships within the vault.
    
    Uses semantic similarity for clustering and graph analysis for 
    relationship mapping between different knowledge domains.
    """
    
    def __init__(
        self,
        vector_searcher: Optional[IVectorSearcher] = None,
        graph_db: Optional[IGraphDatabase] = None,
        vault_reader: Optional[IVaultReader] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """Initialize the domain analyzer."""
        self.vector_searcher = vector_searcher
        self.graph_db = graph_db
        self.vault_reader = vault_reader
        self.config = config or {}
        
        # Configuration
        self.max_processing_time = self.config.get("max_processing_time_seconds", 15)
        self.clustering_threshold = self.config.get("clustering_threshold", 0.7)
        self.min_cluster_size = self.config.get("min_cluster_size", 3)
        self.max_domains = self.config.get("max_domains", 20)
        
        # Domain analysis state
        self._note_embeddings: Dict[str, np.ndarray] = {}
        self._note_metadata: Dict[str, Dict[str, Any]] = {}
        self._similarity_cache: Dict[Tuple[str, str], float] = {}
        
        logger.debug("KnowledgeDomainAnalyzer initialized")
    
    async def cluster_by_semantic_similarity(
        self,
        embeddings: Dict[str, np.ndarray]
    ) -> List[SemanticCluster]:
        """
        Group notes by semantic similarity using clustering algorithms.
        
        Args:
            embeddings: Dictionary mapping note paths to embedding vectors
            
        Returns:
            List of SemanticCluster objects
        """
        start_time = time.time()
        
        try:
            if len(embeddings) < self.min_cluster_size:
                raise InsufficientDataError("domain_analyzer", self.min_cluster_size, len(embeddings))
            
            # Convert embeddings to matrix
            note_paths = list(embeddings.keys())
            embedding_matrix = np.array([embeddings[path] for path in note_paths])
            
            # Normalize embeddings for cosine similarity
            norms = np.linalg.norm(embedding_matrix, axis=1, keepdims=True)
            normalized_embeddings = embedding_matrix / np.maximum(norms, 1e-8)
            
            # Perform clustering
            clusters = await self._perform_clustering(normalized_embeddings, note_paths)
            
            # Convert clusters to SemanticCluster objects
            semantic_clusters = []
            for i, (cluster_indices, coherence) in enumerate(clusters):
                if len(cluster_indices) < self.min_cluster_size:
                    continue
                
                cluster_notes = [note_paths[idx] for idx in cluster_indices]
                centroid_idx = self._find_centroid_note(cluster_indices, normalized_embeddings)
                centroid_note = note_paths[centroid_idx]
                
                # Extract keywords from cluster notes
                keywords = await self._extract_cluster_keywords(cluster_notes)
                
                # Generate cluster description
                description = self._generate_cluster_description(cluster_notes, keywords)
                
                semantic_cluster = SemanticCluster(
                    id=f"cluster_{i}",
                    centroid_note=centroid_note,
                    notes=cluster_notes,
                    coherence_score=coherence,
                    keywords=keywords,
                    description=description
                )
                semantic_clusters.append(semantic_cluster)
                
                # Check processing time
                if time.time() - start_time > self.max_processing_time:
                    logger.warning("Semantic clustering timeout")
                    break
            
            # Sort by coherence score (descending)
            semantic_clusters.sort(key=lambda c: c.coherence_score, reverse=True)
            
            return semantic_clusters[:self.max_domains]
            
        except Exception as e:
            if isinstance(e, (InsufficientDataError, AnalysisTimeoutError)):
                raise
            logger.error(f"Semantic clustering error: {e}")
            raise ModelError("semantic_clustering", "clustering", str(e))
    
    async def analyze_cross_domain_connections(
        self,
        graph_data: Dict[str, Any]
    ) -> List[DomainConnection]:
        """
        Analyze connections between different knowledge domains.
        
        Args:
            graph_data: Graph relationship data
            
        Returns:
            List of DomainConnection objects
        """
        try:
            if not self.graph_db or not self.graph_db.is_healthy:
                raise ServiceUnavailableError("graph_database", "domain connection analysis")
            
            domain_connections = []
            
            # First, identify domains from clusters
            clusters = await self._get_existing_clusters(graph_data)
            domain_map = self._map_notes_to_domains(clusters)
            
            # Analyze connections between domains
            connection_counts = defaultdict(lambda: defaultdict(int))
            bridge_notes = defaultdict(lambda: defaultdict(list))
            
            # Process graph relationships
            for note_path, connections in graph_data.get("connections", {}).items():
                source_domain = domain_map.get(note_path)
                if not source_domain:
                    continue
                
                for connected_note in connections.get("outbound", []):
                    target_domain = domain_map.get(connected_note)
                    if target_domain and target_domain != source_domain:
                        # Cross-domain connection found
                        connection_counts[source_domain][target_domain] += 1
                        bridge_notes[source_domain][target_domain].append(note_path)
            
            # Create DomainConnection objects
            for from_domain, targets in connection_counts.items():
                for to_domain, count in targets.items():
                    if count >= 2:  # Minimum connection threshold
                        # Calculate connection strength
                        strength = self._calculate_connection_strength(
                            from_domain, to_domain, count, domain_map
                        )
                        
                        # Determine connection type
                        connection_type = self._classify_connection_type(
                            from_domain, to_domain, bridge_notes[from_domain][to_domain]
                        )
                        
                        domain_connection = DomainConnection(
                            from_domain=from_domain,
                            to_domain=to_domain,
                            connection_strength=strength,
                            connection_count=count,
                            bridge_notes=bridge_notes[from_domain][to_domain][:5],
                            connection_type=connection_type
                        )
                        domain_connections.append(domain_connection)
            
            # Sort by connection strength
            domain_connections.sort(key=lambda c: c.connection_strength, reverse=True)
            
            return domain_connections
            
        except Exception as e:
            if isinstance(e, ServiceUnavailableError):
                raise
            logger.error(f"Cross-domain analysis error: {e}")
            raise AnalyticsError(f"Failed to analyze domain connections: {e}",
                               "domain_analyzer", "connection_analysis")
    
    async def identify_bridge_opportunities(
        self,
        domains: List[KnowledgeDomain]
    ) -> List[BridgeOpportunity]:
        """
        Identify opportunities to connect related but unlinked domains.
        
        Args:
            domains: List of identified knowledge domains
            
        Returns:
            List of BridgeOpportunity objects
        """
        try:
            if len(domains) < 2:
                return []
            
            bridge_opportunities = []
            
            # Compare all domain pairs
            for i, domain_a in enumerate(domains):
                for domain_b in domains[i+1:]:
                    # Calculate semantic similarity between domains
                    similarity = await self._calculate_domain_similarity(domain_a, domain_b)
                    
                    if similarity > 0.3:  # Threshold for potential bridges
                        # Check current connection level
                        current_connections = self._count_existing_connections(domain_a, domain_b)
                        
                        if current_connections < 3:  # Low connection domains
                            # Find potential connection pairs
                            potential_connections = await self._find_potential_connections(
                                domain_a, domain_b
                            )
                            
                            if potential_connections:
                                # Generate rationale and strategies
                                rationale = self._generate_bridge_rationale(
                                    domain_a, domain_b, similarity
                                )
                                strategies = self._generate_bridge_strategies(
                                    domain_a, domain_b, potential_connections
                                )
                                seed_notes = self._identify_seed_notes(
                                    domain_a, domain_b, potential_connections
                                )
                                
                                # Calculate priority
                                priority = self._calculate_bridge_priority(
                                    similarity, current_connections, len(potential_connections)
                                )
                                
                                bridge_opportunity = BridgeOpportunity(
                                    domain_a=domain_a.name,
                                    domain_b=domain_b.name,
                                    similarity_score=similarity,
                                    potential_connections=potential_connections[:10],
                                    rationale=rationale,
                                    priority=priority,
                                    bridge_strategies=strategies,
                                    seed_notes=seed_notes
                                )
                                bridge_opportunities.append(bridge_opportunity)
            
            # Sort by similarity score and priority
            bridge_opportunities.sort(
                key=lambda b: (b.priority == "high", b.similarity_score),
                reverse=True
            )
            
            return bridge_opportunities[:10]  # Return top 10 opportunities
            
        except Exception as e:
            logger.error(f"Bridge opportunity identification error: {e}")
            raise AnalyticsError(f"Failed to identify bridge opportunities: {e}",
                               "domain_analyzer", "bridge_identification")
    
    async def _perform_clustering(
        self,
        embeddings: np.ndarray,
        note_paths: List[str]
    ) -> List[Tuple[List[int], float]]:
        """Perform clustering on normalized embeddings."""
        clusters = []
        
        try:
            # Try DBSCAN first for automatic cluster number detection
            dbscan = DBSCAN(
                eps=1 - self.clustering_threshold,  # Convert similarity to distance
                min_samples=self.min_cluster_size,
                metric='cosine'
            )
            cluster_labels = dbscan.fit_predict(embeddings)
            
            # Process DBSCAN results
            unique_labels = set(cluster_labels)
            for label in unique_labels:
                if label == -1:  # Skip noise points
                    continue
                
                cluster_indices = [i for i, l in enumerate(cluster_labels) if l == label]
                if len(cluster_indices) >= self.min_cluster_size:
                    coherence = self._calculate_cluster_coherence(
                        cluster_indices, embeddings
                    )
                    clusters.append((cluster_indices, coherence))
            
            # If DBSCAN found too few clusters, fall back to K-means
            if len(clusters) < 2:
                # Estimate number of clusters
                n_clusters = min(self.max_domains, max(2, len(embeddings) // 10))
                
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                cluster_labels = kmeans.fit_predict(embeddings)
                
                # Process K-means results
                for label in range(n_clusters):
                    cluster_indices = [i for i, l in enumerate(cluster_labels) if l == label]
                    if len(cluster_indices) >= self.min_cluster_size:
                        coherence = self._calculate_cluster_coherence(
                            cluster_indices, embeddings
                        )
                        clusters.append((cluster_indices, coherence))
            
        except Exception as e:
            logger.error(f"Clustering algorithm error: {e}")
            # Fallback: simple similarity-based clustering
            clusters = self._fallback_similarity_clustering(embeddings)
        
        return clusters
    
    def _calculate_cluster_coherence(
        self,
        cluster_indices: List[int],
        embeddings: np.ndarray
    ) -> float:
        """Calculate coherence score for a cluster."""
        if len(cluster_indices) < 2:
            return 0.0
        
        cluster_embeddings = embeddings[cluster_indices]
        
        # Calculate pairwise similarities within cluster
        similarities = cosine_similarity(cluster_embeddings)
        
        # Get upper triangle (excluding diagonal)
        n = len(cluster_indices)
        total_similarities = 0
        count = 0
        
        for i in range(n):
            for j in range(i+1, n):
                total_similarities += similarities[i][j]
                count += 1
        
        return total_similarities / count if count > 0 else 0.0
    
    def _find_centroid_note(
        self,
        cluster_indices: List[int],
        embeddings: np.ndarray
    ) -> int:
        """Find the note closest to the cluster centroid."""
        cluster_embeddings = embeddings[cluster_indices]
        centroid = np.mean(cluster_embeddings, axis=0)
        
        # Find note closest to centroid
        similarities = cosine_similarity([centroid], cluster_embeddings)[0]
        best_idx = np.argmax(similarities)
        
        return cluster_indices[best_idx]
    
    async def _extract_cluster_keywords(self, cluster_notes: List[str]) -> List[str]:
        """Extract keywords that characterize a cluster."""
        keywords = []
        
        try:
            # Extract keywords from file names
            file_words = []
            for note_path in cluster_notes:
                path = Path(note_path)
                # Split filename on common separators
                words = path.stem.replace('-', ' ').replace('_', ' ').split()
                file_words.extend([w.lower() for w in words if len(w) > 2])
            
            # Get most common words
            word_counts = Counter(file_words)
            keywords.extend([word for word, count in word_counts.most_common(10)])
            
            # Extract keywords from content if vault reader available
            if self.vault_reader:
                content_words = []
                for note_path in cluster_notes[:5]:  # Sample first 5 notes
                    try:
                        content, _ = self.vault_reader.read_file(note_path)
                        words = content.lower().split()
                        content_words.extend([w for w in words if len(w) > 3])
                    except Exception:
                        continue
                
                if content_words:
                    content_counts = Counter(content_words)
                    keywords.extend([word for word, count in content_counts.most_common(5)])
            
        except Exception as e:
            logger.warning(f"Keyword extraction error: {e}")
            # Fallback: use path components
            for note_path in cluster_notes[:3]:
                keywords.extend(Path(note_path).parts[-2:])  # Use parent dir and filename
        
        # Remove duplicates and return top keywords
        unique_keywords = list(dict.fromkeys(keywords))
        return unique_keywords[:8]
    
    def _generate_cluster_description(
        self,
        cluster_notes: List[str],
        keywords: List[str]
    ) -> str:
        """Generate a description for a semantic cluster."""
        if not keywords:
            return f"Cluster of {len(cluster_notes)} related notes"
        
        # Use top keywords to create description
        primary_keywords = keywords[:3]
        if len(primary_keywords) >= 2:
            return f"Notes about {', '.join(primary_keywords[:-1])} and {primary_keywords[-1]}"
        else:
            return f"Notes about {primary_keywords[0]}"
    
    async def _get_existing_clusters(self, graph_data: Dict[str, Any]) -> List[SemanticCluster]:
        """Get existing clusters from graph data or create new ones."""
        # This would typically load from cache or previous analysis
        # For now, create simple clusters based on graph structure
        clusters = []
        
        # Group notes by their connection patterns
        note_groups = defaultdict(list)
        for note_path, connections in graph_data.get("connections", {}).items():
            # Simple grouping by number of connections
            connection_count = len(connections.get("outbound", []))
            group_key = f"group_{connection_count // 5}"  # Group by connection density
            note_groups[group_key].append(note_path)
        
        for i, (group_key, notes) in enumerate(note_groups.items()):
            if len(notes) >= self.min_cluster_size:
                cluster = SemanticCluster(
                    id=f"graph_cluster_{i}",
                    centroid_note=notes[0],
                    notes=notes,
                    coherence_score=0.7,  # Default coherence
                    keywords=[group_key],
                    description=f"Notes with similar connectivity patterns"
                )
                clusters.append(cluster)
        
        return clusters
    
    def _map_notes_to_domains(self, clusters: List[SemanticCluster]) -> Dict[str, str]:
        """Map note paths to domain names."""
        domain_map = {}
        
        for cluster in clusters:
            domain_name = f"domain_{cluster.id}"
            for note_path in cluster.notes:
                domain_map[note_path] = domain_name
        
        return domain_map
    
    def _calculate_connection_strength(
        self,
        from_domain: str,
        to_domain: str,
        connection_count: int,
        domain_map: Dict[str, str]
    ) -> float:
        """Calculate strength of connection between domains."""
        # Count notes in each domain
        from_domain_size = sum(1 for domain in domain_map.values() if domain == from_domain)
        to_domain_size = sum(1 for domain in domain_map.values() if domain == to_domain)
        
        # Normalize by domain sizes
        max_possible_connections = from_domain_size * to_domain_size
        strength = connection_count / max(1, max_possible_connections * 0.1)
        
        return min(1.0, strength)
    
    def _classify_connection_type(
        self,
        from_domain: str,
        to_domain: str,
        bridge_notes: List[str]
    ) -> str:
        """Classify the type of connection between domains."""
        # Simple classification based on connection patterns
        if len(bridge_notes) > 5:
            return "hierarchical"
        elif len(bridge_notes) > 2:
            return "associative"
        else:
            return "sequential"
    
    async def _calculate_domain_similarity(
        self,
        domain_a: KnowledgeDomain,
        domain_b: KnowledgeDomain
    ) -> float:
        """Calculate semantic similarity between two domains."""
        try:
            # Compare keywords
            keywords_a = set(domain_a.keywords)
            keywords_b = set(domain_b.keywords)
            
            if not keywords_a or not keywords_b:
                return 0.0
            
            # Jaccard similarity for keywords
            intersection = len(keywords_a.intersection(keywords_b))
            union = len(keywords_a.union(keywords_b))
            keyword_similarity = intersection / union if union > 0 else 0.0
            
            # If vector searcher available, use embedding similarity
            if self.vector_searcher:
                try:
                    # Get representative notes from each domain
                    notes_a = domain_a.representative_notes[:3]
                    notes_b = domain_b.representative_notes[:3]
                    
                    if notes_a and notes_b:
                        # This would require implementing embedding similarity
                        # For now, use keyword similarity
                        pass
                except Exception:
                    pass
            
            return keyword_similarity
            
        except Exception as e:
            logger.warning(f"Domain similarity calculation error: {e}")
            return 0.0
    
    def _count_existing_connections(
        self,
        domain_a: KnowledgeDomain,
        domain_b: KnowledgeDomain
    ) -> int:
        """Count existing connections between two domains."""
        # This would check graph data for existing connections
        # For now, return a default value
        return domain_a.external_connections // 4  # Rough estimate
    
    async def _find_potential_connections(
        self,
        domain_a: KnowledgeDomain,
        domain_b: KnowledgeDomain
    ) -> List[Tuple[str, str]]:
        """Find potential note pairs that could be connected."""
        potential_connections = []
        
        # Simple approach: pair notes with similar keywords
        for note_a in domain_a.representative_notes[:5]:
            for note_b in domain_b.representative_notes[:5]:
                # Check if these notes have semantic similarity
                # For now, just create some potential connections
                if self._notes_could_be_related(note_a, note_b):
                    potential_connections.append((note_a, note_b))
        
        return potential_connections[:5]
    
    def _notes_could_be_related(self, note_a: str, note_b: str) -> bool:
        """Simple heuristic to check if two notes could be related."""
        # Extract words from note paths
        words_a = set(Path(note_a).stem.lower().replace('-', ' ').replace('_', ' ').split())
        words_b = set(Path(note_b).stem.lower().replace('-', ' ').replace('_', ' ').split())
        
        # Check for common words
        common_words = words_a.intersection(words_b)
        return len(common_words) > 0
    
    def _generate_bridge_rationale(
        self,
        domain_a: KnowledgeDomain,
        domain_b: KnowledgeDomain,
        similarity: float
    ) -> str:
        """Generate rationale for why domains should be connected."""
        return (
            f"Domains '{domain_a.name}' and '{domain_b.name}' show {similarity:.1%} "
            f"semantic similarity and could benefit from stronger connections to "
            f"enable knowledge transfer and reduce information silos."
        )
    
    def _generate_bridge_strategies(
        self,
        domain_a: KnowledgeDomain,
        domain_b: KnowledgeDomain,
        potential_connections: List[Tuple[str, str]]
    ) -> List[str]:
        """Generate strategies for bridging domains."""
        strategies = []
        
        if potential_connections:
            strategies.append("Create cross-references between related notes")
            strategies.append("Add tags that span both domains")
        
        if len(domain_a.keywords) > 0 and len(domain_b.keywords) > 0:
            strategies.append("Create index notes that reference both domains")
        
        strategies.append("Write connecting notes that explain relationships")
        
        return strategies
    
    def _identify_seed_notes(
        self,
        domain_a: KnowledgeDomain,
        domain_b: KnowledgeDomain,
        potential_connections: List[Tuple[str, str]]
    ) -> List[str]:
        """Identify good starting points for creating connections."""
        seed_notes = []
        
        # Use representative notes as seeds
        seed_notes.extend(domain_a.representative_notes[:2])
        seed_notes.extend(domain_b.representative_notes[:2])
        
        # Add notes from potential connections
        for note_a, note_b in potential_connections[:2]:
            seed_notes.extend([note_a, note_b])
        
        return list(set(seed_notes))[:5]  # Remove duplicates, limit to 5
    
    def _calculate_bridge_priority(
        self,
        similarity: float,
        current_connections: int,
        potential_connections_count: int
    ) -> str:
        """Calculate priority for bridge opportunity."""
        if similarity > 0.7 and current_connections < 2:
            return "high"
        elif similarity > 0.5 and current_connections < 3:
            return "medium"
        else:
            return "low"
    
    def _fallback_similarity_clustering(self, embeddings: np.ndarray) -> List[Tuple[List[int], float]]:
        """Fallback clustering using simple similarity thresholds."""
        clusters = []
        used_indices = set()
        
        for i in range(len(embeddings)):
            if i in used_indices:
                continue
            
            # Start new cluster
            cluster_indices = [i]
            used_indices.add(i)
            
            # Find similar notes
            similarities = cosine_similarity([embeddings[i]], embeddings)[0]
            for j, sim in enumerate(similarities):
                if j != i and j not in used_indices and sim > self.clustering_threshold:
                    cluster_indices.append(j)
                    used_indices.add(j)
            
            if len(cluster_indices) >= self.min_cluster_size:
                coherence = self._calculate_cluster_coherence(cluster_indices, embeddings)
                clusters.append((cluster_indices, coherence))
        
        return clusters