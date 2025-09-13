"""
Analyze Domains Plugin for MCP Tools.

This plugin provides knowledge domain analysis including semantic clustering,
cross-domain connections, and bridge opportunity identification.
"""

import json
from typing import Any

from jarvis.core.interfaces import IVaultAnalyticsService
from jarvis.mcp.plugins.base import UtilityPlugin
from jarvis.mcp.schemas import UtilitySchemaConfig, create_utility_schema
from jarvis.utils.errors import PluginError
import logging
from mcp import types

logger = logging.getLogger(__name__)


class AnalyzeDomainsPlugin(UtilityPlugin):
    """Plugin for knowledge domain analysis and relationship mapping."""

    @property
    def name(self) -> str:
        """Get the plugin name."""
        return "analyze-domains"

    @property
    def description(self) -> str:
        """Get the plugin description."""
        return "Analyze knowledge domains, identify semantic clusters, and discover bridge opportunities between related domains"

    @property
    def version(self) -> str:
        """Get the plugin version."""
        return "1.0.0"

    @property
    def author(self) -> str:
        """Get the plugin author."""
        return "Jarvis Assistant"

    @property
    def tags(self) -> list[str]:
        """Get plugin tags."""
        return ["analytics", "domains", "clustering", "knowledge", "relationships", "semantic"]

    def get_required_services(self) -> list[type]:
        """Get required service interfaces."""
        return [IVaultAnalyticsService]

    def get_tool_definition(self) -> types.Tool:
        """Get the MCP tool definition using standardized schema."""
        schema_config = UtilitySchemaConfig(
            supported_formats=["json"],
            additional_properties={
                "vault": {
                    "type": "string",
                    "description": "Vault name",
                    "default": "default",
                },
                "include_bridges": {
                    "type": "boolean",
                    "description": "Include bridge opportunities",
                    "default": True,
                },
                "include_clusters": {
                    "type": "boolean",
                    "description": "Include semantic clusters",
                    "default": True,
                },
                "show_connections": {
                    "type": "boolean",
                    "description": "Show cross-domain connections",
                    "default": True,
                },
                "min_domain_size": {
                    "type": "integer",
                    "description": "Minimum notes per domain",
                    "default": 3,
                    "minimum": 2,
                },
            },
        )

        input_schema = create_utility_schema(schema_config)

        return types.Tool(
            name=self.name,
            description=self.description,
            inputSchema=input_schema,
        )

    async def execute(self, arguments: dict[str, Any]) -> list[types.TextContent]:
        """Execute domain analysis."""
        try:
            if not self.container:
                raise PluginError("Service container not available")

            analytics_service = self.container.get(IVaultAnalyticsService)
            if not analytics_service:
                raise PluginError("Analytics service not available")

            # Extract arguments
            vault_name = arguments.get("vault", "default")
            output_format = arguments.get("format", "json")
            include_bridges = arguments.get("include_bridges", True)
            include_clusters = arguments.get("include_clusters", True)
            show_connections = arguments.get("show_connections", True)
            min_domain_size = arguments.get("min_domain_size", 3)

            # Get domain analysis
            domain_map = await analytics_service.map_knowledge_domains(vault_name)

            if output_format == "json":
                # Return structured JSON data
                return [types.TextContent(
                    type="text",
                    text=json.dumps(domain_map, indent=2, default=str)
                )]
            else:
                # Return formatted markdown
                response = self._format_domain_analysis_markdown(
                    domain_map,
                    vault_name,
                    include_bridges,
                    include_clusters,
                    show_connections,
                    min_domain_size
                )
                return [types.TextContent(type="text", text=response)]

        except Exception as e:
            logger.error(f"Domain analysis error: {e}")
            return [types.TextContent(
                type="text",
                text=f"❌ Domain analysis failed: {e!s}"
            )]

    def _format_domain_analysis_markdown(
        self,
        domain_map: dict[str, Any],
        vault_name: str,
        include_bridges: bool,
        include_clusters: bool,
        show_connections: bool,
        min_domain_size: int
    ) -> str:
        """Format domain analysis as human-readable markdown."""
        lines = [f"# 🎯 Knowledge Domain Analysis: {vault_name}\n"]

        # Analysis Overview
        domains = domain_map.get('domains', [])
        semantic_clusters = domain_map.get('semantic_clusters', [])
        domain_connections = domain_map.get('domain_connections', [])
        bridge_opportunities = domain_map.get('bridge_opportunities', [])
        isolated_notes = domain_map.get('isolated_notes', [])

        lines.append("## 📊 Analysis Overview")
        lines.append(f"- **Identified Domains:** {len(domains)}")
        lines.append(f"- **Semantic Clusters:** {len(semantic_clusters)}")
        lines.append(f"- **Cross-Domain Connections:** {len(domain_connections)}")
        lines.append(f"- **Bridge Opportunities:** {len(bridge_opportunities)}")
        lines.append(f"- **Isolated Notes:** {len(isolated_notes)}")

        # Analysis performance
        processing_time = domain_map.get('processing_time_ms', 0)
        confidence_score = domain_map.get('confidence_score', 0.0)
        cache_hit_rate = domain_map.get('cache_hit_rate', 0.0)

        lines.append(f"- **Processing Time:** {processing_time:.0f}ms")
        lines.append(f"- **Confidence:** {confidence_score:.1%}")
        lines.append(f"- **Cache Hit Rate:** {cache_hit_rate:.1%}")
        lines.append("")

        # Knowledge Domains
        if domains:
            lines.append("## 🏛️ Knowledge Domains")

            # Sort domains by note count
            sorted_domains = sorted(domains, key=lambda d: d.get('note_count', 0), reverse=True)

            for i, domain in enumerate(sorted_domains, 1):
                name = domain.get('name', f'Domain {i}')
                description = domain.get('description', 'No description')
                note_count = domain.get('note_count', 0)
                avg_quality = domain.get('average_quality', 0.0)
                keywords = domain.get('keywords', [])

                # Skip domains below minimum size
                if note_count < min_domain_size:
                    continue

                lines.append(f"### {i}. 🎯 {name}")
                lines.append(f"**Description:** {description}")
                lines.append(f"**Notes:** {note_count} | **Avg Quality:** {avg_quality:.1%}")

                if keywords:
                    lines.append(f"**Keywords:** {', '.join(keywords[:8])}")

                # Quality distribution
                quality_dist = domain.get('quality_distribution', {})
                if quality_dist:
                    quality_summary = []
                    for emoji, count in quality_dist.items():
                        if count > 0:
                            quality_summary.append(f"{emoji}{count}")
                    if quality_summary:
                        lines.append(f"**Quality:** {' '.join(quality_summary)}")

                # Connection info
                internal_connections = domain.get('internal_connections', 0)
                external_connections = domain.get('external_connections', 0)
                isolation_score = domain.get('isolation_score', 0.0)

                lines.append(f"**Connections:** {internal_connections} internal, {external_connections} external")
                if isolation_score > 0.7:
                    lines.append("⚠️ **High isolation** - consider adding connections")

                # Representative notes
                representative_notes = domain.get('representative_notes', [])
                if representative_notes:
                    lines.append("**Key Notes:**")
                    for note in representative_notes[:3]:  # Top 3 notes
                        lines.append(f"  - {self._extract_filename(note)}")

                lines.append("")

        # Semantic Clusters
        if include_clusters and semantic_clusters:
            lines.append("## 🧩 Semantic Clusters")

            # Sort clusters by coherence score
            sorted_clusters = sorted(semantic_clusters, key=lambda c: c.get('coherence_score', 0.0), reverse=True)

            for i, cluster in enumerate(sorted_clusters[:10], 1):  # Top 10 clusters
                cluster_id = cluster.get('id', f'cluster_{i}')
                description = cluster.get('description', 'No description')
                note_count = len(cluster.get('notes', []))
                coherence = cluster.get('coherence_score', 0.0)
                keywords = cluster.get('keywords', [])
                centroid_note = cluster.get('centroid_note', '')

                lines.append(f"### {i}. 🧩 {cluster_id}")
                lines.append(f"**Description:** {description}")
                lines.append(f"**Notes:** {note_count} | **Coherence:** {coherence:.2f}")

                if keywords:
                    lines.append(f"**Keywords:** {', '.join(keywords[:5])}")

                if centroid_note:
                    lines.append(f"**Centroid:** {self._extract_filename(centroid_note)}")

                lines.append("")

        # Cross-Domain Connections
        if show_connections and domain_connections:
            lines.append("## 🌉 Cross-Domain Connections")

            # Sort by connection strength
            sorted_connections = sorted(
                domain_connections,
                key=lambda c: c.get('connection_strength', 0.0),
                reverse=True
            )

            for i, connection in enumerate(sorted_connections[:10], 1):  # Top 10 connections
                from_domain = connection.get('from_domain', 'Unknown')
                to_domain = connection.get('to_domain', 'Unknown')
                strength = connection.get('connection_strength', 0.0)
                count = connection.get('connection_count', 0)
                connection_type = connection.get('connection_type', 'unknown')
                bridge_notes = connection.get('bridge_notes', [])

                strength_indicator = self._get_strength_indicator(strength)

                lines.append(f"### {i}. {strength_indicator} {from_domain} → {to_domain}")
                lines.append(f"**Strength:** {strength:.2f} | **Connections:** {count} | **Type:** {connection_type}")

                if bridge_notes:
                    lines.append("**Bridge Notes:**")
                    for note in bridge_notes[:3]:  # Top 3 bridge notes
                        lines.append(f"  - {self._extract_filename(note)}")

                lines.append("")

        # Bridge Opportunities
        if include_bridges and bridge_opportunities:
            lines.append("## 🌉 Bridge Opportunities")
            lines.append("*Opportunities to connect related but unlinked domains*")
            lines.append("")

            # Sort by priority and similarity
            sorted_bridges = sorted(
                bridge_opportunities,
                key=lambda b: (b.get('priority') == 'high', b.get('similarity_score', 0.0)),
                reverse=True
            )

            for i, bridge in enumerate(sorted_bridges[:8], 1):  # Top 8 opportunities
                domain_a = bridge.get('domain_a', 'Unknown')
                domain_b = bridge.get('domain_b', 'Unknown')
                similarity = bridge.get('similarity_score', 0.0)
                priority = bridge.get('priority', 'medium')
                rationale = bridge.get('rationale', 'No rationale provided')

                priority_emoji = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(priority, "⚪")

                lines.append(f"### {i}. {priority_emoji} {domain_a} ↔ {domain_b}")
                lines.append(f"**Similarity:** {similarity:.1%} | **Priority:** {priority.title()}")
                lines.append(f"**Rationale:** {rationale}")

                # Bridge strategies
                strategies = bridge.get('bridge_strategies', [])
                if strategies:
                    lines.append("**Strategies:**")
                    for strategy in strategies[:3]:  # Top 3 strategies
                        lines.append(f"  - {strategy}")

                # Seed notes
                seed_notes = bridge.get('seed_notes', [])
                if seed_notes:
                    lines.append("**Suggested Starting Points:**")
                    for note in seed_notes[:3]:  # Top 3 seed notes
                        lines.append(f"  - {self._extract_filename(note)}")

                # Potential connections
                potential_connections = bridge.get('potential_connections', [])
                if potential_connections:
                    lines.append(f"**Potential Links:** {len(potential_connections)} note pairs identified")

                lines.append("")

        # Isolated Notes
        if isolated_notes:
            lines.append("## 🏝️ Isolated Notes")
            lines.append("*Notes with few connections that might benefit from integration*")
            lines.append("")

            isolated_count = len(isolated_notes)
            if isolated_count > 10:
                lines.append(f"Found {isolated_count} isolated notes. Showing first 10:")

            for note in isolated_notes[:10]:
                lines.append(f"- {self._extract_filename(note)}")

            if isolated_count > 10:
                lines.append(f"... and {isolated_count - 10} more")
            lines.append("")

        # Domain Analysis Tips
        lines.append("## 💡 Domain Development Tips")
        lines.append("1. **Strengthen Weak Domains:** Add more notes to domains with < 5 notes")
        lines.append("2. **Bridge Related Domains:** Create connections between similar domains")
        lines.append("3. **Integrate Isolated Notes:** Link isolated notes to relevant domains")
        lines.append("4. **Develop Key Concepts:** Expand notes in your most important domains")
        lines.append("5. **Cross-Reference:** Add links between related concepts across domains")
        lines.append("")

        # Analysis completion timestamp
        analysis_timestamp = domain_map.get('analysis_timestamp', 0)
        if analysis_timestamp > 0:
            from datetime import datetime
            dt = datetime.fromtimestamp(analysis_timestamp)
            lines.append(f"*Analysis completed at: {dt.strftime('%Y-%m-%d %H:%M:%S')}*")

        lines.append("---")
        lines.append("*Generated by Jarvis Assistant Domain Analyzer*")

        return "\n".join(lines)

    def _get_strength_indicator(self, strength: float) -> str:
        """Get visual indicator for connection strength."""
        if strength >= 0.8:
            return "🔗🔗🔗"  # Very strong
        elif strength >= 0.6:
            return "🔗🔗"    # Strong
        elif strength >= 0.4:
            return "🔗"      # Medium
        else:
            return "⚪"      # Weak

    def _extract_filename(self, path: str) -> str:
        """Extract filename from path."""
        from pathlib import Path
        return Path(path).name
