"""
Get Vault Context Plugin for MCP Tools.

This plugin provides comprehensive vault analysis and context generation,
delivering structured insights about vault organization, quality, and domains.
"""

import json
from typing import Any

from jarvis.core.interfaces import IVaultAnalyticsService
from jarvis.mcp.plugins.base import UtilityPlugin
from jarvis.mcp.schemas import UtilitySchemaConfig, create_utility_schema
from jarvis.utils.errors import PluginError
from jarvis.utils.logging import setup_logging
from mcp import types

logger = setup_logging(__name__)


class GetVaultContextPlugin(UtilityPlugin):
    """Plugin for comprehensive vault context analysis."""

    @property
    def name(self) -> str:
        """Get the plugin name."""
        return "get-vault-context"

    @property
    def description(self) -> str:
        """Get the plugin description."""
        return "Generate comprehensive vault context with organization patterns, quality metrics, and actionable insights"

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
        return ["analytics", "vault", "context", "insights", "organization", "quality"]

    def get_required_services(self) -> list[type]:
        """Get required service interfaces."""
        return [IVaultAnalyticsService]

    def get_tool_definition(self) -> types.Tool:
        """Get the MCP tool definition using standardized schema."""
        schema_config = UtilitySchemaConfig(
            supported_formats=["json", "markdown"],
            additional_properties={
                "vault": {
                    "type": "string",
                    "description": "Vault name",
                    "default": "default",
                },
                "include_recommendations": {
                    "type": "boolean",
                    "description": "Include actionable recommendations",
                    "default": True,
                },
                "include_quality_gaps": {
                    "type": "boolean",
                    "description": "Include quality gap analysis",
                    "default": True,
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
        """Execute vault context analysis."""
        try:
            if not self.container:
                raise PluginError("Service container not available")

            analytics_service = self.container.get(IVaultAnalyticsService)
            if not analytics_service:
                raise PluginError("Analytics service not available")

            # Extract arguments
            vault_name = arguments.get("vault", "default")
            output_format = arguments.get("format", "markdown")
            include_recommendations = arguments.get("include_recommendations", True)
            include_quality_gaps = arguments.get("include_quality_gaps", True)

            # Get vault context
            vault_context = await analytics_service.get_vault_context(vault_name)

            if output_format == "json":
                # Return structured JSON data
                return [types.TextContent(
                    type="text",
                    text=json.dumps(vault_context, indent=2, default=str)
                )]
            else:
                # Return formatted markdown
                response = self._format_vault_context_markdown(
                    vault_context,
                    include_recommendations,
                    include_quality_gaps
                )
                return [types.TextContent(type="text", text=response)]

        except Exception as e:
            logger.error(f"Vault context analysis error: {e}")
            return [types.TextContent(
                type="text",
                text=f"❌ Vault context analysis failed: {e!s}"
            )]

    def _format_vault_context_markdown(
        self,
        vault_context: dict[str, Any],
        include_recommendations: bool,
        include_quality_gaps: bool
    ) -> str:
        """Format vault context as human-readable markdown."""
        lines = ["# 🏛️ Vault Context Analysis\n"]

        # Basic Statistics
        lines.append("## 📊 Basic Statistics")
        lines.append(f"- **Total Notes:** {vault_context.get('total_notes', 0):,}")
        lines.append(f"- **Total Size:** {self._format_bytes(vault_context.get('total_size_bytes', 0))}")
        lines.append(f"- **Vault Name:** {vault_context.get('vault_name', 'Unknown')}")

        analysis_time = vault_context.get('analysis_timestamp', 0)
        if analysis_time:
            from datetime import datetime
            dt = datetime.fromtimestamp(analysis_time)
            lines.append(f"- **Analysis Time:** {dt.strftime('%Y-%m-%d %H:%M:%S')}")

        lines.append("")

        # Organization Pattern
        org_pattern = vault_context.get('organization_pattern')
        if org_pattern:
            lines.append("## 🗂️ Organization Pattern")
            method = org_pattern.get('method', 'unknown')
            confidence = org_pattern.get('confidence', 0.0)
            lines.append(f"- **Method:** {method.upper() if method else 'Unknown'}")
            lines.append(f"- **Confidence:** {confidence:.1%}")

            indicators = org_pattern.get('indicators', [])
            if indicators:
                lines.append("- **Evidence:**")
                for indicator in indicators[:5]:  # Top 5 indicators
                    lines.append(f"  - {indicator}")
            lines.append("")

        # Quality Distribution
        quality_dist = vault_context.get('quality_distribution', {})
        avg_quality = vault_context.get('average_quality_score', 0.0)
        if quality_dist:
            lines.append("## 📈 Quality Distribution")
            lines.append(f"- **Average Quality:** {avg_quality:.1%}")
            lines.append("- **Quality Levels:**")

            # Sort by quality level (seedling to comprehensive)
            quality_order = ["🌱", "🌿", "🌳", "🗺️"]
            for emoji in quality_order:
                count = quality_dist.get(emoji, 0)
                if count > 0:
                    level_name = self._get_quality_level_name(emoji)
                    percentage = count / max(1, sum(quality_dist.values())) * 100
                    lines.append(f"  - {emoji} **{level_name}:** {count} notes ({percentage:.1f}%)")
            lines.append("")

        # Folder Structure
        folder_structure = vault_context.get('folder_structure')
        if folder_structure:
            lines.append("## 📁 Folder Structure")
            lines.append(f"- **Total Folders:** {folder_structure.get('total_folders', 0)}")
            lines.append(f"- **Max Depth:** {folder_structure.get('max_depth', 0)}")
            lines.append(f"- **Average Depth:** {folder_structure.get('average_depth', 0.0):.1f}")

            root_folders = folder_structure.get('root_folders', [])
            if root_folders:
                lines.append("- **Root Folders:**")
                for folder in root_folders[:10]:  # Top 10 root folders
                    lines.append(f"  - {folder}")
            lines.append("")

        # Identified Domains
        domains = vault_context.get('identified_domains', [])
        if domains:
            lines.append("## 🎯 Knowledge Domains")
            lines.append(f"Found {len(domains)} knowledge domains:")
            for domain in domains[:5]:  # Top 5 domains
                name = domain.get('name', 'Unknown')
                note_count = domain.get('note_count', 0)
                avg_quality = domain.get('average_quality', 0.0)
                lines.append(f"- **{name}:** {note_count} notes (avg quality: {avg_quality:.1%})")
            lines.append("")

        # Performance Metrics
        processing_time = vault_context.get('processing_time_ms', 0)
        cache_hit_rate = vault_context.get('cache_hit_rate', 0.0)
        confidence_score = vault_context.get('confidence_score', 0.0)

        lines.append("## ⚡ Performance Metrics")
        lines.append(f"- **Processing Time:** {processing_time:.0f}ms")
        lines.append(f"- **Cache Hit Rate:** {cache_hit_rate:.1%}")
        lines.append(f"- **Analysis Confidence:** {confidence_score:.1%}")
        lines.append("")

        # Analysis Completeness
        analysis_complete = vault_context.get('analysis_complete', {})
        if analysis_complete:
            lines.append("## ✅ Analysis Coverage")
            for component, completed in analysis_complete.items():
                emoji = "✅" if completed else "❌"
                lines.append(f"- **{component.title()}:** {emoji}")
            lines.append("")

        # Recommendations
        if include_recommendations:
            recommendations = vault_context.get('recommendations', [])
            if recommendations:
                lines.append("## 💡 Actionable Recommendations")
                for i, rec in enumerate(recommendations[:5], 1):
                    title = rec.get('title', 'Untitled')
                    description = rec.get('description', 'No description')
                    priority = rec.get('priority', 'medium')
                    estimated_time = rec.get('estimated_time', 'Unknown')

                    priority_emoji = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(priority, "⚪")

                    lines.append(f"### {i}. {priority_emoji} {title}")
                    lines.append(f"{description}")
                    lines.append(f"**Estimated Time:** {estimated_time}")

                    action_items = rec.get('action_items', [])
                    if action_items:
                        lines.append("**Action Items:**")
                        for item in action_items:
                            lines.append(f"- [ ] {item}")
                    lines.append("")

        # Quality Gaps
        if include_quality_gaps:
            quality_gaps = vault_context.get('quality_gaps', [])
            if quality_gaps:
                lines.append("## 🔍 Quality Improvement Opportunities")
                for i, gap in enumerate(quality_gaps[:5], 1):
                    note_path = gap.get('note_path', 'Unknown')
                    gap_type = gap.get('gap_type', 'general')
                    priority = gap.get('priority', 'medium')
                    current_quality = gap.get('current_quality', 0.0)
                    potential_quality = gap.get('potential_quality', 0.0)

                    priority_emoji = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(priority, "⚪")
                    improvement = potential_quality - current_quality

                    lines.append(f"### {i}. {priority_emoji} {self._extract_filename(note_path)}")
                    lines.append(f"**Gap Type:** {gap_type.title()}")
                    lines.append(f"**Current Quality:** {current_quality:.1%} → **Potential:** {potential_quality:.1%} (+{improvement:.1%})")

                    issues = gap.get('issues', [])
                    if issues:
                        lines.append("**Issues:**")
                        for issue in issues:
                            lines.append(f"- {issue}")

                    suggestions = gap.get('suggestions', [])
                    if suggestions:
                        lines.append("**Suggestions:**")
                        for suggestion in suggestions[:3]:  # Top 3 suggestions
                            lines.append(f"- {suggestion}")
                    lines.append("")

        # Bridge Opportunities
        bridge_opportunities = vault_context.get('bridge_opportunities', [])
        if bridge_opportunities:
            lines.append("## 🌉 Bridge Opportunities")
            lines.append("Opportunities to connect related but unlinked knowledge domains:")

            for i, bridge in enumerate(bridge_opportunities[:3], 1):
                domain_a = bridge.get('domain_a', 'Unknown')
                domain_b = bridge.get('domain_b', 'Unknown')
                similarity = bridge.get('similarity_score', 0.0)
                priority = bridge.get('priority', 'medium')

                priority_emoji = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(priority, "⚪")

                lines.append(f"### {i}. {priority_emoji} {domain_a} ↔ {domain_b}")
                lines.append(f"**Similarity:** {similarity:.1%}")
                lines.append(f"**Rationale:** {bridge.get('rationale', 'No rationale provided')}")

                strategies = bridge.get('bridge_strategies', [])
                if strategies:
                    lines.append("**Strategies:**")
                    for strategy in strategies:
                        lines.append(f"- {strategy}")
                lines.append("")

        # Cache Status
        cache_status = vault_context.get('cache_status')
        if cache_status:
            lines.append("## 💾 Cache Status")
            lines.append(f"- **Hit Rate:** {cache_status.get('cache_hit_rate', 0.0):.1%}")
            lines.append(f"- **Total Entries:** {cache_status.get('total_entries', 0):,}")
            lines.append(f"- **Memory Usage:** {cache_status.get('memory_usage_mb', 0.0):.1f} MB")
            lines.append(f"- **Efficiency Score:** {cache_status.get('cache_efficiency_score', 0.0):.1%}")
            lines.append("")

        # Footer
        lines.append("---")
        lines.append("*Generated by Jarvis Assistant Analytics Engine*")

        return "\n".join(lines)

    def _format_bytes(self, bytes_value: int) -> str:
        """Format bytes in human-readable format."""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if bytes_value < 1024.0:
                return f"{bytes_value:.1f} {unit}"
            bytes_value /= 1024.0
        return f"{bytes_value:.1f} TB"

    def _get_quality_level_name(self, emoji: str) -> str:
        """Get quality level name from emoji."""
        mapping = {
            "🌱": "Seedling",
            "🌿": "Growing",
            "🌳": "Mature",
            "🗺️": "Comprehensive"
        }
        return mapping.get(emoji, "Unknown")

    def _extract_filename(self, path: str) -> str:
        """Extract filename from path."""
        from pathlib import Path
        return Path(path).name
