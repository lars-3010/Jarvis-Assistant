"""
Assess Quality Plugin for MCP Tools.

This plugin provides detailed quality analysis for individual notes or vault-wide
quality distribution assessment with actionable improvement recommendations.
"""

import json
from typing import Dict, Any, List, Type

from mcp import types
from jarvis.mcp.plugins.base import UtilityPlugin
from jarvis.core.interfaces import IVaultAnalyticsService
from jarvis.utils.logging import setup_logging
from jarvis.utils.errors import PluginError

logger = setup_logging(__name__)


class AssessQualityPlugin(UtilityPlugin):
    """Plugin for note and vault quality assessment."""
    
    @property
    def name(self) -> str:
        """Get the plugin name."""
        return "assess-quality"
    
    @property
    def description(self) -> str:
        """Get the plugin description."""
        return "Assess content quality for individual notes or analyze vault-wide quality patterns with improvement suggestions"
    
    @property
    def version(self) -> str:
        """Get the plugin version."""
        return "1.0.0"
    
    @property
    def author(self) -> str:
        """Get the plugin author."""
        return "Jarvis Assistant"
    
    @property
    def tags(self) -> List[str]:
        """Get plugin tags."""
        return ["analytics", "quality", "assessment", "improvement", "content"]
    
    def get_required_services(self) -> List[Type]:
        """Get required service interfaces."""
        return [IVaultAnalyticsService]
    
    def get_tool_definition(self) -> types.Tool:
        """Get the MCP tool definition."""
        return types.Tool(
            name=self.name,
            description=self.description,
            inputSchema={
                "type": "object",
                "properties": {
                    "scope": {
                        "type": "string",
                        "enum": ["note", "vault"],
                        "description": "Analysis scope: 'note' for individual note analysis or 'vault' for vault-wide analysis",
                        "default": "vault"
                    },
                    "note_path": {
                        "type": "string",
                        "description": "Path to specific note (required when scope='note')"
                    },
                    "vault": {
                        "type": "string",
                        "description": "Name of the vault to analyze (defaults to 'default')",
                        "default": "default"
                    },
                    "format": {
                        "type": "string",
                        "enum": ["markdown", "json"],
                        "description": "Output format: 'markdown' for human-readable or 'json' for structured data",
                        "default": "markdown"
                    },
                    "include_suggestions": {
                        "type": "boolean",
                        "description": "Include improvement suggestions in the output",
                        "default": True
                    },
                    "show_detailed_metrics": {
                        "type": "boolean",
                        "description": "Show detailed quality metrics breakdown",
                        "default": False
                    }
                },
                "additionalProperties": False
            }
        )
    
    async def execute(self, arguments: Dict[str, Any]) -> List[types.TextContent]:
        """Execute quality assessment."""
        try:
            if not self.container:
                raise PluginError("Service container not available")
            
            analytics_service = self.container.get(IVaultAnalyticsService)
            if not analytics_service:
                raise PluginError("Analytics service not available")
            
            # Extract arguments
            scope = arguments.get("scope", "vault")
            note_path = arguments.get("note_path")
            vault_name = arguments.get("vault", "default")
            output_format = arguments.get("format", "markdown")
            include_suggestions = arguments.get("include_suggestions", True)
            show_detailed_metrics = arguments.get("show_detailed_metrics", False)
            
            # Validate arguments
            if scope == "note" and not note_path:
                raise PluginError("note_path is required when scope='note'")
            
            # Perform analysis based on scope
            if scope == "note":
                result = await analytics_service.assess_note_quality(note_path, vault_name)
                
                if output_format == "json":
                    return [types.TextContent(
                        type="text",
                        text=json.dumps(result, indent=2, default=str)
                    )]
                else:
                    response = self._format_note_quality_markdown(
                        result, note_path, include_suggestions, show_detailed_metrics
                    )
                    return [types.TextContent(type="text", text=response)]
            
            else:  # scope == "vault"
                result = await analytics_service.analyze_quality_distribution(vault_name)
                
                if output_format == "json":
                    return [types.TextContent(
                        type="text",
                        text=json.dumps(result, indent=2, default=str)
                    )]
                else:
                    response = self._format_vault_quality_markdown(
                        result, vault_name, include_suggestions, show_detailed_metrics
                    )
                    return [types.TextContent(type="text", text=response)]
            
        except Exception as e:
            logger.error(f"Quality assessment error: {e}")
            return [types.TextContent(
                type="text",
                text=f"❌ Quality assessment failed: {str(e)}"
            )]
    
    def _format_note_quality_markdown(
        self,
        quality_score: Dict[str, Any],
        note_path: str,
        include_suggestions: bool,
        show_detailed_metrics: bool
    ) -> str:
        """Format individual note quality assessment as markdown."""
        lines = [f"# 📊 Quality Assessment: {self._extract_filename(note_path)}\n"]
        
        # Overall Quality
        overall_score = quality_score.get('overall_score', 0.0)
        level = quality_score.get('level', {})
        level_emoji = level.get('value', '🌱') if hasattr(level, 'get') else str(level)
        confidence = quality_score.get('confidence', 0.0)
        
        lines.append("## 🎯 Overall Quality")
        lines.append(f"- **Score:** {overall_score:.1%}")
        lines.append(f"- **Level:** {level_emoji} {self._get_quality_level_name(level_emoji)}")
        lines.append(f"- **Assessment Confidence:** {confidence:.1%}")
        lines.append("")
        
        # Component Scores
        lines.append("## 📈 Quality Components")
        components = [
            ("completeness", "Completeness", "📝"),
            ("structure", "Structure", "🏗️"),
            ("connections", "Connections", "🔗"),
            ("freshness", "Freshness", "🕒")
        ]
        
        for key, label, emoji in components:
            score = quality_score.get(key, 0.0)
            lines.append(f"- {emoji} **{label}:** {score:.1%}")
        lines.append("")
        
        # Content Metrics
        lines.append("## 📏 Content Metrics")
        word_count = quality_score.get('word_count', 0)
        link_count = quality_score.get('link_count', 0)
        backlink_count = quality_score.get('backlink_count', 0)
        headers_count = quality_score.get('headers_count', 0)
        list_items_count = quality_score.get('list_items_count', 0)
        
        lines.append(f"- **Word Count:** {word_count:,}")
        lines.append(f"- **Headers:** {headers_count}")
        lines.append(f"- **List Items:** {list_items_count}")
        lines.append(f"- **Outbound Links:** {link_count}")
        lines.append(f"- **Backlinks:** {backlink_count}")
        lines.append("")
        
        # Connection Analysis
        connection_metrics = quality_score.get('connection_metrics', {})
        if connection_metrics and show_detailed_metrics:
            lines.append("## 🕸️ Connection Analysis")
            lines.append(f"- **Connection Density:** {connection_metrics.get('connection_density', 0.0):.1%}")
            lines.append(f"- **Hub Score:** {connection_metrics.get('hub_score', 0.0):.1%}")
            lines.append(f"- **Authority Score:** {connection_metrics.get('authority_score', 0.0):.1%}")
            lines.append(f"- **Bidirectional Links:** {connection_metrics.get('bidirectional_links', 0)}")
            
            broken_links = connection_metrics.get('broken_links', 0)
            if broken_links > 0:
                lines.append(f"- **⚠️ Broken Links:** {broken_links}")
            lines.append("")
        
        # Domain and Tags
        domain = quality_score.get('domain')
        tags = quality_score.get('tags', [])
        if domain or tags:
            lines.append("## 🏷️ Classification")
            if domain:
                lines.append(f"- **Domain:** {domain.title()}")
            if tags:
                lines.append(f"- **Tags:** {', '.join(tags)}")
            lines.append("")
        
        # Last Modified
        last_modified = quality_score.get('last_modified', 0)
        if last_modified > 0:
            from datetime import datetime
            dt = datetime.fromtimestamp(last_modified)
            lines.append("## 🕒 Freshness")
            lines.append(f"- **Last Modified:** {dt.strftime('%Y-%m-%d %H:%M:%S')}")
            days_ago = (datetime.now().timestamp() - last_modified) / (24 * 3600)
            lines.append(f"- **Age:** {days_ago:.0f} days ago")
            lines.append("")
        
        # Improvement Suggestions
        if include_suggestions:
            suggestions = quality_score.get('suggestions', [])
            if suggestions:
                lines.append("## 💡 Improvement Suggestions")
                for i, suggestion in enumerate(suggestions, 1):
                    lines.append(f"{i}. {suggestion}")
                lines.append("")
        
        # Quality Level Guidance
        lines.append("## 📚 Quality Level Guide")
        lines.append("- 🌱 **Seedling (0-25%):** Basic content, needs development")
        lines.append("- 🌿 **Growing (25-50%):** Developing content with some structure")
        lines.append("- 🌳 **Mature (50-75%):** Well-developed, well-connected content")
        lines.append("- 🗺️ **Comprehensive (75-100%):** Authoritative, comprehensive content")
        lines.append("")
        
        lines.append("---")
        lines.append("*Assessment generated by Jarvis Assistant Quality Analyzer*")
        
        return "\n".join(lines)
    
    def _format_vault_quality_markdown(
        self,
        quality_analysis: Dict[str, Any],
        vault_name: str,
        include_suggestions: bool,
        show_detailed_metrics: bool
    ) -> str:
        """Format vault-wide quality analysis as markdown."""
        lines = [f"# 📊 Vault Quality Analysis: {vault_name}\n"]
        
        # Overall Statistics
        avg_quality = quality_analysis.get('average_quality', 0.0)
        total_analyzed = len(quality_analysis.get('note_scores', {}))
        confidence = quality_analysis.get('confidence_score', 0.0)
        
        lines.append("## 🎯 Overall Quality")
        lines.append(f"- **Average Quality:** {avg_quality:.1%}")
        lines.append(f"- **Notes Analyzed:** {total_analyzed:,}")
        lines.append(f"- **Analysis Confidence:** {confidence:.1%}")
        lines.append("")
        
        # Quality Distribution
        quality_distribution = quality_analysis.get('quality_distribution', {})
        if quality_distribution:
            lines.append("## 📈 Quality Distribution")
            total_notes = sum(quality_distribution.values())
            
            # Sort by quality level
            quality_order = ["🌱", "🌿", "🌳", "🗺️"]
            for emoji in quality_order:
                count = quality_distribution.get(emoji, 0)
                if total_notes > 0:
                    percentage = count / total_notes * 100
                    level_name = self._get_quality_level_name(emoji)
                    lines.append(f"- {emoji} **{level_name}:** {count:,} notes ({percentage:.1f}%)")
            lines.append("")
        
        # Quality Trends
        quality_trends = quality_analysis.get('quality_trends', [])
        if quality_trends and show_detailed_metrics:
            lines.append("## 📊 Quality Trends")
            for trend in quality_trends[:3]:  # Show last 3 trends
                timestamp = trend.get('timestamp', 0)
                avg_quality = trend.get('average_quality', 0.0)
                improvement_rate = trend.get('improvement_rate', 0.0)
                
                if timestamp > 0:
                    from datetime import datetime
                    dt = datetime.fromtimestamp(timestamp)
                    rate_indicator = "↗️" if improvement_rate > 0 else "↘️" if improvement_rate < 0 else "➡️"
                    lines.append(f"- **{dt.strftime('%Y-%m-%d')}:** {avg_quality:.1%} {rate_indicator}")
            lines.append("")
        
        # Top Quality Gaps
        quality_gaps = quality_analysis.get('quality_gaps', [])
        if quality_gaps and include_suggestions:
            lines.append("## 🔍 Priority Quality Improvements")
            
            high_priority_gaps = [g for g in quality_gaps if g.get('priority') == 'high'][:5]
            medium_priority_gaps = [g for g in quality_gaps if g.get('priority') == 'medium'][:3]
            
            if high_priority_gaps:
                lines.append("### 🔴 High Priority")
                for gap in high_priority_gaps:
                    note_path = gap.get('note_path', 'Unknown')
                    gap_type = gap.get('gap_type', 'general')
                    current_quality = gap.get('current_quality', 0.0)
                    potential_quality = gap.get('potential_quality', 0.0)
                    improvement = potential_quality - current_quality
                    
                    lines.append(f"- **{self._extract_filename(note_path)}** ({gap_type}): {current_quality:.1%} → {potential_quality:.1%} (+{improvement:.1%})")
                lines.append("")
            
            if medium_priority_gaps:
                lines.append("### 🟡 Medium Priority")
                for gap in medium_priority_gaps:
                    note_path = gap.get('note_path', 'Unknown')
                    gap_type = gap.get('gap_type', 'general')
                    current_quality = gap.get('current_quality', 0.0)
                    potential_quality = gap.get('potential_quality', 0.0)
                    improvement = potential_quality - current_quality
                    
                    lines.append(f"- **{self._extract_filename(note_path)}** ({gap_type}): {current_quality:.1%} → {potential_quality:.1%} (+{improvement:.1%})")
                lines.append("")
        
        # Improvement Priorities
        improvement_priorities = quality_analysis.get('improvement_priorities', [])
        if improvement_priorities and include_suggestions:
            lines.append("## 💡 Recommended Actions")
            for i, priority in enumerate(improvement_priorities[:5], 1):
                lines.append(f"{i}. {priority}")
            lines.append("")
        
        # Performance Metrics
        processing_time = quality_analysis.get('processing_time_ms', 0)
        cache_hit_rate = quality_analysis.get('cache_hit_rate', 0.0)
        
        if show_detailed_metrics:
            lines.append("## ⚡ Analysis Performance")
            lines.append(f"- **Processing Time:** {processing_time:.0f}ms")
            lines.append(f"- **Cache Hit Rate:** {cache_hit_rate:.1%}")
            lines.append("")
        
        # Quality Improvement Tips
        if include_suggestions:
            lines.append("## 📚 Quality Improvement Tips")
            lines.append("1. **Add Structure:** Use headers and lists to organize content")
            lines.append("2. **Expand Content:** Aim for 200+ words for substantial notes")
            lines.append("3. **Create Connections:** Link to related notes and concepts")
            lines.append("4. **Add Context:** Use tags and categories for organization")
            lines.append("5. **Regular Updates:** Keep content fresh and relevant")
            lines.append("")
        
        # Analysis timestamp
        analysis_timestamp = quality_analysis.get('analysis_timestamp', 0)
        if analysis_timestamp > 0:
            from datetime import datetime
            dt = datetime.fromtimestamp(analysis_timestamp)
            lines.append(f"*Analysis completed at: {dt.strftime('%Y-%m-%d %H:%M:%S')}*")
        
        lines.append("---")
        lines.append("*Generated by Jarvis Assistant Quality Analyzer*")
        
        return "\n".join(lines)
    
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