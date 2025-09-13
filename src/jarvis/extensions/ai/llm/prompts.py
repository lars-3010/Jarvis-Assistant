"""
Prompt template system with Jinja2 support.

This module provides a flexible prompt template system for LLM operations
with support for dynamic template rendering and task-specific customization.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, Template, TemplateError
from pydantic import BaseModel, Field

from jarvis.utils.logging import setup_logging

from .interfaces import TaskType
from .models import LLMError, PromptTemplate

logger = setup_logging(__name__)


class PromptVariables(BaseModel):
    """Variables that can be used in prompts."""
    # Content variables
    text: str | None = None
    content: str | None = None
    question: str | None = None
    context: str | None = None

    # List variables
    items: list[str] = Field(default_factory=list)
    categories: list[str] = Field(default_factory=list)
    key_points: list[str] = Field(default_factory=list)

    # Metadata
    task_type: TaskType | None = None
    style: str | None = None
    max_length: int | None = None
    analysis_type: str | None = None

    # User preferences
    tone: str = "professional"
    language: str = "en"
    format: str = "markdown"

    # System variables
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    model_name: str | None = None

    class Config:
        use_enum_values = True


class PromptTemplateManager:
    """Manages prompt templates with Jinja2 rendering."""

    def __init__(self, template_dir: Path | None = None):
        """Initialize prompt template manager.
        
        Args:
            template_dir: Directory containing template files
        """
        self.template_dir = template_dir or self._get_default_template_dir()
        self.env = Environment(
            loader=FileSystemLoader(str(self.template_dir)),
            trim_blocks=True,
            lstrip_blocks=True
        )

        # Built-in templates
        self.builtin_templates: dict[str, PromptTemplate] = {}
        self.loaded_templates: dict[str, PromptTemplate] = {}

        # Initialize built-in templates
        self._initialize_builtin_templates()

        # Load templates from directory
        self._load_templates_from_directory()

        logger.info(f"Prompt template manager initialized with {len(self.builtin_templates)} built-in templates")

    def _get_default_template_dir(self) -> Path:
        """Get default template directory."""
        current_dir = Path(__file__).parent
        template_dir = current_dir / "templates"
        template_dir.mkdir(exist_ok=True)
        return template_dir

    def _initialize_builtin_templates(self) -> None:
        """Initialize built-in prompt templates."""

        # Summarization templates
        self.builtin_templates["summarize_bullet"] = PromptTemplate(
            name="summarize_bullet",
            template="""Summarize the following text using bullet points{% if max_length %} in {{ max_length }} words or less{% endif %}.

Text to summarize:
{{ text }}

Summary:
• """,
            variables=["text", "max_length"],
            task_type=TaskType.SUMMARIZE,
            default_config={"temperature": 0.3, "max_tokens": 500}
        )

        self.builtin_templates["summarize_paragraph"] = PromptTemplate(
            name="summarize_paragraph",
            template="""Summarize the following text in paragraph form{% if max_length %} in {{ max_length }} words or less{% endif %}.

Text to summarize:
{{ text }}

Summary:""",
            variables=["text", "max_length"],
            task_type=TaskType.SUMMARIZE,
            default_config={"temperature": 0.3, "max_tokens": 500}
        )

        # Analysis templates
        self.builtin_templates["analyze_general"] = PromptTemplate(
            name="analyze_general",
            template="""Analyze the following content and answer the question.

Content:
{{ content }}

Question: {{ question }}

{% if analysis_type %}Analysis type: {{ analysis_type }}{% endif %}

Please provide:
1. A clear answer to the question
2. Key points from the content
3. Supporting evidence

Answer:""",
            variables=["content", "question", "analysis_type"],
            task_type=TaskType.ANALYZE,
            default_config={"temperature": 0.5, "max_tokens": 1000}
        )

        self.builtin_templates["analyze_sentiment"] = PromptTemplate(
            name="analyze_sentiment",
            template="""Analyze the sentiment of the following text.

Text:
{{ text }}

Provide:
1. Overall sentiment (positive, negative, neutral)
2. Confidence score (0-1)
3. Key emotional indicators
4. Brief explanation

Analysis:""",
            variables=["text"],
            task_type=TaskType.ANALYZE,
            default_config={"temperature": 0.3, "max_tokens": 400}
        )

        # Quick answer templates
        self.builtin_templates["quick_answer"] = PromptTemplate(
            name="quick_answer",
            template="""{% if context %}Context: {{ context }}

{% endif %}Answer concisely: {{ question }}

Answer:""",
            variables=["question", "context"],
            task_type=TaskType.QUICK_ANSWER,
            default_config={"temperature": 0.2, "max_tokens": 200}
        )

        # Information extraction templates
        self.builtin_templates["extract_structured"] = PromptTemplate(
            name="extract_structured",
            template="""Extract the following information from the text and return as JSON:

Schema:
{{ schema | tojson(indent=2) }}

Text:
{{ text }}

JSON:""",
            variables=["text", "schema"],
            task_type=TaskType.EXTRACT,
            default_config={"temperature": 0.1, "max_tokens": 800}
        )

        self.builtin_templates["extract_key_points"] = PromptTemplate(
            name="extract_key_points",
            template="""Extract the key points from the following text.

Text:
{{ text }}

Key Points:
1. """,
            variables=["text"],
            task_type=TaskType.EXTRACT,
            default_config={"temperature": 0.3, "max_tokens": 600}
        )

        # Classification templates
        self.builtin_templates["classify_categories"] = PromptTemplate(
            name="classify_categories",
            template="""Classify the following text into one or more of these categories: {{ categories | join(', ') }}

Text:
{{ text }}

Classification:""",
            variables=["text", "categories"],
            task_type=TaskType.CLASSIFY,
            default_config={"temperature": 0.1, "max_tokens": 100}
        )

        # Generation templates
        self.builtin_templates["generate_creative"] = PromptTemplate(
            name="generate_creative",
            template="""{% if context %}Context: {{ context }}

{% endif %}{{ prompt }}

{% if style %}Style: {{ style }}{% endif %}
{% if tone %}Tone: {{ tone }}{% endif %}

Generated content:""",
            variables=["prompt", "context", "style", "tone"],
            task_type=TaskType.GENERATE,
            default_config={"temperature": 0.8, "max_tokens": 2000}
        )

        # System templates
        self.builtin_templates["system_message"] = PromptTemplate(
            name="system_message",
            template="""You are a helpful AI assistant. Your task is to {{ task_description }}.

Guidelines:
- Be accurate and helpful
- Provide clear, concise responses
- Use {{ format }} formatting when appropriate
- Maintain a {{ tone }} tone

{% if additional_instructions %}Additional instructions:
{{ additional_instructions }}{% endif %}""",
            variables=["task_description", "format", "tone", "additional_instructions"],
            task_type=TaskType.GENERAL
        )

    def _load_templates_from_directory(self) -> None:
        """Load templates from the template directory."""
        if not self.template_dir.exists():
            return

        for template_file in self.template_dir.glob("*.json"):
            try:
                with open(template_file, encoding='utf-8') as f:
                    template_data = json.load(f)

                template = PromptTemplate(**template_data)
                self.loaded_templates[template.name] = template

                logger.debug(f"Loaded template: {template.name}")

            except Exception as e:
                logger.error(f"Failed to load template {template_file}: {e}")

    def get_template(self, name: str) -> PromptTemplate | None:
        """Get a template by name.
        
        Args:
            name: Template name
            
        Returns:
            PromptTemplate or None if not found
        """
        # Check loaded templates first
        if name in self.loaded_templates:
            return self.loaded_templates[name]

        # Check built-in templates
        if name in self.builtin_templates:
            return self.builtin_templates[name]

        return None

    def list_templates(self, task_type: TaskType | None = None) -> list[str]:
        """List available template names.
        
        Args:
            task_type: Filter by task type
            
        Returns:
            List of template names
        """
        all_templates = {**self.builtin_templates, **self.loaded_templates}

        if task_type:
            return [
                name for name, template in all_templates.items()
                if template.task_type == task_type
            ]

        return list(all_templates.keys())

    def render_template(
        self,
        template_name: str,
        variables: dict[str, Any] | PromptVariables,
        **kwargs
    ) -> str:
        """Render a template with variables.
        
        Args:
            template_name: Name of the template
            variables: Variables to substitute
            **kwargs: Additional variables
            
        Returns:
            Rendered template string
            
        Raises:
            LLMError: If template not found or rendering fails
        """
        template = self.get_template(template_name)
        if not template:
            raise LLMError(f"Template not found: {template_name}")

        # Convert PromptVariables to dict if needed
        if isinstance(variables, PromptVariables):
            var_dict = variables.model_dump(exclude_none=True)
        else:
            var_dict = variables.copy()

        # Add kwargs
        var_dict.update(kwargs)

        try:
            # Create Jinja2 template
            jinja_template = Template(template.template)

            # Render with variables
            rendered = jinja_template.render(**var_dict)

            return rendered.strip()

        except TemplateError as e:
            raise LLMError(f"Template rendering failed: {e}")

    def render_template_string(
        self,
        template_string: str,
        variables: dict[str, Any] | PromptVariables,
        **kwargs
    ) -> str:
        """Render a template string directly.
        
        Args:
            template_string: Template string to render
            variables: Variables to substitute
            **kwargs: Additional variables
            
        Returns:
            Rendered string
        """
        # Convert PromptVariables to dict if needed
        if isinstance(variables, PromptVariables):
            var_dict = variables.model_dump(exclude_none=True)
        else:
            var_dict = variables.copy()

        # Add kwargs
        var_dict.update(kwargs)

        try:
            template = Template(template_string)
            return template.render(**var_dict).strip()

        except TemplateError as e:
            raise LLMError(f"Template string rendering failed: {e}")

    def create_template(
        self,
        name: str,
        template_string: str,
        variables: list[str],
        task_type: TaskType = TaskType.GENERAL,
        save_to_file: bool = False
    ) -> PromptTemplate:
        """Create a new template.
        
        Args:
            name: Template name
            template_string: Template content
            variables: List of variable names
            task_type: Task type for the template
            save_to_file: Whether to save to file
            
        Returns:
            Created PromptTemplate
        """
        template = PromptTemplate(
            name=name,
            template=template_string,
            variables=variables,
            task_type=task_type
        )

        self.loaded_templates[name] = template

        if save_to_file:
            self._save_template_to_file(template)

        return template

    def _save_template_to_file(self, template: PromptTemplate) -> None:
        """Save a template to file."""
        try:
            template_file = self.template_dir / f"{template.name}.json"

            with open(template_file, 'w', encoding='utf-8') as f:
                json.dump(template.model_dump(), f, indent=2, ensure_ascii=False)

            logger.info(f"Saved template to: {template_file}")

        except Exception as e:
            logger.error(f"Failed to save template {template.name}: {e}")

    def get_template_for_task(
        self,
        task_type: TaskType,
        style: str | None = None
    ) -> PromptTemplate | None:
        """Get the best template for a task type.
        
        Args:
            task_type: Task type
            style: Optional style preference
            
        Returns:
            Best matching template or None
        """
        # Build preference order
        candidates = []

        # Task-specific templates with style
        if style:
            template_name = f"{task_type.value}_{style}"
            candidates.append(template_name)

        # Task-specific templates
        candidates.extend([
            f"{task_type.value}_general",
            f"{task_type.value}_default",
            task_type.value
        ])

        # Check candidates
        for candidate in candidates:
            template = self.get_template(candidate)
            if template:
                return template

        # Fallback to any template for task type
        all_templates = {**self.builtin_templates, **self.loaded_templates}
        for template in all_templates.values():
            if template.task_type == task_type:
                return template

        return None

    def validate_template(self, template: PromptTemplate) -> list[str]:
        """Validate a template.
        
        Args:
            template: Template to validate
            
        Returns:
            List of validation errors
        """
        errors = []

        try:
            # Test Jinja2 template compilation
            Template(template.template)
        except TemplateError as e:
            errors.append(f"Template syntax error: {e}")

        # Check for basic requirements
        if not template.name:
            errors.append("Template name is required")

        if not template.template:
            errors.append("Template content is required")

        return errors


# Global template manager instance
_template_manager: PromptTemplateManager | None = None


def get_template_manager() -> PromptTemplateManager:
    """Get the global template manager instance."""
    global _template_manager
    if _template_manager is None:
        _template_manager = PromptTemplateManager()
    return _template_manager


def render_prompt(
    template_name: str,
    variables: dict[str, Any] | PromptVariables,
    **kwargs
) -> str:
    """Convenience function to render a prompt template.
    
    Args:
        template_name: Name of the template
        variables: Variables to substitute
        **kwargs: Additional variables
        
    Returns:
        Rendered prompt string
    """
    manager = get_template_manager()
    return manager.render_template(template_name, variables, **kwargs)


def render_prompt_string(
    template_string: str,
    variables: dict[str, Any] | PromptVariables,
    **kwargs
) -> str:
    """Convenience function to render a prompt string.
    
    Args:
        template_string: Template string to render
        variables: Variables to substitute
        **kwargs: Additional variables
        
    Returns:
        Rendered string
    """
    manager = get_template_manager()
    return manager.render_template_string(template_string, variables, **kwargs)
