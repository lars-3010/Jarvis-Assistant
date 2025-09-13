"""
Configurable model routing system for LLM services.

This module provides intelligent model selection and routing based on
task types, performance metrics, and user preferences.
"""

from collections import defaultdict
from datetime import datetime, timedelta
from typing import Any

import logging

from .interfaces import IModelRouter, ModelInfo, TaskType
from .models import (
    LLMError,
    LLMProviderConfig,
    ModelPerformanceMetrics,
    ModelRoutingConfig,
    TaskRoutingRule,
)

logger = logging.getLogger(__name__)


class ModelRouter(IModelRouter):
    """Intelligent model routing system with performance optimization."""

    def __init__(self, config: ModelRoutingConfig, provider_config: LLMProviderConfig):
        """Initialize model router.
        
        Args:
            config: Routing configuration
            provider_config: Provider configuration with available models
        """
        self.config = config
        self.provider_config = provider_config

        # Performance tracking
        self.performance_metrics: dict[str, ModelPerformanceMetrics] = {}
        self.failed_models: dict[str, datetime] = {}

        # Model availability
        self.available_models: set[str] = set()
        self.model_info: dict[str, ModelInfo] = {}

        # Load models from provider config
        self._load_available_models()

        # Initialize default routing rules if none exist
        if not self.config.routing_rules:
            self._initialize_default_routing_rules()

        logger.info(f"Model router initialized with {len(self.available_models)} models")

    def _load_available_models(self) -> None:
        """Load available models from provider configuration."""
        for model_info in self.provider_config.available_models:
            if model_info.is_available:
                self.available_models.add(model_info.name)
                self.model_info[model_info.name] = model_info

    def _initialize_default_routing_rules(self) -> None:
        """Initialize default routing rules for common tasks."""
        default_rules = [
            TaskRoutingRule(
                task_type=TaskType.SUMMARIZE,
                model_name="mistral:7b-instruct-q4_K_M",
                conditions={"max_tokens": 500, "temperature": 0.3},
                priority=10
            ),
            TaskRoutingRule(
                task_type=TaskType.ANALYZE,
                model_name="llama3:8b-instruct-q8_0",
                conditions={"max_tokens": 1000, "temperature": 0.5},
                priority=10
            ),
            TaskRoutingRule(
                task_type=TaskType.QUICK_ANSWER,
                model_name="tinyllama:1.1b",
                conditions={"max_tokens": 200, "temperature": 0.2},
                priority=10
            ),
            TaskRoutingRule(
                task_type=TaskType.GENERATE,
                model_name="llama3:8b-instruct-q8_0",
                conditions={"max_tokens": 2000, "temperature": 0.8},
                priority=10
            ),
            TaskRoutingRule(
                task_type=TaskType.EXTRACT,
                model_name="mistral:7b-instruct-q4_K_M",
                conditions={"max_tokens": 800, "temperature": 0.1},
                priority=10
            ),
            TaskRoutingRule(
                task_type=TaskType.CLASSIFY,
                model_name="mistral:7b-instruct-q4_K_M",
                conditions={"max_tokens": 100, "temperature": 0.1},
                priority=10
            ),
            TaskRoutingRule(
                task_type=TaskType.GENERAL,
                model_name="mistral:7b-instruct-q4_K_M",
                conditions={"max_tokens": 1000, "temperature": 0.7},
                priority=5
            )
        ]

        # Only add rules for available models
        for rule in default_rules:
            if rule.model_name in self.available_models:
                self.config.routing_rules.append(rule)

    def select_model(
        self,
        task_type: TaskType,
        context_length: int = 0,
        quality_preference: str = "balanced"
    ) -> str:
        """Select the best model for a task.
        
        Args:
            task_type: Type of task
            context_length: Length of context in tokens
            quality_preference: "speed", "quality", or "balanced"
            
        Returns:
            Model name
        """
        # Get routing rule for task type
        routing_rule = self.config.get_routing_rule(task_type)

        # Get candidate models
        candidates = self._get_candidate_models(task_type, routing_rule)

        if not candidates:
            # Fallback to any available model
            if self.available_models:
                return list(self.available_models)[0]
            raise LLMError(f"No available models for task type: {task_type}")

        # Filter by availability and health
        healthy_candidates = self._filter_healthy_models(candidates)

        if not healthy_candidates:
            # Use candidates even if not perfectly healthy
            healthy_candidates = candidates

        # Select best model based on strategy
        return self._select_best_model(
            healthy_candidates,
            task_type,
            context_length,
            quality_preference
        )

    def _get_candidate_models(
        self,
        task_type: TaskType,
        routing_rule: TaskRoutingRule | None
    ) -> list[str]:
        """Get candidate models for a task type."""
        candidates = []

        # Primary candidate from routing rule
        if routing_rule and routing_rule.model_name in self.available_models:
            candidates.append(routing_rule.model_name)

        # Additional candidates from model specializations
        for model_name, model_info in self.model_info.items():
            if task_type in model_info.specialized_for:
                if model_name not in candidates:
                    candidates.append(model_name)

        # Fallback to all available models
        if not candidates:
            candidates = list(self.available_models)

        return candidates

    def _filter_healthy_models(self, candidates: list[str]) -> list[str]:
        """Filter out unhealthy models."""
        healthy = []
        current_time = datetime.now()

        for model_name in candidates:
            # Check if model recently failed
            if model_name in self.failed_models:
                failure_time = self.failed_models[model_name]
                if current_time - failure_time < timedelta(minutes=5):
                    continue  # Skip recently failed models

            # Check performance metrics
            metrics_key = f"{model_name}_{TaskType.GENERAL.value}"
            if metrics_key in self.performance_metrics:
                metrics = self.performance_metrics[metrics_key]

                # Check success rate
                if metrics.success_rate < self.config.min_success_rate:
                    continue

                # Check response time
                if metrics.avg_response_time_ms > self.config.max_response_time_ms:
                    continue

            healthy.append(model_name)

        return healthy

    def _select_best_model(
        self,
        candidates: list[str],
        task_type: TaskType,
        context_length: int,
        quality_preference: str
    ) -> str:
        """Select the best model from candidates."""
        if len(candidates) == 1:
            return candidates[0]

        # Score each candidate
        scores = {}
        for model_name in candidates:
            score = self._calculate_model_score(
                model_name,
                task_type,
                context_length,
                quality_preference
            )
            scores[model_name] = score

        # Return highest scoring model
        return max(scores, key=scores.get)

    def _calculate_model_score(
        self,
        model_name: str,
        task_type: TaskType,
        context_length: int,
        quality_preference: str
    ) -> float:
        """Calculate score for a model."""
        score = 0.0

        # Model info score
        model_info = self.model_info.get(model_name)
        if model_info:
            # Context window score
            if context_length <= model_info.context_window:
                score += 20.0
            else:
                score -= 10.0  # Penalty for exceeding context window

            # Specialization score
            if task_type in model_info.specialized_for:
                score += 15.0

            # Size-based score (larger models generally better quality)
            if "13b" in model_info.size:
                score += 10.0
            elif "70b" in model_info.size:
                score += 15.0
            elif "7b" in model_info.size:
                score += 5.0
            elif "1.1b" in model_info.size:
                score += 1.0

        # Performance metrics score
        metrics_key = f"{model_name}_{task_type.value}"
        if metrics_key in self.performance_metrics:
            metrics = self.performance_metrics[metrics_key]

            # Success rate score
            score += metrics.success_rate * 30.0

            # Response time score (inverted - faster is better)
            if metrics.avg_response_time_ms > 0:
                time_score = max(0, 10.0 - (metrics.avg_response_time_ms / 1000.0))
                score += time_score

            # Confidence score
            if metrics.avg_confidence > 0:
                score += metrics.avg_confidence * 10.0

        # Quality preference adjustments
        if quality_preference == "speed":
            # Favor smaller, faster models
            if model_info and "1.1b" in model_info.size:
                score += 10.0
            elif model_info and "7b" in model_info.size:
                score += 5.0
        elif quality_preference == "quality":
            # Favor larger, higher quality models
            if model_info and "70b" in model_info.size:
                score += 15.0
            elif model_info and "13b" in model_info.size:
                score += 10.0

        return score

    def get_fallback_model(self, failed_model: str, task_type: TaskType) -> str | None:
        """Get a fallback model if the primary fails.
        
        Args:
            failed_model: Name of the failed model
            task_type: Type of task
            
        Returns:
            Fallback model name or None
        """
        # Record failure
        self.failed_models[failed_model] = datetime.now()

        # Get candidates excluding failed model
        candidates = [
            model for model in self.available_models
            if model != failed_model
        ]

        if not candidates:
            return None

        # Use fallback model if configured
        if self.config.fallback_model and self.config.fallback_model in candidates:
            return self.config.fallback_model

        # Select best alternative
        return self.select_model(task_type, quality_preference="speed")

    def update_model_performance(
        self,
        model_name: str,
        task_type: TaskType,
        response_time: float,
        success: bool,
        confidence: float | None = None,
        tokens_per_second: float | None = None
    ) -> None:
        """Update model performance metrics.
        
        Args:
            model_name: Name of the model
            task_type: Type of task
            response_time: Response time in milliseconds
            success: Whether the request succeeded
            confidence: Optional confidence score
            tokens_per_second: Optional tokens per second
        """
        metrics_key = f"{model_name}_{task_type.value}"

        # Get or create metrics
        if metrics_key not in self.performance_metrics:
            self.performance_metrics[metrics_key] = ModelPerformanceMetrics(
                model_name=model_name,
                task_type=task_type
            )

        # Update metrics
        self.performance_metrics[metrics_key].update_metrics(
            response_time_ms=response_time,
            success=success,
            confidence=confidence,
            tokens_per_second=tokens_per_second
        )

        # Clear failure record if successful
        if success and model_name in self.failed_models:
            del self.failed_models[model_name]

    def get_model_statistics(self) -> dict[str, Any]:
        """Get model performance statistics.
        
        Returns:
            Dictionary with model statistics
        """
        stats = {
            "available_models": len(self.available_models),
            "failed_models": len(self.failed_models),
            "performance_metrics": {}
        }

        for key, metrics in self.performance_metrics.items():
            stats["performance_metrics"][key] = {
                "success_rate": metrics.success_rate,
                "avg_response_time_ms": metrics.avg_response_time_ms,
                "total_requests": metrics.total_requests,
                "avg_confidence": metrics.avg_confidence
            }

        return stats

    def update_available_models(self, models: list[ModelInfo]) -> None:
        """Update available models.
        
        Args:
            models: List of available models
        """
        self.available_models.clear()
        self.model_info.clear()

        for model_info in models:
            if model_info.is_available:
                self.available_models.add(model_info.name)
                self.model_info[model_info.name] = model_info

        logger.info(f"Updated available models: {len(self.available_models)} models")

    def add_routing_rule(self, rule: TaskRoutingRule) -> None:
        """Add a routing rule.
        
        Args:
            rule: Routing rule to add
        """
        self.config.add_routing_rule(rule)
        logger.info(f"Added routing rule: {rule.task_type} -> {rule.model_name}")

    def remove_routing_rule(self, task_type: TaskType, model_name: str) -> None:
        """Remove a routing rule.
        
        Args:
            task_type: Task type
            model_name: Model name
        """
        self.config.remove_routing_rule(task_type, model_name)
        logger.info(f"Removed routing rule: {task_type} -> {model_name}")

    def get_routing_rules(self) -> list[TaskRoutingRule]:
        """Get all routing rules.
        
        Returns:
            List of routing rules
        """
        return self.config.routing_rules

    def optimize_routing(self) -> None:
        """Optimize routing rules based on performance metrics."""
        logger.info("Optimizing routing rules based on performance")

        # Group metrics by task type
        task_metrics = defaultdict(list)
        for key, metrics in self.performance_metrics.items():
            model_name, task_type_str = key.rsplit("_", 1)
            task_type = TaskType(task_type_str)
            task_metrics[task_type].append((model_name, metrics))

        # Update routing rules for each task type
        for task_type, metrics_list in task_metrics.items():
            # Find best performing model for this task
            best_model = max(
                metrics_list,
                key=lambda x: x[1].success_rate * 0.5 + (1.0 / max(x[1].avg_response_time_ms, 1.0)) * 0.5
            )

            model_name = best_model[0]

            # Update or create routing rule
            existing_rule = self.config.get_routing_rule(task_type)
            if existing_rule:
                if existing_rule.model_name != model_name:
                    logger.info(f"Updating routing rule: {task_type} -> {model_name}")
                    existing_rule.model_name = model_name
            else:
                # Create new rule
                new_rule = TaskRoutingRule(
                    task_type=task_type,
                    model_name=model_name,
                    priority=10,
                    enabled=True
                )
                self.config.add_routing_rule(new_rule)
                logger.info(f"Created new routing rule: {task_type} -> {model_name}")

    def health_check(self) -> dict[str, Any]:
        """Perform health check on model router.
        
        Returns:
            Health check results
        """
        return {
            "status": "healthy",
            "available_models": len(self.available_models),
            "failed_models": len(self.failed_models),
            "routing_rules": len(self.config.routing_rules),
            "performance_metrics": len(self.performance_metrics),
            "strategy": self.config.strategy
        }
