Observability
=============

Purpose
- Central place for runtime telemetry: metrics now; logging/tracing later.

Contents
- `metrics.py`: JarvisMetrics implementing `IMetrics` with counters, gauges, histograms, timers, and decorators.

Guidelines
- Services should depend on `jarvis.observability.metrics` via interfaces defined in `jarvis.core.interfaces`.
- Avoid direct logging configuration here for now; add `logging_config.py` later to centralize logging setup.

Notes
- This directory replaces the former `jarvis/monitoring`.
