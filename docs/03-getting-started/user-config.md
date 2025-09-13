# User YAML Configuration

Jarvis supports YAML-based user configuration in addition to environment variables and CLI flags.

## Precedence

1. CLI flags (highest)
2. Environment variables (prefix `JARVIS_`)
3. YAML: `config/base.yaml` then `config/local.yaml` (deep-merged)
4. Defaults (lowest)

## Property Extraction Mapping

VaultReader uses this mapping to extract structured metadata from notes:

```yaml
property_extraction:
  frontmatter_tag_keys: ["tags", "keywords"]
  frontmatter_alias_keys: ["aliases"]
  inline_tag_prefixes: ["#"]
  parse_frontmatter: true
  parse_inline_tags: true
```

This yields `metadata.tags`, `metadata.aliases`, and `metadata.frontmatter` in `read_file` results, enabling downstream analyzers and tools to reason over tags and aliases consistently.

## Loader

The loader at `src/jarvis/utils/user_config.py` deep-merges the YAML files and exposes helpers for components to read.

