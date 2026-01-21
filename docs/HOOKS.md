# Hook-Based Skill Activation

This document describes the hook-based activation system that automatically activates skills based on user prompts and file context.

## Overview

Skills can now be automatically activated when:
1. User prompts contain specific **keywords** (e.g., "bayesian", "marimo")
2. User prompts match **intent patterns** (regex, e.g., "create a GP model")
3. Files being edited match **path patterns** (glob, e.g., `**/*pymc*.py`)
4. Files contain **content patterns** (regex, e.g., `import pymc`)

## Skill Rules (`rules.json`)

Each skill can define activation rules in a `rules.json` file:

```json
{
  "$schema": "../../schemas/skill-rules.schema.json",
  "name": "pymc-modeling",
  "version": "1.0.0",
  "activation": {
    "mode": "auto",
    "priority": "high"
  },
  "promptTriggers": {
    "keywords": ["pymc", "bayesian", "mcmc", "posterior"],
    "intentPatterns": [
      "(create|build).*?(bayesian|probabilistic).*?(model)",
      "(sample|run).*?(mcmc|posterior)"
    ]
  },
  "fileTriggers": {
    "pathPatterns": ["**/*pymc*.py", "**/*bayesian*.py"],
    "contentPatterns": ["import pymc", "pm\\.Model"]
  },
  "contextTriggers": {
    "projectIndicators": ["pymc", "arviz"],
    "fileExtensions": [".py", ".ipynb"]
  }
}
```

### Activation Modes

- **auto**: Skill activates automatically on any trigger match
- **manual**: Skill only activates on explicit invocation
- **hybrid**: Skill suggests activation on match but requires confirmation

### Priority Levels

When multiple skills match, they're activated in priority order:
- **critical**: Always first (e.g., security-related skills)
- **high**: Before medium (e.g., domain-specific like PyMC)
- **medium**: Default priority
- **low**: After others (e.g., general utilities)

## Hook Script

The `hooks/skill-eval.sh` script evaluates prompts against installed skill rules:

```bash
# Evaluate a prompt
echo "Create a Bayesian hierarchical model" | ~/.claude/hooks/skill-eval.sh

# Evaluate with file context
~/.claude/hooks/skill-eval.sh --file model.py "Improve this code"

# Debug mode
VERBOSE=1 ~/.claude/hooks/skill-eval.sh "fit a GP"
```

## Integration Points

### Session Claude (Marimo/Fly.io)

For remote Claude sessions, activate skills by evaluating on the session:

```bash
# On the session, run the hook
echo "Create a GP model with seasonal patterns" | ~/.claude/hooks/skill-eval.sh
```

The hook will output which skills to activate and preview their content.

## Best Practices

### Writing Good Keywords

- Use domain-specific terms that uniquely identify when the skill applies
- Include common misspellings and abbreviations
- Keep the list focused to avoid false positives

### Writing Intent Patterns

- Use regex capturing groups for flexibility: `(create|build|make)`
- Include word boundaries where needed
- Test patterns against real user queries

### Priority Selection

- **high**: Domain-specific skills with clear activation criteria
- **medium**: General-purpose skills
- **low**: Utility/helper skills
