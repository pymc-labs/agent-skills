# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Purpose

This is a Claude Code skill for marimo reactive Python notebook development. The primary content is in `SKILL.md`, which provides comprehensive marimo documentation that gets loaded when the skill is invoked.

## Repository Structure

- `SKILL.md` - Main skill definition with marimo API reference, patterns, and best practices
- `assets/` - Template notebooks (`minimal_template.py`, `data_analysis_template.py`)
- `scripts/` - Utility scripts (`convert_notebook.py` for Jupyter conversion)
- `references/` - Extended documentation for UI components and wigglystuff widgets

## Commands

```bash
# Convert Jupyter to marimo
python scripts/convert_notebook.py input.ipynb [output.py]

# Validate marimo notebook
marimo check notebook.py
marimo check notebook.py --fix

# Run notebook as app
marimo run notebook.py

# Edit notebook in browser
marimo edit notebook.py
```

## Marimo File Detection

When editing any `.py` file, check if it's a marimo notebook by looking for these signatures:
- `import marimo` at the top
- `app = marimo.App(...)`
- `@app.cell` decorators on functions

If ANY of these patterns are present, invoke the `marimo-notebooks` skill before making changes. Marimo notebooks have specific conventions (reactivity, variable scoping, return tuples) that differ from regular Python.

## Skill Development Notes

When modifying this skill:
- Keep `SKILL.md` concise—it's loaded into context on every invocation
- Reference documentation goes in `references/` to avoid bloating the main skill file
- Templates in `assets/` should demonstrate idiomatic marimo patterns
- The skill description in `SKILL.md` frontmatter determines when the skill triggers
