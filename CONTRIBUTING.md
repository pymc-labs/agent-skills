# Contributing Guide

Thank you for your interest in contributing to the Agent Skills repository!

## Ways to Contribute

- **Add new skills** for probabilistic programming tools and workflows
- **Improve existing skills** with better examples, documentation, or coverage
- **Fix bugs** in skills or the install script
- **Improve documentation** for clarity and completeness

## Adding a New Skill

### 1. Create the Skill Directory

```bash
# Copy the template
cp -r template skills/your-skill-name

# The directory name must match the skill's `name` field
```

### 2. Edit SKILL.md

The `SKILL.md` file is the heart of your skill. It must contain:

**Required frontmatter:**
```yaml
---
name: your-skill-name
description: >
  Clear description of what this skill does and when to use it.
  Include trigger keywords. Max 1024 characters.
---
```

**Recommended frontmatter:**
```yaml
---
name: your-skill-name
description: >
  Clear description...
license: MIT
compatibility: Claude Code, OpenCode, Gemini CLI, Cursor, VS Code Copilot
metadata:
  author: Your Name
  version: "1.0"
---
```

**Body content should include:**
- Overview of what the skill enables
- When to use the skill (trigger conditions)
- Quick start with minimal example
- Core concepts with code examples
- Common patterns
- Best practices
- Common pitfalls
- References to detailed docs in `references/`

### 3. Add Supporting Files

**README.md** (recommended): User-facing documentation
- What the skill does
- Prerequisites and dependencies
- Installation notes
- Usage examples

**references/** (optional): Detailed topic guides
- Keep SKILL.md under 500 lines
- Move detailed content to reference files
- Use relative links: `[topic](references/topic.md)`

### 4. Test Your Skill

```bash
# Install to your preferred platform
./install.sh claude -- your-skill-name

# Test with the AI assistant
# Verify it loads for relevant tasks
# Check that guidance is accurate and helpful
```

### 5. Submit a Pull Request

1. Fork the repository
2. Create a feature branch: `git checkout -b add-skill-name`
3. Commit your changes: `git commit -m "Add skill-name skill"`
4. Push to your fork: `git push origin add-skill-name`
5. Open a pull request

## Skill Guidelines

### Naming Conventions

Skill names must:
- Be 1-64 characters
- Use only lowercase letters, numbers, and hyphens
- Not start or end with a hyphen
- Not contain consecutive hyphens (`--`)
- Match the parent directory name

**Good:** `pymc-modeling`, `polars-dataframes`, `arviz-diagnostics`
**Bad:** `PyMC_Modeling`, `-my-skill`, `my--skill`

### Description Quality

The description determines when the AI loads your skill. Make it specific:

**Good:**
```yaml
description: >
  Bayesian statistical modeling with PyMC v5+. Use when building probabilistic
  models, specifying priors, running MCMC inference, diagnosing convergence,
  or comparing models. Covers PyMC, ArviZ, nutpie, and JAX/NumPyro backends.
  Triggers on: Bayesian inference, posterior sampling, hierarchical models,
  GLMs, time series, Gaussian processes, BART, mixture models, LOO-CV, WAIC.
```

**Bad:**
```yaml
description: PyMC stuff
```

### Code Examples

- Provide runnable examples
- Use realistic but minimal code
- Include both basic and advanced patterns
- Show expected output where helpful
- Follow the tool/library's best practices

### Content Organization

```
skill-name/
├── SKILL.md              # < 500 lines, core guidance
├── README.md             # User documentation
└── references/
    ├── advanced.md       # Detailed advanced topics
    ├── troubleshooting.md
    └── api-reference.md
```

### Scope

Skills in this repository focus on **probabilistic programming and Bayesian inference**:

- Statistical modeling (PyMC, Stan, etc.)
- MCMC and variational inference
- Model diagnostics and comparison
- Bayesian workflows
- Related data science tools (ArviZ, Polars, etc.)

## Improving Existing Skills

### What to improve

- Add missing patterns or use cases
- Improve code examples
- Fix outdated information
- Add troubleshooting guidance
- Improve clarity and organization

### Process

1. Open an issue describing the improvement (optional but helpful)
2. Make your changes
3. Test the updated skill
4. Submit a pull request with clear description

## Code Style

### Markdown

- Use ATX-style headers (`#`, `##`, `###`)
- Fence code blocks with language identifier
- Use tables for structured information
- Keep lines under 100 characters where practical

### YAML Frontmatter

- Use `>` for multi-line descriptions (folded style)
- Quote strings containing special characters
- Keep metadata minimal and relevant

### Python Examples

- Follow PEP 8
- Use type hints where they add clarity
- Include imports in examples
- Prefer explicit over implicit

## Pull Request Checklist

Before submitting:

- [ ] Skill directory name matches `name` field in SKILL.md
- [ ] SKILL.md has valid YAML frontmatter
- [ ] Description is specific and includes trigger keywords
- [ ] Code examples are runnable
- [ ] README.md documents prerequisites
- [ ] Tested on at least one platform
- [ ] No sensitive information (API keys, credentials)

## Questions?

Open an issue for:
- Questions about contributing
- Suggestions for new skills
- Feedback on existing skills

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
