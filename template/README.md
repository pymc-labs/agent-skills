# Creating a New Skill

This template provides a starting point for creating new Agent Skills. Copy this directory to `skills/your-skill-name/` and customize it.

## Quick Start

```bash
# Copy the template
cp -r template skills/my-new-skill

# Edit the skill files
cd skills/my-new-skill
# Edit SKILL.md with your content
```

## Required Files

### SKILL.md

The only required file. Must contain:

1. **YAML Frontmatter** with required fields:
   - `name`: Unique identifier (1-64 chars, lowercase, hyphens only)
   - `description`: What the skill does and when to use it (1-1024 chars)

2. **Markdown Body** with instructions for the AI assistant

### Frontmatter Fields

| Field | Required | Description |
|-------|----------|-------------|
| `name` | Yes | Must match directory name. Lowercase alphanumeric + hyphens. |
| `description` | Yes | Describes what and when. Include trigger keywords. |
| `license` | No | License name or reference to LICENSE file |
| `compatibility` | No | Environment requirements (max 500 chars) |
| `metadata` | No | Arbitrary key-value pairs (author, version, etc.) |

### Name Validation

The `name` field must:
- Be 1-64 characters
- Contain only lowercase letters, numbers, and hyphens
- Not start or end with a hyphen
- Not contain consecutive hyphens (`--`)
- Match the parent directory name

**Valid:** `pymc-modeling`, `data-analysis`, `code-review`
**Invalid:** `PyMC-Modeling`, `-my-skill`, `my--skill`

## Recommended Structure

```
my-new-skill/
├── SKILL.md           # Required: main instructions
├── README.md          # Recommended: user documentation
└── references/        # Optional: detailed topic guides
    ├── topic-a.md
    └── topic-b.md
```

### Optional Directories

| Directory | Purpose |
|-----------|---------|
| `references/` | Detailed documentation loaded on-demand |
| `scripts/` | Executable code the AI can run |
| `assets/` | Templates, examples, static resources |

## Writing Effective Skills

### Description Tips

The `description` is critical - it determines when the AI loads your skill.

**Good:**
```yaml
description: >
  Bayesian statistical modeling with PyMC v5+. Use when building probabilistic
  models, specifying priors, running MCMC inference, or diagnosing convergence.
  Triggers on: Bayesian inference, posterior sampling, hierarchical models,
  GLMs, ArviZ diagnostics, LOO-CV, WAIC.
```

**Poor:**
```yaml
description: Helps with PyMC.
```

### Body Content Tips

- Keep SKILL.md under 500 lines; use `references/` for details
- Include runnable code examples
- Explain both "what" and "why"
- Use relative links: `[topic](references/topic.md)`
- Structure with clear headings for scannability

### Progressive Disclosure

Skills load in stages:
1. **Metadata** (~100 tokens): `name` and `description` loaded at startup
2. **Instructions** (<5k tokens): SKILL.md body loaded when triggered
3. **Resources** (as needed): Reference files loaded only when accessed

This means you can include extensive documentation in `references/` without impacting context when unused.

## Validation

Use the [skills-ref](https://github.com/agentskills/agentskills/tree/main/skills-ref) CLI to validate:

```bash
skills-ref validate ./skills/my-new-skill
```

## Testing Your Skill

1. Install to your preferred platform:
   ```bash
   ./install.sh claude -- my-new-skill
   ```

2. Start a conversation and try tasks that should trigger your skill

3. Verify the skill loads and provides appropriate guidance

## References

- [Agent Skills Specification](https://agentskills.io/specification)
- [anthropics/skills](https://github.com/anthropics/skills) - Reference implementations
