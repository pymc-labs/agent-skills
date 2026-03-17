# Agent Skills for Probabilistic Programming

A library of [Agent Skills](https://agentskills.io) for probabilistic programming and Bayesian inference. These skills provide AI coding assistants with specialized knowledge for building statistical models, running MCMC inference, and analyzing results.

## What are Agent Skills?

Agent Skills are portable packages of instructions and resources that AI coding assistants load on-demand. When you work on a task matching a skill's description, the assistant loads the relevant expertise to provide more accurate, informed help.

Skills follow an [open standard](https://agentskills.io/specification) and work across multiple platforms:

| Platform | Documentation |
|----------|---------------|
| [Claude Code](https://claude.ai/code) | [Skills docs](https://docs.anthropic.com/en/docs/claude-code/skills) |
| [OpenCode](https://opencode.ai) | [Skills docs](https://opencode.ai/docs/skills/) |
| [Gemini CLI](https://geminicli.com) | [Skills docs](https://geminicli.com/docs/cli/skills/) |
| [Cursor](https://cursor.com) | [Skills docs](https://cursor.com/docs/context/skills) |
| [VS Code Copilot](https://code.visualstudio.com) | [Skills docs](https://code.visualstudio.com/docs/copilot/customization/agent-skills) |

## Available Skills

| Skill | Description |
|-------|-------------|
| [marimo-notebooks](skills/marimo-notebooks/) | Reactive Python notebooks stored as pure `.py` files. Covers marimo CLI, UI components, layout functions, SQL integration, caching, state management, and wigglystuff widgets. |
| [pymc-modeling](skills/pymc-modeling/) | Bayesian statistical modeling with PyMC v5+. Covers model specification, MCMC inference, ArviZ diagnostics, hierarchical models, GLMs, GPs, BART, time series, and more. |

## Installation

### Overview

| Platform | Install Location | Auto-Discovered |
|----------|-----------------|-----------------|
| Claude Code | `~/.claude/skills/` | Yes |
| OpenCode | `~/.config/opencode/skills/` | Yes |
| Gemini CLI | `~/.gemini/skills/` | Yes |
| Cursor | `~/.cursor/skills/` | Yes |
| VS Code Copilot | `~/.copilot/skills/` | Yes |

### Quick Install

```bash
# Clone the repository
git clone https://github.com/pymc-labs/agent-skills.git
cd agent-skills

# Install to your platform(s)
./install.sh claude              # Claude Code
./install.sh opencode            # OpenCode
./install.sh gemini              # Gemini CLI
./install.sh cursor              # Cursor
./install.sh copilot             # VS Code Copilot
./install.sh all                 # All platforms
```

### Platform Details

#### Claude Code

```bash
./install.sh claude
```

Skills are auto-discovered from `~/.claude/skills/`. Claude loads them automatically when relevant to your task, or you can invoke directly with `/skill-name`.

#### OpenCode

```bash
./install.sh opencode
```

Skills are auto-discovered from `~/.config/opencode/skills/`. No additional configuration needed.

#### Gemini CLI

```bash
./install.sh gemini
```

Skills are auto-discovered from `~/.gemini/skills/`. Verify and manage with:

```bash
gemini skills list
gemini skills enable pymc-modeling
gemini skills disable pymc-modeling
```

**Note:** Skills are experimental in Gemini CLI. Enable via `experimental.skills` setting.

#### Cursor

```bash
./install.sh cursor
```

Skills are auto-discovered from `~/.cursor/skills/`. View discovered skills in **Cursor Settings > Rules > Agent Decides**.

**Note:** Agent Skills in Cursor require the nightly release channel. Switch via **Cursor Settings > Beta > Update Channel > Nightly**.

#### VS Code Copilot

```bash
./install.sh copilot
```

Skills are auto-discovered from `~/.copilot/skills/`. Enable the `chat.useAgentSkills` setting:

1. Open VS Code Settings (`Cmd+,` / `Ctrl+,`)
2. Search for `useAgentSkills`
3. Enable **Chat: Use Agent Skills**

### Installing Specific Skills

```bash
# Install only pymc-modeling to Claude Code
./install.sh claude -- pymc-modeling

# Install specific skill to multiple platforms
./install.sh claude opencode -- pymc-modeling
```

### Manual Installation

If you prefer not to use the install script:

1. Copy the skill directory to your platform's skills folder:

   | Platform | Location |
   |----------|----------|
   | Claude Code | `~/.claude/skills/` |
   | OpenCode | `~/.config/opencode/skills/` |
   | Gemini CLI | `~/.gemini/skills/` |
   | Cursor | `~/.cursor/skills/` |
   | VS Code Copilot | `~/.copilot/skills/` |

2. Skills are auto-discovered on all platforms. No additional registration needed.

### Updating Skills

```bash
cd agent-skills
git pull
./install.sh <your-platforms>
```

### Uninstalling Skills

Remove the skill directory from your platform's skills folder:

```bash
rm -rf ~/.claude/skills/pymc-modeling
rm -rf ~/.config/opencode/skills/pymc-modeling
rm -rf ~/.gemini/skills/pymc-modeling
rm -rf ~/.cursor/skills/pymc-modeling
rm -rf ~/.copilot/skills/pymc-modeling
```

## Hook-Based Skill Activation

Skills now include **rules.json** files that define automatic activation triggers. This enables context-aware skill loading based on:

- **Keywords** in user prompts (e.g., "bayesian", "marimo")
- **Intent patterns** matching user requests (regex)
- **File content** being edited (e.g., `import pymc`)
- **File paths** matching patterns (e.g., `**/*pymc*.py`)

### Using the Skill Evaluation Hook

After installation, a hook script is available to evaluate prompts:

```bash
# Evaluate a prompt
python ~/.claude/hooks/skill-eval.py "Create a Bayesian hierarchical model"

# Output shows which skills should be activated:
# SKILL_ACTIVATION_REQUIRED
# The following skills should be activated for this task:
# ### pymc-modeling
# ...

# Evaluate with file context
python ~/.claude/hooks/skill-eval.py --file model.py "Improve this code"

# Debug mode
VERBOSE=1 python ~/.claude/hooks/skill-eval.py "fit a GP"
```

### rules.json Schema

Each skill can define activation rules:

```json
{
  "name": "pymc-modeling",
  "version": "1.0.0",
  "activation": {
    "mode": "auto",
    "priority": "high"
  },
  "promptTriggers": {
    "keywords": ["pymc", "bayesian", "mcmc", "posterior"],
    "intentPatterns": ["(create|build).*?(bayesian|probabilistic).*?(model)"]
  },
  "fileTriggers": {
    "pathPatterns": ["**/*pymc*.py"],
    "contentPatterns": ["import pymc", "pm\\.Model"]
  }
}
```

See [docs/HOOKS.md](docs/HOOKS.md) for complete documentation.

### Troubleshooting

**Skill not loading:**

1. Verify the skill directory exists in the correct location
2. Check `SKILL.md` has valid YAML frontmatter with `name` and `description`
3. Platform-specific checks:
   - Cursor: Requires nightly release channel
   - Gemini CLI: Check `experimental.skills` is enabled
   - VS Code: Check `chat.useAgentSkills` is enabled

**Permission denied:**

```bash
chmod +x install.sh
```

## Skill Structure

Each skill follows the [Agent Skills specification](https://agentskills.io/specification):

```
skill-name/
├── SKILL.md           # Main instructions (required)
├── rules.json         # Activation rules (optional)
├── README.md          # User documentation
└── references/        # Detailed reference documents
    ├── topic-a.md
    └── topic-b.md
```

See the [template/](template/) directory for a skill template.

## Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

To add a new skill:

1. Copy `template/` to `skills/your-skill-name/`
2. Edit `SKILL.md` with your content
3. Add `rules.json` with activation triggers
4. Add a `README.md` for user documentation
5. Submit a pull request

## References

- [Agent Skills Specification](https://agentskills.io/specification)
- [anthropics/skills](https://github.com/anthropics/skills) - Anthropic's reference skills

## License

MIT License. See [LICENSE](LICENSE) for details.
