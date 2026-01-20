#!/usr/bin/env bash
#
# Install agent skills for AI coding assistants
#
# Usage:
#   ./install.sh <platform...> [-- skill...]
#
# Platforms:
#   claude   - Install to ~/.claude/skills/
#   opencode - Install to ~/.config/opencode/skills/
#   gemini   - Install to ~/.gemini/skills/
#   cursor   - Install to ~/.cursor/skills/
#   copilot  - Install to ~/.copilot/skills/
#   all      - Install to all platform directories
#
# Examples:
#   ./install.sh claude                       # All skills to Claude Code
#   ./install.sh gemini -- pymc-modeling      # Specific skill to Gemini CLI
#   ./install.sh claude cursor opencode       # All skills to multiple platforms
#   ./install.sh all                          # All skills to all platforms
#   ./install.sh all -- pymc-modeling         # Specific skill to all platforms

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SKILLS_DIR="$SCRIPT_DIR/skills"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m' # No Color

info() { echo -e "${BLUE}[INFO]${NC} $*"; }
success() { echo -e "${GREEN}[OK]${NC} $*"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*" >&2; }

usage() {
    cat << EOF
Usage: ./install.sh <platform...> [-- skill...]

Platforms:
  claude    ~/.claude/skills/
  opencode  ~/.config/opencode/skills/
  gemini    ~/.gemini/skills/
  cursor    ~/.cursor/skills/
  copilot   ~/.copilot/skills/
  all       All of the above

Examples:
  ./install.sh claude                    # All skills to Claude Code
  ./install.sh gemini -- pymc-modeling   # Specific skill to Gemini CLI
  ./install.sh claude cursor opencode    # All skills to multiple platforms
  ./install.sh all                       # All skills to all platforms

Available skills:
EOF
    list_skills | sed 's/^/  /'
    exit 1
}

# Get target directory for platform
get_target_dir() {
    local platform="$1"
    case "$platform" in
        claude)   echo "$HOME/.claude/skills" ;;
        opencode) echo "$HOME/.config/opencode/skills" ;;
        gemini)   echo "$HOME/.gemini/skills" ;;
        cursor)   echo "$HOME/.cursor/skills" ;;
        copilot)  echo "$HOME/.copilot/skills" ;;
        *) error "Unknown platform: $platform"; exit 1 ;;
    esac
}

# Check if argument is a valid platform
is_platform() {
    case "$1" in
        claude|opencode|gemini|cursor|copilot|all) return 0 ;;
        *) return 1 ;;
    esac
}

# Install a single skill to a target directory
install_skill() {
    local skill="$1"
    local target_dir="$2"
    local skill_src="$SKILLS_DIR/$skill"
    local skill_dst="$target_dir/$skill"

    if [[ ! -d "$skill_src" ]]; then
        error "Skill not found: $skill"
        return 1
    fi

    mkdir -p "$target_dir"

    if [[ -d "$skill_dst" ]]; then
        warn "Overwriting existing skill: $skill_dst"
        rm -rf "$skill_dst"
    fi

    cp -r "$skill_src" "$skill_dst"
    success "Installed $skill -> $skill_dst"
}

# List available skills
list_skills() {
    find "$SKILLS_DIR" -maxdepth 1 -mindepth 1 -type d -exec basename {} \; 2>/dev/null | sort
}

# Extract description from SKILL.md frontmatter
extract_description() {
    local skill_file="$1"
    local desc=""
    
    # Try to extract multi-line description
    desc=$(awk '
        /^---$/ { if (++n == 2) exit }
        n == 1 && /^description:/ {
            # Remove "description:" prefix and optional ">" or "|"
            sub(/^description:[[:space:]]*[>|]?[[:space:]]*/, "")
            if (length($0) > 0) print
            capture = 1
            next
        }
        capture && /^[[:space:]]/ {
            sub(/^[[:space:]]+/, "")
            printf " %s", $0
        }
        capture && /^[a-z]/ { exit }
    ' "$skill_file" 2>/dev/null | tr -s ' ' | sed 's/^ //')
    
    # Fallback: simple single-line extraction
    if [[ -z "$desc" ]]; then
        desc=$(grep -m1 '^description:' "$skill_file" 2>/dev/null | sed 's/^description:[[:space:]]*//')
    fi
    
    echo "$desc"
}

# Print Claude Code registration instructions
print_claude_instructions() {
    local skills=("$@")
    
    echo ""
    echo -e "${BOLD}${YELLOW}Claude Code requires manual registration.${NC}"
    echo ""
    echo "Add the following to your ~/.claude/CLAUDE.md file in the <available_skills> section:"
    echo ""
    echo -e "${BOLD}---${NC}"
    
    for skill in "${skills[@]}"; do
        local skill_file="$SKILLS_DIR/$skill/SKILL.md"
        if [[ -f "$skill_file" ]]; then
            local desc
            desc=$(extract_description "$skill_file")
            echo "<skill>"
            echo "  <name>$skill</name>"
            echo "  <description>$desc</description>"
            echo "</skill>"
        fi
    done
    
    echo -e "${BOLD}---${NC}"
    echo ""
}

main() {
    if [[ $# -eq 0 ]]; then
        usage
    fi

    local platforms=()
    local skills=()
    local parsing_skills=false

    # Parse arguments
    for arg in "$@"; do
        if [[ "$arg" == "--" ]]; then
            parsing_skills=true
            continue
        fi
        
        if $parsing_skills; then
            skills+=("$arg")
        elif is_platform "$arg"; then
            if [[ "$arg" == "all" ]]; then
                platforms=(claude opencode gemini cursor copilot)
            else
                platforms+=("$arg")
            fi
        else
            # Assume it's a skill name if not a platform
            skills+=("$arg")
        fi
    done

    # Validate we have at least one platform
    if [[ ${#platforms[@]} -eq 0 ]]; then
        error "No valid platform specified"
        echo ""
        usage
    fi

    # Remove duplicates from platforms
    local unique_platforms=()
    for p in "${platforms[@]}"; do
        local found=false
        for up in "${unique_platforms[@]:-}"; do
            if [[ "$p" == "$up" ]]; then
                found=true
                break
            fi
        done
        if ! $found; then
            unique_platforms+=("$p")
        fi
    done
    platforms=("${unique_platforms[@]}")

    # If no skills specified, install all
    if [[ ${#skills[@]} -eq 0 ]]; then
        mapfile -t skills < <(list_skills)
    fi

    if [[ ${#skills[@]} -eq 0 ]]; then
        error "No skills found in $SKILLS_DIR"
        exit 1
    fi

    info "Skills to install: ${skills[*]}"
    info "Target platforms: ${platforms[*]}"
    echo ""

    # Install to each platform
    for p in "${platforms[@]}"; do
        local target_dir
        target_dir=$(get_target_dir "$p")
        info "Installing to $p ($target_dir)..."
        
        for skill in "${skills[@]}"; do
            install_skill "$skill" "$target_dir"
        done
        echo ""
    done

    # Print Claude-specific instructions if Claude was a target
    for p in "${platforms[@]}"; do
        if [[ "$p" == "claude" ]]; then
            print_claude_instructions "${skills[@]}"
            break
        fi
    done

    success "Installation complete!"
}

main "$@"
