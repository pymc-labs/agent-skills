#!/usr/bin/env bash
#
# Skill Activation Hook
#
# Evaluates user prompts against installed skill rules and outputs
# activation instructions for matching skills.
#
# Usage:
#   echo "user prompt" | ./skill-eval.sh
#   ./skill-eval.sh "user prompt"
#   ./skill-eval.sh --file /path/to/file.py
#
# Environment:
#   SKILLS_DIR - Override default skills directory (default: ~/.claude/skills)
#   VERBOSE    - Set to 1 for debug output
#
# Output:
#   Prints skill activation instructions for each matching skill

set -euo pipefail

# Configuration
SKILLS_DIR="${SKILLS_DIR:-$HOME/.claude/skills}"
VERBOSE="${VERBOSE:-0}"

# Colors (disabled if not TTY)
if [[ -t 1 ]]; then
    GREEN='\033[0;32m'
    YELLOW='\033[0;33m'
    BLUE='\033[0;34m'
    NC='\033[0m'
else
    GREEN='' YELLOW='' BLUE='' NC=''
fi

debug() { [[ "$VERBOSE" == "1" ]] && echo -e "${BLUE}[DEBUG]${NC} $*" >&2 || true; }
info() { echo -e "${GREEN}[SKILL]${NC} $*" >&2; }

# Check if jq is available
check_dependencies() {
    if ! command -v jq &> /dev/null; then
        echo "Error: jq is required but not installed" >&2
        exit 1
    fi
}

# Evaluate prompt against keyword list (case-insensitive)
match_keywords() {
    local prompt="$1"
    local rules_file="$2"
    
    local prompt_lower
    prompt_lower=$(echo "$prompt" | tr '[:upper:]' '[:lower:]')
    
    local keywords
    keywords=$(jq -r '.promptTriggers.keywords // [] | .[]' "$rules_file" 2>/dev/null)
    
    while IFS= read -r keyword; do
        [[ -z "$keyword" ]] && continue
        keyword_lower=$(echo "$keyword" | tr '[:upper:]' '[:lower:]')
        if [[ "$prompt_lower" == *"$keyword_lower"* ]]; then
            debug "Keyword match: '$keyword'"
            return 0
        fi
    done <<< "$keywords"
    
    return 1
}

# Evaluate prompt against intent patterns (regex)
match_intent_patterns() {
    local prompt="$1"
    local rules_file="$2"
    
    local prompt_lower
    prompt_lower=$(echo "$prompt" | tr '[:upper:]' '[:lower:]')
    
    local patterns
    patterns=$(jq -r '.promptTriggers.intentPatterns // [] | .[]' "$rules_file" 2>/dev/null)
    
    while IFS= read -r pattern; do
        [[ -z "$pattern" ]] && continue
        if echo "$prompt_lower" | grep -qiE "$pattern" 2>/dev/null; then
            debug "Intent pattern match: '$pattern'"
            return 0
        fi
    done <<< "$patterns"
    
    return 1
}

# Evaluate file content against content patterns
match_file_content() {
    local file_path="$1"
    local rules_file="$2"
    
    [[ ! -f "$file_path" ]] && return 1
    
    local patterns
    patterns=$(jq -r '.fileTriggers.contentPatterns // [] | .[]' "$rules_file" 2>/dev/null)
    
    while IFS= read -r pattern; do
        [[ -z "$pattern" ]] && continue
        if grep -qE "$pattern" "$file_path" 2>/dev/null; then
            debug "File content match: '$pattern' in $file_path"
            return 0
        fi
    done <<< "$patterns"
    
    return 1
}

# Get skill priority as number for sorting
get_priority_num() {
    local priority="$1"
    case "$priority" in
        critical) echo 4 ;;
        high)     echo 3 ;;
        medium)   echo 2 ;;
        low)      echo 1 ;;
        *)        echo 2 ;;
    esac
}

# Evaluate all skills and return matches
evaluate_skills() {
    local prompt="$1"
    local file_path="${2:-}"
    local matches=()
    
    if [[ ! -d "$SKILLS_DIR" ]]; then
        debug "Skills directory not found: $SKILLS_DIR"
        return
    fi
    
    # Find all rules.json files
    for skill_dir in "$SKILLS_DIR"/*/; do
        [[ ! -d "$skill_dir" ]] && continue
        
        local skill_name
        skill_name=$(basename "$skill_dir")
        local rules_file="$skill_dir/rules.json"
        
        if [[ ! -f "$rules_file" ]]; then
            debug "No rules.json for skill: $skill_name"
            continue
        fi
        
        local mode
        mode=$(jq -r '.activation.mode // "auto"' "$rules_file")
        
        # Skip manual-only skills
        if [[ "$mode" == "manual" ]]; then
            debug "Skipping manual skill: $skill_name"
            continue
        fi
        
        local matched=false
        
        # Check prompt triggers
        if match_keywords "$prompt" "$rules_file"; then
            matched=true
        elif match_intent_patterns "$prompt" "$rules_file"; then
            matched=true
        fi
        
        # Check file triggers if file provided
        if [[ -n "$file_path" ]] && ! $matched; then
            if match_file_content "$file_path" "$rules_file"; then
                matched=true
            fi
        fi
        
        if $matched; then
            local priority
            priority=$(jq -r '.activation.priority // "medium"' "$rules_file")
            local priority_num
            priority_num=$(get_priority_num "$priority")
            matches+=("$priority_num:$skill_name")
            debug "Skill matched: $skill_name (priority: $priority)"
        fi
    done
    
    # Sort by priority (highest first) and output
    printf '%s\n' "${matches[@]}" | sort -t: -k1 -rn | cut -d: -f2
}

# Output activation instructions for matched skills
output_activation() {
    local skills=("$@")
    
    if [[ ${#skills[@]} -eq 0 ]]; then
        debug "No skills matched"
        return
    fi
    
    echo "SKILL_ACTIVATION_REQUIRED"
    echo ""
    echo "The following skills should be activated for this task:"
    echo ""
    
    for skill in "${skills[@]}"; do
        local skill_file="$SKILLS_DIR/$skill/SKILL.md"
        if [[ -f "$skill_file" ]]; then
            echo "### $skill"
            echo ""
            echo "To activate, read the skill file:"
            echo "\`\`\`"
            echo "cat $skill_file"
            echo "\`\`\`"
            echo ""
            echo "Or inline activation:"
            echo ""
            # Output first 50 lines as preview
            head -n 50 "$skill_file"
            echo ""
            echo "... (read full file for complete guidelines)"
            echo ""
            echo "---"
            echo ""
        fi
    done
}

# Main
main() {
    check_dependencies
    
    local prompt=""
    local file_path=""
    
    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --file)
                file_path="$2"
                shift 2
                ;;
            --help|-h)
                echo "Usage: $0 [--file /path/to/file.py] [prompt]"
                echo "       echo 'prompt' | $0"
                exit 0
                ;;
            *)
                prompt="$1"
                shift
                ;;
        esac
    done
    
    # Read from stdin if no prompt argument
    if [[ -z "$prompt" ]]; then
        prompt=$(cat)
    fi
    
    if [[ -z "$prompt" && -z "$file_path" ]]; then
        echo "Error: No prompt or file provided" >&2
        exit 1
    fi
    
    debug "Evaluating prompt: $prompt"
    debug "File path: ${file_path:-none}"
    debug "Skills dir: $SKILLS_DIR"
    
    # Evaluate skills
    local matched_skills=()
    while IFS= read -r skill; do
        [[ -n "$skill" ]] && matched_skills+=("$skill")
    done < <(evaluate_skills "$prompt" "$file_path")
    
    # Output activation instructions
    output_activation "${matched_skills[@]}"
}

main "$@"
