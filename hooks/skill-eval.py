#!/usr/bin/env python3
"""
Skill Activation Hook

Evaluates user prompts against installed skill rules and outputs
activation instructions for matching skills.

Usage:
    echo "user prompt" | python skill-eval.py
    python skill-eval.py "user prompt"
    python skill-eval.py --file /path/to/file.py
    python skill-eval.py --file /path/to/file.py "user prompt"

Environment:
    SKILLS_DIR - Override default skills directory (default: ~/.claude/skills)
    VERBOSE    - Set to 1 for debug output
"""

import json
import os
import re
import sys
from pathlib import Path
from typing import List, Optional, Tuple


def get_skills_dir() -> Path:
    """Get the skills directory from environment or default."""
    return Path(os.environ.get("SKILLS_DIR", Path.home() / ".claude" / "skills"))


def debug(msg: str) -> None:
    """Print debug message if VERBOSE is set."""
    if os.environ.get("VERBOSE") == "1":
        print(f"[DEBUG] {msg}", file=sys.stderr)


def load_rules(skill_dir: Path) -> Optional[dict]:
    """Load rules.json from a skill directory."""
    rules_file = skill_dir / "rules.json"
    if not rules_file.exists():
        return None
    try:
        with open(rules_file) as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        debug(f"Invalid JSON in {rules_file}: {e}")
        return None


def match_keywords(prompt: str, rules: dict) -> bool:
    """Check if prompt contains any trigger keywords."""
    prompt_lower = prompt.lower()
    keywords = rules.get("promptTriggers", {}).get("keywords", [])
    
    for keyword in keywords:
        if keyword.lower() in prompt_lower:
            debug(f"Keyword match: '{keyword}'")
            return True
    return False


def match_intent_patterns(prompt: str, rules: dict) -> bool:
    """Check if prompt matches any intent patterns."""
    prompt_lower = prompt.lower()
    patterns = rules.get("promptTriggers", {}).get("intentPatterns", [])
    
    for pattern in patterns:
        try:
            if re.search(pattern, prompt_lower, re.IGNORECASE):
                debug(f"Intent pattern match: '{pattern}'")
                return True
        except re.error as e:
            debug(f"Invalid regex pattern '{pattern}': {e}")
    return False


def match_file_content(file_path: Path, rules: dict) -> bool:
    """Check if file content matches any content patterns."""
    if not file_path.exists():
        return False
    
    try:
        content = file_path.read_text()
    except Exception as e:
        debug(f"Error reading {file_path}: {e}")
        return False
    
    patterns = rules.get("fileTriggers", {}).get("contentPatterns", [])
    
    for pattern in patterns:
        try:
            if re.search(pattern, content):
                debug(f"File content match: '{pattern}' in {file_path}")
                return True
        except re.error as e:
            debug(f"Invalid regex pattern '{pattern}': {e}")
    return False


def get_priority_num(priority: str) -> int:
    """Convert priority string to number for sorting."""
    priorities = {"critical": 4, "high": 3, "medium": 2, "low": 1}
    return priorities.get(priority, 2)


def evaluate_skills(prompt: str, file_path: Optional[Path] = None) -> List[Tuple[str, int]]:
    """Evaluate all skills and return matches sorted by priority."""
    skills_dir = get_skills_dir()
    
    if not skills_dir.exists():
        debug(f"Skills directory not found: {skills_dir}")
        return []
    
    matches = []
    
    for skill_dir in skills_dir.iterdir():
        if not skill_dir.is_dir():
            continue
        
        skill_name = skill_dir.name
        rules = load_rules(skill_dir)
        
        if not rules:
            debug(f"No rules.json for skill: {skill_name}")
            continue
        
        mode = rules.get("activation", {}).get("mode", "auto")
        
        # Skip manual-only skills
        if mode == "manual":
            debug(f"Skipping manual skill: {skill_name}")
            continue
        
        matched = False
        
        # Check prompt triggers
        if prompt:
            if match_keywords(prompt, rules):
                matched = True
            elif match_intent_patterns(prompt, rules):
                matched = True
        
        # Check file triggers if file provided
        if file_path and not matched:
            if match_file_content(file_path, rules):
                matched = True
        
        if matched:
            priority = rules.get("activation", {}).get("priority", "medium")
            priority_num = get_priority_num(priority)
            matches.append((skill_name, priority_num))
            debug(f"Skill matched: {skill_name} (priority: {priority})")
    
    # Sort by priority (highest first)
    matches.sort(key=lambda x: x[1], reverse=True)
    return matches


def output_activation(matches: List[Tuple[str, int]]) -> None:
    """Output activation instructions for matched skills."""
    if not matches:
        debug("No skills matched")
        return
    
    skills_dir = get_skills_dir()
    
    print("SKILL_ACTIVATION_REQUIRED")
    print()
    print("The following skills should be activated for this task:")
    print()
    
    for skill_name, _ in matches:
        skill_file = skills_dir / skill_name / "SKILL.md"
        if skill_file.exists():
            print(f"### {skill_name}")
            print()
            print("To activate, read the skill file:")
            print("```")
            print(f"cat {skill_file}")
            print("```")
            print()
            print("Or inline activation:")
            print()
            
            # Output first 50 lines as preview
            with open(skill_file) as f:
                for i, line in enumerate(f):
                    if i >= 50:
                        break
                    print(line, end="")
            
            print()
            print("... (read full file for complete guidelines)")
            print()
            print("---")
            print()


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate prompt against skill activation rules")
    parser.add_argument("prompt", nargs="?", help="User prompt to evaluate")
    parser.add_argument("--file", "-f", type=Path, help="File path for context-based activation")
    
    args = parser.parse_args()
    
    # Read prompt from stdin if not provided as argument
    prompt = args.prompt
    if not prompt and not sys.stdin.isatty():
        prompt = sys.stdin.read().strip()
    
    if not prompt and not args.file:
        print("Error: No prompt or file provided", file=sys.stderr)
        sys.exit(1)
    
    debug(f"Evaluating prompt: {prompt}")
    debug(f"File path: {args.file or 'none'}")
    debug(f"Skills dir: {get_skills_dir()}")
    
    matches = evaluate_skills(prompt or "", args.file)
    output_activation(matches)


if __name__ == "__main__":
    main()
