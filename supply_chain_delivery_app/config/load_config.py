"""
Load agent and task definitions from Markdown config files.

Prompt files live under  config/prompts/  in two subdirectories:
  - agents/   — per-agent instruction files (role, goal, backstory, task, output)
  - shared/   — cross-cutting prompts (format_summary, safety_rules, etc.)

Combines role, goal, backstory into a single instruction string for OpenAI SDK Agent.
"""

import re
from pathlib import Path
from typing import Any

import yaml


def _config_dir() -> Path:
    """Return the config directory (next to this file)."""
    return Path(__file__).resolve().parent


def _prompts_dir() -> Path:
    """Return the prompts/ directory tree root."""
    return _config_dir() / "prompts"


def _agents_dir() -> Path:
    """Return the agents/ subdirectory containing per-agent .md files."""
    return _prompts_dir() / "agents"


def _shared_dir() -> Path:
    """Return the shared/ subdirectory for cross-cutting prompt files."""
    return _prompts_dir() / "shared"


def load_yaml(name: str) -> dict[str, Any]:
    """Load a YAML file from the config directory (kept for backward compatibility)."""
    path = _config_dir() / name
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _parse_agent_md(text: str) -> dict[str, str]:
    """Parse a Markdown agent file into a dict of section_name -> content."""
    sections: dict[str, str] = {}
    current_key: str | None = None
    current_lines: list[str] = []

    for line in text.splitlines():
        heading = re.match(r"^#{1,2}\s+(.+)$", line)
        if heading:
            # Save previous section
            if current_key is not None:
                sections[current_key] = "\n".join(current_lines).strip()
            current_key = heading.group(1).strip().lower()
            current_lines = []
        else:
            current_lines.append(line)

    # Save last section
    if current_key is not None:
        sections[current_key] = "\n".join(current_lines).strip()

    return sections


def _find_prompt_file(agent_key: str) -> Path:
    """Locate the .md file for *agent_key*, checking shared/ then agents/."""
    for d in (_shared_dir(), _agents_dir()):
        p = d / f"{agent_key}.md"
        if p.exists():
            return p
    raise FileNotFoundError(
        f"Prompt file not found for '{agent_key}' in agents/ or shared/"
    )


def _load_agent_md(agent_key: str) -> dict[str, str]:
    """Load and parse an agent .md file by key name."""
    path = _find_prompt_file(agent_key)
    return _parse_agent_md(path.read_text(encoding="utf-8"))


def build_instruction(agent_key: str, task_key: str | None = None) -> str:
    """
    Build the full instruction string for an agent from its .md file.

    For regular agents, combines Role + Goal + Backstory as:
      You are {role}. Your goal is {goal}. {backstory}

    If task_key is provided OR the .md file has ## Task / ## Expected Output,
    the task description is appended.

    For master_expert, the full .md content is returned as-is (it's already
    a complete instruction prompt).
    """
    sections = _load_agent_md(agent_key)

    # Master expert: prepend security guardrails + chatbot behavior (shared), then agent content.
    # Layer order (highest priority first): security_guardrails → chatbot_behavior → master_expert
    if agent_key == "master_expert":
        _shared = _config_dir() / "prompts" / "shared"
        _security = (_shared / "security_guardrails.md").read_text(encoding="utf-8").strip()
        _chatbot = (_shared / "chatbot_behavior.md").read_text(encoding="utf-8").strip()
        _expert = _find_prompt_file(agent_key).read_text(encoding="utf-8").strip()
        return "\n\n---\n\n".join([_security, _chatbot, _expert])

    # format_summary: return full file content as-is
    if agent_key == "format_summary":
        return _find_prompt_file(agent_key).read_text(encoding="utf-8").strip()

    role = sections.get("role", "").strip()
    goal = sections.get("goal", "").strip()
    backstory = sections.get("backstory", "").strip()

    parts = []
    if role:
        parts.append(f"You are {role}.")
    if goal:
        parts.append(f"Your goal is {goal}.")
    if backstory:
        parts.append(backstory)

    instruction = " ".join(parts)

    # Append task from the .md file itself
    task_desc = sections.get("task", "").strip()
    task_output = sections.get("expected output", "").strip()
    if task_desc or task_output:
        instruction += "\n\nTask: " + (task_desc or "")
        if task_output:
            instruction += "\nExpected output: " + task_output

    # Override with tasks.yaml if task_key provided (backward compat)
    if task_key:
        tasks_path = _config_dir() / "tasks.yaml"
        if tasks_path.exists():
            tasks_cfg = load_yaml("tasks.yaml")
            if task_key in tasks_cfg:
                task = tasks_cfg[task_key]
                desc = (task.get("description") or "").strip()
                out = (task.get("expected_output") or "").strip()
                if desc or out:
                    instruction += "\n\nTask: " + (desc or "")
                    if out:
                        instruction += "\nExpected output: " + out

    return instruction.strip()


def get_instruction(agent_key: str, task_key: str | None = None) -> str:
    """
    Public API: get the combined agent (and optional task) instruction.
    Use this in notebook/agent code where Agent(instructions=...) is set.
    """
    return build_instruction(agent_key, task_key)
