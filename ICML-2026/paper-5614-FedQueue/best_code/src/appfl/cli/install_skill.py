#!/usr/bin/env python3
"""APPFL agent skill installer — appfl-install-skill."""

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

SKILL_REPO = "https://github.com/APPFL/appfl-agent-skill"
ISSUES_URL = "https://github.com/APPFL/APPFL/issues"
NEW_ISSUE_URL = "https://github.com/APPFL/APPFL/issues/new"

# ─── Terminal color helpers ───────────────────────────────────────────────────

_colors = sys.stdout.isatty() and os.environ.get("NO_COLOR") is None


def _c(code: str, text: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _colors else text


def cyan(t: str) -> str:
    return _c("96", t)


def green(t: str) -> str:
    return _c("92", t)


def yellow(t: str) -> str:
    return _c("93", t)


def red(t: str) -> str:
    return _c("91", t)


def blue(t: str) -> str:
    return _c("94", t)


def bold(t: str) -> str:
    return _c("1", t)


def dim(t: str) -> str:
    return _c("2", t)


def magenta(t: str) -> str:
    return _c("95", t)


# ─── Banner ───────────────────────────────────────────────────────────────────

_BANNER_TITLE = "·  A P P F L   A G E N T   S K I L L S  ·"
_BANNER_SUB1 = "Advanced Privacy-Preserving Federated Learning"
_BANNER_SUB2 = "Skill Installer  v1.0"
_BOX_W = max(len(_BANNER_TITLE), len(_BANNER_SUB1)) + 8  # interior width


def _make_banner() -> list[str]:
    W = _BOX_W
    H = "═" * W
    rows = [
        f"╔{H}╗",
        f"║{'':{W}}║",
        f"║{_BANNER_TITLE:^{W}}║",
        f"║{'':{W}}║",
        f"║{_BANNER_SUB1:^{W}}║",
        f"║{_BANNER_SUB2:^{W}}║",
        f"║{'':{W}}║",
        f"╚{H}╝",
    ]
    return ["", *("  " + r for r in rows), ""]


def print_banner() -> None:
    for line in _make_banner():
        if _BANNER_TITLE in line:
            print(bold(cyan(line)))
        elif _BANNER_SUB1 in line or _BANNER_SUB2 in line:
            print(dim(line))
        elif line.strip():
            print(blue(line))
        else:
            print(line)


# ─── Agent definitions ────────────────────────────────────────────────────────

AGENTS: dict[str, dict] = {
    "claude": {
        "label": "Claude Code  (Anthropic)",
        "global_path": Path.home() / ".claude" / "skills" / "appfl",
        "local_subdir": Path(".claude") / "skills" / "appfl",
        "invoke": "/appfl",
        "docs_url": "https://docs.anthropic.com/en/docs/claude-code/skills",
    },
    "codex": {
        "label": "Codex CLI    (OpenAI)",
        "global_path": Path.home() / ".codex" / "skills" / "appfl",
        "local_subdir": Path(".codex") / "skills" / "appfl",
        "invoke": "$appfl",
        "docs_url": "https://github.com/openai/codex",
    },
}

# Files/dirs to copy from the cloned repo (everything except .git and README.md)
_SKIP = {".git", "README.md"}


# ─── Utilities ────────────────────────────────────────────────────────────────


def _hr(char: str = "─", width: int = 60) -> str:
    return dim(char * width)


def _step(n: int, text: str) -> None:
    print(f"\n  {bold(cyan(f'[{n}]'))} {text}")


def _ok(text: str) -> None:
    print(f"      {green('✔')}  {text}")


def _warn(text: str) -> None:
    print(f"      {yellow('!')}  {text}")


def _err(text: str) -> None:
    print(f"      {red('✗')}  {text}", file=sys.stderr)


def _prompt(question: str, choices: list[str], default: str | None = None) -> str:
    """Print a numbered menu and return the user's choice string."""
    print()
    for i, choice in enumerate(choices, 1):
        marker = bold(cyan(f"  [{i}]"))
        suffix = dim("  ← default") if default and choice == default else ""
        print(f"{marker}  {choice}{suffix}")
    print()

    default_hint = (
        f" [{bold(cyan(str(choices.index(default) + 1)))}]" if default else ""
    )
    while True:
        raw = input(f"  {question}{default_hint}: ").strip()
        if raw == "" and default:
            return default
        if raw.isdigit() and 1 <= int(raw) <= len(choices):
            return choices[int(raw) - 1]
        _warn("Invalid choice — please enter a number from the list.")


def _check_git() -> bool:
    return shutil.which("git") is not None


def _run(
    cmd: list[str], cwd: Path | None = None, capture: bool = True
) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        cwd=cwd,
        capture_output=capture,
        text=True,
    )


# ─── Fetch skill ──────────────────────────────────────────────────────────────


def fetch_skill(tmp_dir: Path) -> Path:
    """Clone or update the skill repo into tmp_dir; return the repo root."""
    dest = tmp_dir / "appfl-agent-skill"
    print(f"\n  {dim('Cloning')} {cyan(SKILL_REPO)} …")
    result = _run(["git", "clone", "--depth=1", SKILL_REPO, str(dest)])
    if result.returncode != 0:
        _err("git clone failed:")
        print(f"  {dim(result.stderr.strip())}", file=sys.stderr)
        sys.exit(1)
    _ok("Fetched latest skill from GitHub")
    return dest


# ─── Install ──────────────────────────────────────────────────────────────────


def install_skill(src: Path, dest: Path) -> None:
    """Copy skill files from src into dest, creating dest if needed."""
    if dest.exists():
        _warn(f"Destination already exists: {dest}")
        reply = input(f"  Overwrite? {bold(cyan('[y/N]'))}: ").strip().lower()
        if reply != "y":
            print(f"  {dim('Skipped.')} Existing installation left unchanged.")
            return
        shutil.rmtree(dest)

    dest.mkdir(parents=True, exist_ok=True)

    for item in src.iterdir():
        if item.name in _SKIP:
            continue
        target = dest / item.name
        if item.is_dir():
            shutil.copytree(item, target)
        else:
            shutil.copy2(item, target)

    _ok(f"Installed → {bold(str(dest))}")


# ─── Post-install instructions ────────────────────────────────────────────────


def _print_claude_instructions(path: Path, scope: str) -> None:
    is_local = scope == "local"
    print(f"""
  {bold(cyan("How to use with Claude Code"))}
  {_hr()}

  The skill is now installed at:
    {green(str(path))}

  {bold("Invoke it")} by typing in any Claude Code session:

    {bold(cyan("/appfl"))}

  Claude will load the skill and guide you through:
    • Configuring APPFL server/client YAML files
    • Choosing the right communicator (gRPC, MPI, Globus Compute)
    • Writing custom trainers and aggregators
    • Setting up metrics, logging, and data readiness reports

  {bold("Example prompts:")}
    {dim("/appfl  Set up an APPFL gRPC federation with 3 clients.")}
    {dim("/appfl  Write a custom FedProx trainer for APPFL.")}
    {dim("/appfl  Configure APPFL with Globus Compute endpoints.")}

  {bold("Scope:")} {"This skill is available in the {bold('current project only')}." if is_local else f"This skill is available {bold('globally')} across all your projects."}

  {bold("Docs:")} https://docs.anthropic.com/en/docs/claude-code/skills
""")


def _print_codex_instructions(path: Path, scope: str) -> None:
    is_local = scope == "local"
    print(f"""
  {bold(cyan("How to use with Codex CLI"))}
  {_hr()}

  The skill is now installed at:
    {green(str(path))}

  {bold("Invoke it")} in a Codex session with:

    {bold(cyan("$appfl"))}       — triggers the skill automatically
    {bold(cyan("Use $appfl"))} — explicit invocation

  {bold("Example prompts:")}
    {dim("Use $appfl to configure an APPFL gRPC application.")}
    {dim("Use $appfl to write a custom APPFL trainer that returns train_loss metadata.")}
    {dim("Use $appfl to set up APPFL with Globus Compute endpoints.")}

  {bold("Scope:")} {"Current project only." if is_local else "Available globally across all your projects."}

  {bold("Docs:")} https://github.com/openai/codex
""")


def _print_other_agent_instructions() -> None:
    print(f"""
  {bold(yellow("Using APPFL skills with other AI coding agents"))}
  {_hr()}

  The APPFL skill lives at:
    {cyan(SKILL_REPO)}

  It contains:
    • {bold("SKILL.md")}          — main instructions and workflow guide
    • {bold("references/")}       — detailed docs for configuration, trainers,
                          aggregators, and metrics/logging

  {bold("General approach for any agent:")}

    1. Clone the repo:
       {dim(f"git clone {SKILL_REPO}")}

    2. Point your agent at SKILL.md at the start of a session, e.g.:
       {dim('"Read appfl-agent-skill/SKILL.md and use it to guide my work."')}

    3. The skill will instruct the agent to load only the relevant
       reference file for your task (configuration, trainer, etc.).

  {bold("Want native support for your agent?")}
  Open a GitHub issue and let us know which agent you use:
    {cyan(NEW_ISSUE_URL)}
""")


# ─── Path selection ───────────────────────────────────────────────────────────


def select_path(agent_key: str, agent_cfg: dict) -> tuple[Path, str]:
    """Return (resolved install path, scope label)."""
    global_path: Path = agent_cfg["global_path"]
    local_path: Path = Path.cwd() / agent_cfg["local_subdir"]

    choice = _prompt(
        "Where should the skill be installed?",
        [
            f"Global  — {dim(str(global_path))}",
            f"Local   — {dim(str(local_path))}  (current project only)",
            "Custom  — enter your own path",
        ],
        default=f"Global  — {dim(str(global_path))}",
    )

    if choice.startswith("Global"):
        return global_path, "global"
    elif choice.startswith("Local"):
        return local_path, "local"
    else:
        while True:
            raw = input(f"\n  {bold('Enter path')}: ").strip()
            if raw:
                p = Path(raw).expanduser().resolve()
                return p, "custom"
            _warn("Path cannot be empty.")


# ─── Main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    print_banner()

    # ── Step 1: Check git ────────────────────────────────────────────────────
    _step(1, "Checking prerequisites …")
    if not _check_git():
        _err("`git` is not installed or not on PATH. Please install git and retry.")
        sys.exit(1)
    _ok("git found")

    # ── Step 2: Fetch skill ──────────────────────────────────────────────────
    _step(2, "Fetching latest APPFL agent skill from GitHub …")
    tmp_dir = Path(tempfile.mkdtemp(prefix="appfl-skill-"))
    try:
        skill_src = fetch_skill(tmp_dir)

        # ── Step 3: Choose agent(s) ──────────────────────────────────────────
        _step(3, "Which coding agent would you like to install for?")
        agent_labels = [cfg["label"] for cfg in AGENTS.values()]
        agent_keys = list(AGENTS.keys())

        both_label = "Both  (Claude Code + Codex)"
        other_label = "Other agent …  (get general instructions)"

        all_choices = agent_labels + [both_label, other_label]
        chosen_label = _prompt("Select an option", all_choices)

        if chosen_label == both_label:
            selected_agents = agent_keys
        elif chosen_label == other_label:
            _print_other_agent_instructions()
            reply = (
                input(
                    f"  Would you also like to install for Claude Code or Codex? {bold(cyan('[y/N]'))}: "
                )
                .strip()
                .lower()
            )
            if reply != "y":
                print(f"\n  {dim('All done. No files were written.')}\n")
                return
            selected_agents = []
            pick = _prompt("Install for which agent?", agent_labels + ["Both"])
            if pick == "Both":
                selected_agents = agent_keys
            else:
                idx = agent_labels.index(pick)
                selected_agents = [agent_keys[idx]]
        else:
            idx = agent_labels.index(chosen_label)
            selected_agents = [agent_keys[idx]]

        # ── Step 4: Install for each selected agent ──────────────────────────
        installs: list[tuple[str, Path, str]] = []  # (agent_key, path, scope)

        for agent_key in selected_agents:
            agent_cfg = AGENTS[agent_key]
            print(f"\n  {_hr()}")
            print(f"  {bold('Installing for')} {cyan(agent_cfg['label'])}")
            print(f"  {_hr()}")
            dest, scope = select_path(agent_key, agent_cfg)
            install_skill(skill_src, dest)
            installs.append((agent_key, dest, scope))

        # ── Step 5: Post-install instructions ────────────────────────────────
        if installs:
            print(f"\n  {_hr('═')}")
            print(
                f"  {bold(green('  Installation complete! Here is how to use your skill:'))}"
            )
            print(f"  {_hr('═')}")

            for agent_key, path, scope in installs:
                if agent_key == "claude":
                    _print_claude_instructions(path, scope)
                elif agent_key == "codex":
                    _print_codex_instructions(path, scope)

        print(f"  {dim('Questions or requests? Open an issue at:')}")
        print(f"  {cyan(ISSUES_URL)}\n")

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
