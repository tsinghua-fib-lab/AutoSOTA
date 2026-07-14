# Installing use-rdblearn for Code Agents

Current RDBLearn version: 0.1.2 (FastDFS 0.2.1)

Make sure you are working with the right version.

## Prerequisites

- [OpenCode.ai](https://opencode.ai), or Codex, Claude Code, or Cursor installed
- Git installed

## Installation Steps

### 1. Create the skill folder
Create a folder so agents' native skill tool discovers use-rdblearn skill, the `<skill-dir>` for Codex is `~/.codex/`, and for `claude code` is `~/.claude/`, and for `opencode` is `~/.opencode`.

```bash
rm -rf ~/<skill-dir>/skills/use-rdblearn
mkdir ~/<skill-dir>/skills/use-rdblearn
mkdir ~/<skill-dir>/skills/use-rdblearn/codes

```

### 2. Clone RDBLearn and FastDFS

Use the **release tags** for this skill bundle (not branches): RDBLearn **v0.1.2**, FastDFS **v0.2.1**.

```bash
git clone --depth 1 https://github.com/HKUSHXLab/rdblearn.git ~/<skill-dir>/skills/use-rdblearn/codes/rdblearn/
cd ~/<skill-dir>/skills/use-rdblearn/codes/rdblearn && git checkout v0.1.2

git clone --depth 1 https://github.com/HKUSHXLab/fastdfs.git ~/<skill-dir>/skills/use-rdblearn/codes/fastdfs/
cd ~/<skill-dir>/skills/use-rdblearn/codes/fastdfs && git checkout v0.2.1
```

(`git clone --branch NAME` also accepts tag names, but `v0.2.1` / `v0.1.2` are annotated tags on GitHub, not branches—`git checkout` after clone is the clearest approach.)

For GPU / flash-attn (optional, not on PyPI metadata):

```bash
pip install -r ~/<skill-dir>/skills/use-rdblearn/codes/rdblearn/requirements-gpu.txt
```

Verify the checkout:

```bash
cd ~/<skill-dir>/skills/use-rdblearn/codes/fastdfs && git describe --tags --exact-match
cd ~/<skill-dir>/skills/use-rdblearn/codes/rdblearn && git describe --tags --exact-match
```

### 3. Create the Skill

Create `~/<skill-dir>/skills/use-rdblearn/SKILL.md`:

```markdown
---
name: use-rdblearn
description: developement based on rdblearn
---

# Use RDBLearn
When asked about running rdblearn, always following:

1. Read the README and examples from `use-rdblearn/docs`.
2. Install RDBLearn
3. Write python code w.r.t. user requests
4. Run the code and debug if error raised
```

Move related documents to the skill directory

```bash
cp -r ~/<skill-dir>/skills/use-rdblearn/codes/rdblearn/README.md -r ~/<skill-dir>/skills/use-rdblearn/docs/rdblearn_README.md
cp -r ~/<skill-dir>/skills/use-rdblearn/codes/rdblearn/examples -r ~/<skill-dir>/skills/use-rdblearn/docs/rdblearn_examples
cp -r ~/<skill-dir>/skills/use-rdblearn/codes/fastdfs/README.md -r ~/<skill-dir>/skills/use-rdblearn/docs/fastdfs_README.md
cp -r ~/<skill-dir>/skills/use-rdblearn/codes/fastdfs/examples -r ~/<skill-dir>/skills/use-rdblearn/docs/fastdfs_examples
```

### 4. Provide Skill Status

Show available skills and check whether `use_rdblearn` is on the list

## Troubleshooting

### Skills not found

1. Makre sure the skill exists: `~/<skill-dir>/skills/use-rdblearn` and there is a `SKILL.md`
2. Use `skill` tool to list what's discovered


## Getting Help

- Report issues: https://github.com/HKUSHXLab/rdblearn/issues
- Full documentation: https://github.com/HKUSHXLab/rdblearn/main/skills/INSTALL.md
