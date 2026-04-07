# ALIFE Cleanup: Git Hygiene Instructions

**Tag:** `pre-ALIFE-cuts` (commit `a1dcca4`) — the last snapshot before any cleanup.
**Goal:** Strip everything from git that isn't source code, config, or lightweight documentation. The repo should be publishable on GitHub without multi-hundred-MB blobs — and without any trace of AI authorship.

---

## 0. HIGH PRIORITY: Remove All AI / Claude Fingerprints

**This section takes precedence over everything else.** Before the repo goes public, every trace of AI tooling must be scrubbed. A reviewer or colleague should not be able to tell that any AI assistant was used.

### 0a. Files to Delete or Exclude from Git Entirely

| File | Why |
|------|-----|
| `CLAUDE.md` | AI assistant config file. **Untrack from git** (`git rm --cached`) **and add to `.gitignore`.** Keep the file locally — it just shouldn't be in the public repo. |
| `memory/feedback_crop_flag.md` | Claude memory system file, tracked in git. **Untrack and gitignore the entire `memory/` directory.** |
| `CLEANUP.md` (this file) | References Claude throughout. **Do not commit this file.** Use it as a working document, then delete or gitignore before pushing. |

### 0b. Git Commit History — Co-Author Tags

Two commits contain `Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>` in their message body:

```
a1dcca4  snapshot before ALIFE cuts: ...
b785c22  version 0.6: FFT default, ...
```

These **must** be rewritten. During the `git filter-repo` pass (Section 4), also use `--message-callback` to strip co-author lines:

```bash
git filter-repo --message-callback '
    import re
    return re.sub(rb"\n\s*Co-Authored-By:.*anthropic.*\n?", b"\n", message, flags=re.IGNORECASE)
'
```

Or amend individually before the filter-repo pass. Either way, no commit message should reference Claude, Anthropic, or AI co-authorship.

### 0c. File Content — Explicit AI Mentions

| File | Line | Issue | Fix |
|------|------|-------|-----|
| `notes/Research Review - State Space Search.md` | 4 | `*Generated 2026-02-05 — Claude research review*` | Remove the italic attribution line entirely, or rewrite as `*Internal review, 2026-02-05*` |

**Grep to verify nothing else slipped through:**
```bash
git grep -i 'claude\|anthropic\|chatgpt\|copilot\|co-authored\|ai.generated\|ai.assisted\|llm'
```

### 0d. Comment & Docstring Style — Remove AI-Sounding Patterns

AI-written code often has a recognizable style. Agents performing the cleanup should scan for and fix these patterns across **all `.py` files**:

1. **Overly narrated docstrings.** A module docstring should say what the file *does* in one line, maybe two. It should not explain the motivation, how the pieces fit together, or narrate what "we" are doing. Compare:
   - Bad: `"""Shared configuration defaults for training and evaluation.\n\nThis ensures training and eval use matching parameters by default.\nOverride via command-line arguments as needed."""`
   - Good: `"""Default constants for grid size, timing, and curriculum stages."""`

2. **Hand-holding inline comments.** Comments that restate what the code already says, or explain obvious stdlib/library behavior. Look for:
   - `# Create the parser` before `parser = argparse.ArgumentParser()`
   - `# Load the config` before `cfg = Config.from_file(path)`
   - `# Iterate over all creatures` before `for code in codes:`
   - `# Return the result` before `return result`
   Comments should explain *why*, not *what*. Delete any comment that adds no information beyond what the code itself says.

3. **Decorative section separators.** Lines like `# ---------------------------------------------------------------------------` used as section breaks inside functions or between code blocks. These are fine between top-level class/function groups if the file is very long, but inside functions they are noise. Found in:
   - `environments/environments.py:414-416`

4. **Formulaic file headers.** Every `.py` file should have a docstring, but it should be a terse, human-sounding description — not a paragraph. The current codebase is mostly OK on this, but review each file header during cleanup. Examples of good headers:
   - `"""Core Lenia environment primitives and kernel utilities."""` (lenia.py — fine as-is)
   - `"""Barrier environments for constrained Lenia simulations."""` (environments.py — trim the multi-line explanation)
   - `"""Exhaustive perturbation sweep over all grid positions."""` (sweep.py — trim)

5. **Unnecessary type annotations on obvious local variables.** Things like `result: List[int] = []` when the type is immediately clear from context. Keep annotations on function signatures and module-level constants; remove obvious local ones.

6. **"Helpful" error messages that over-explain.** Messages like `raise ValueError("grid_size must be positive because negative grid sizes are not physically meaningful")`. Just `raise ValueError(f"grid_size must be > 0, got {grid_size}")`.

### 0e. Git Exclude Updates for AI Files

Add to `.git/info/exclude` (local-only, not committed):
```gitignore
# AI assistant config
CLAUDE.md
memory/
.claude/
```

### 0f. Verification Checklist

Before considering this section complete, an agent must confirm:

- [ ] `git grep -i 'claude\|anthropic'` returns zero results (excluding `.gitignore` lines)
- [ ] `git log --all --format='%B' | grep -i 'claude\|anthropic\|co-authored'` returns nothing
- [ ] No file named `CLAUDE.md` or `memory/` is tracked
- [ ] Spot-check 10 random `.py` files: headers are 1-2 lines, no narration
- [ ] No decorative `# ---` separators inside functions
- [ ] `notes/Research Review - State Space Search.md` has no AI attribution

---

## 1. What Belongs in Git (KEEP)

These are source code, config, and lightweight text. They stay tracked.

| Path | Why |
|------|-----|
| `substrate/` | Core Lenia engine (lenia.py, animals.py, simulation.py, scaling.py) |
| `metrics_and_machinery/` | Distance metrics, interventions, reward, trajectory metrics |
| `experiments/` | All `.py` analysis/sweep/plotting scripts |
| `figure_generation/` | `.py` scripts only (not outputs) |
| ~~`excitation_tests/`~~ | ~~`.py` scripts only~~ — **gitignored, see Section 3** |
| `environments/__init__.py`, `environments.py`, `make_envs.py`, `env_previews.py` | Environment logic |
| `orbits/__init__.py`, `orbits.py`, `orbit_batch.py` | Orbit pipeline code |
| `initializations/calibrate_headings.py`, `generate_*.py`, `verify_*.py`, `heading_offsets.json` | Init scripts + heading config |
| `utils/`, `viz/`, `tests/` | Utilities, visualization code, tests |
| `scripts/` | One-off helper scripts (all `.py` and `.sh`) |
| `cluster/job.slurm` | Cluster job template |
| `config.py`, `run.py` | Root config and entry point |
| `animals.json`, `animals_to_run.json`, `environments.json` | Creature/env catalogs |
| `README.md` | Documentation (CLAUDE.md is excluded — see Section 0) |
| ~~`notes/*.md`~~ | ~~Research notes~~ — **gitignored, see Section 3** |
| `.gitignore` | Ignore rules |

## 2. What Must Be REMOVED from Git (currently tracked, shouldn't be)

### 2a. Large Binary Data — Orbits (~1.5 GB tracked)

Every `.pt` file under `orbits/` is tracked. These are derived data (reproducible from code + `animals.json`). **Remove all of them.**

Files to untrack:
```
orbits/K4s/s4/K4s_s4_distances.pt    (127 MB)
orbits/K4s/s4/K4s_s4_orbit.pt        (28 KB)
orbits/K4s/s4/K4s_s4_profile.pt      (128 MB)
orbits/K6s/s4/K6s_s4_distances.pt    (127 MB)
orbits/K6s/s4/K6s_s4_orbit.pt        (36 KB)
orbits/K6s/s4/K6s_s4_profile.pt      (173 MB)
orbits/O2u/s4/O2u_s4_distances.pt    (127 MB)
orbits/O2u/s4/O2u_s4_orbit.pt        (16 KB)
orbits/O2u/s4/O2u_s4_profile.pt      (74 MB)
orbits/O2v/s4/O2v_s4_distances.pt    (127 MB)
orbits/O2v/s4/O2v_s4_orbit.pt        (16 KB)
orbits/O2v/s4/O2v_s4_profile.pt      (72 MB)
orbits/P4al/s4/P4al_s4_distances.pt  (127 MB)
orbits/P4al/s4/P4al_s4_orbit.pt      (56 KB)
orbits/P4al/s4/P4al_s4_profile.pt    (282 MB)
orbits/S1s/s4/S1s_s4_distances.pt    (127 MB)
orbits/S1s/s4/S1s_s4_orbit.pt        (20 KB)
orbits/S1s/s4/S1s_s4_profile.pt      (86 MB)
```

Also untrack the orbit `.json` sidecars (tiny, but derived):
```
orbits/*/s4/*_orbit.json   (6 files, <1 KB each)
```

### 2b. Large Binary Data — Initializations (~557 MB tracked)

All `.pt` files under `initializations/` are tracked. These are generated by `generate_initializations.py` + `generate_all_orientations.py`. **Remove all of them.**

Files to untrack:
```
initializations/K4s/s4/K4s_s4_all_orientations.pt   (90 MB)
initializations/K4s/s4/K4s_s4_o0..o3.pt             (4x ~1 MB)
initializations/K6s/s4/K6s_s4_all_orientations.pt   (90 MB)
initializations/K6s/s4/K6s_s4_o0..o3.pt             (4x ~1 MB)
initializations/O2u/s4/O2u_s4_all_orientations.pt   (51 MB)
initializations/O2u/s4/O2u_s4_o0..o3.pt + o120.pt   (5x ~1 MB)
initializations/O2v/s4/O2v_s4_all_orientations.pt   (90 MB)
initializations/O2v/s4/O2v_s4_o0..o3.pt             (4x ~1 MB)
initializations/P4al/s4/P4al_s4_all_orientations.pt (90 MB)
initializations/P4al/s4/P4al_s4_o0..o3.pt           (4x ~1 MB)
initializations/S1s/s4/S1s_s4_all_orientations.pt   (23 MB)
initializations/S1s/s4/S1s_s4_o0..o3.pt             (4x ~1 MB)
```

### 2c. Binary Design Asset (1.1 MB)

```
figure_generation/Dynamical-System.ai   (1.1 MB Adobe Illustrator file)
```

This is a design asset, not code. Remove from git.

### 2d. FINDINGS — .npy Arrays (tracked via `!results/FINDINGS/` exception)

```
results/FINDINGS/nice_orange_example/analysis/*.npy   (11 files, ~680 KB total)
```

These are derived analysis outputs. The markdown file `blindness_architecture_review.md` is fine to keep (text). The `.npy` files should be removed.

### 2e. Dispatch Scripts (cluster-specific, not portable)

```
dispatch                        (main dispatch wrapper)
dispatch_additive_sweeps.sh
dispatch_all_orientations.sh
dispatch_all_sweeps.sh
dispatch_env_competency.sh
dispatch_env_sweeps.sh
dispatch_initializations.sh
dispatch_sweep_x4.sh
```

These are local/cluster convenience scripts with hardcoded paths and job configs. Untrack and gitignore (`dispatch*`). Keep locally.

### 2f. Stale Root-Level Files

```
blind_env_thoughts.md   (520 bytes — scratch notes in root, not in notes/)
```

Either move to `notes/` or delete. Should not be a root-level tracked file.

## 3. .gitignore Updates Needed

The current `.gitignore` has explicit allowances that caused the bloat. After removing the files above, update `.gitignore`:

```gitignore
# REMOVE these lines (they override the *.pt block):
# !initializations/**/*.pt
# !orbits/**/*.pt

# FIX: change backups/* to backups/ (ignore the directory, not just contents):
# backups/*  →  backups/

# ADD these lines:
orbits/**/*.pt
orbits/**/*.json
initializations/**/*.pt
*.ai
excitation_tests/

# KEEP the existing FINDINGS exception but tighten it:
# Change: !results/FINDINGS/
# To:
!results/FINDINGS/**/*.md
```

This ensures only markdown findings survive, not `.npy` data.

## 3b. .git/info/exclude Updates (local-only, never committed)

These are personal/workflow files that shouldn't be in `.gitignore` (which ships with the repo) but should be invisible to git on this machine. Add all of these to `.git/info/exclude`:

```gitignore
# AI assistant config
CLAUDE.md
CLEANUP.md
memory/
.claude/

# Cluster dispatch (local convenience scripts, hardcoded paths)
dispatch
dispatch_*.sh

# Research notes (personal, some have AI attribution)
notes/
```

## 4. Git History Rewrite (REQUIRED)

Simply untracking files only affects future commits. The `.git/` directory is already **1.2 GB** because of historical blobs. After untracking, you must rewrite history to actually shrink the repo.

**Tool:** `git filter-repo` (preferred) or `BFG Repo Cleaner`.

```bash
# Install if needed
brew install git-filter-repo

# Remove large files from all history
git filter-repo --path-glob 'orbits/**/*.pt' --invert-paths
git filter-repo --path-glob 'orbits/**/*.json' --invert-paths
git filter-repo --path-glob 'initializations/**/*.pt' --invert-paths
git filter-repo --path 'figure_generation/Dynamical-System.ai' --invert-paths
git filter-repo --path-glob 'results/FINDINGS/**/*.npy' --invert-paths
```

Or in one pass (preferred — also strips AI co-author tags from commit messages):
```bash
git filter-repo \
  --path-glob 'orbits/**/*.pt' \
  --path-glob 'orbits/**/*.json' \
  --path-glob 'initializations/**/*.pt' \
  --path 'figure_generation/Dynamical-System.ai' \
  --path-glob 'results/FINDINGS/**/*.npy' \
  --path 'CLAUDE.md' \
  --path-glob 'memory/*' \
  --invert-paths \
  --message-callback '
import re
return re.sub(rb"\n\s*Co-Authored-By:.*\n?", b"\n", message, flags=re.IGNORECASE).rstrip() + b"\n"
'
```

**WARNING:** This rewrites all commit SHAs. The `pre-ALIFE-cuts` tag will be rewritten too. Coordinate with anyone who has cloned the repo — they will need to re-clone.

## 5. Backup Plan (BEFORE any removal)

### What to Back Up

The following data is **not reproducible from code alone** or is expensive to regenerate. Archive it before removing from git.

| Data | Size | Reproducible? | Back up? |
|------|------|---------------|----------|
| `orbits/**/*.pt` + `.json` | ~1.5 GB | Yes (expensive GPU hours) | **YES** |
| `initializations/**/*.pt` | ~557 MB | Yes (expensive GPU hours) | **YES** |
| `figure_generation/Dynamical-System.ai` | 1.1 MB | No (hand-made) | **YES** |
| `results/FINDINGS/**/*.npy` | ~680 KB | Yes (from sweep re-runs) | **YES** |
| `results/sweep/` | ~595 MB | Yes (expensive) | Already gitignored, verify backup exists |
| `results/env_competency/` | ~623 MB | Yes (expensive) | Already gitignored, verify backup exists |
| `results/new/` | ~12 GB | Yes (expensive) | Already gitignored, verify backup exists |
| `backups/` | ~4.1 GB | N/A (are backups) | Already exists |

### Backup Command (Keka / 7z)

Use Keka (installed at `/Applications/Keka.app`) to create a 7z archive. From the command line, Keka's CLI helper can be invoked, or use the GUI drag-and-drop.

**Recommended archive:** Create one archive of everything being removed from git:

```bash
# Collect all files to archive into a staging list
find orbits -name '*.pt' -o -name '*.json' > /tmp/backup_list.txt
find initializations -name '*.pt' >> /tmp/backup_list.txt
echo "figure_generation/Dynamical-System.ai" >> /tmp/backup_list.txt
find results/FINDINGS -name '*.npy' >> /tmp/backup_list.txt

# Use Keka's command-line helper (if available), or use p7zip:
#   brew install p7zip
#   7z a -t7z -m0=lzma2 -mx=9 backups/pre-ALIFE-git-purge.7z @/tmp/backup_list.txt

# Or: open Keka, drag the orbits/ and initializations/ folders into the window,
# select 7z format, and save as backups/pre-ALIFE-git-purge.7z
```

**Verify the archive extracts correctly before proceeding with any deletion or history rewrite.**

Also verify that your existing `backups/` already covers these:
- `backups/orbits.7z` — likely covers `orbits/`
- `backups/initializations.7z` — likely covers `initializations/`

If those are current, you may only need to archive `Dynamical-System.ai` and the FINDINGS `.npy` files.

## 6. Documentation Updates Needed

### README.md

The README references directories/concepts that are stale or missing:

1. **`learning/` directory** — Listed in the layout table but does not exist on disk. **Remove from the table.**
2. **`experiments/` description** — Says "orbits.py (orbit pipeline)" but orbits live in `orbits/`. Update to reflect current script list.
3. **Orbit paths** — Examples say `results/orbits/` but actual data lives in `orbits/` (top-level). Update all orbit path examples.
4. **Missing directories from table:**
   - `orbits/` — orbit pipeline code and data
   - `figure_generation/` — ALIFE figure scripts
   - `excitation_tests/` — excitation experiment scripts
   - `environments/` — barrier environment definitions + generation
   - `initializations/` — pre-settled creature tensors + generation scripts
   - `notes/` — research notes
   - `scripts/` — one-off helper scripts
   - `cluster/` — SLURM job template
5. **Sweep output naming** — The examples use the old `results/grid_search/` convention. Update to show the current `results/sweep/{CODE}/{CODE}_x{SCALE}_i{SIZE}` pattern.
6. **Dispatch flags** — Missing `--crop` and `--preempt` which are now standard (per project feedback). Add them to examples.
7. **`--init` flag** — Sweeps now support `--init` for pre-settled initializations. Document this.
8. **Recovery lambda** — K4s requires `--recovery-lambda 2.0`. Note this in the sweep section.

### CLAUDE.md

**Untrack and gitignore** (see Section 0a). Just `git rm --cached CLAUDE.md` and ensure `.gitignore` has the entry. The file stays local for continued use, it just won't be in the public repo.

### .gitignore

See Section 3 above. The allowances for `!initializations/**/*.pt` and `!orbits/**/*.pt` are the root cause of the bloat.

## 7. Execution Order

Agents should execute this cleanup in the following order:

1. **Scrub AI fingerprints (Section 0)** — This is the highest priority.
   - `git rm --cached CLAUDE.md memory/feedback_crop_flag.md`
   - Remove AI attribution from `notes/Research Review - State Space Search.md`
   - Audit and trim all `.py` docstrings/comments per Section 0d guidelines
   - Add `CLAUDE.md`, `memory/`, `.claude/` to `.git/info/exclude`
2. **Verify backups** — Confirm `backups/orbits.7z` and `backups/initializations.7z` are current. If not, create the 7z archive (Section 5).
3. **Untrack binary files** — `git rm --cached` for everything in Section 2.
4. **Update .gitignore** — Apply changes from Section 3. **Update `.git/info/exclude`** — Apply Section 3b (dispatch, notes, AI files).
5. **Update README.md** — Apply changes from Section 6. Do not reference AI tools.
6. **Move `blind_env_thoughts.md`** — To `notes/` or delete if content is already captured elsewhere.
7. **Commit** — Single commit with a human-sounding message, e.g.: "clean repo for publication: strip binaries, tighten gitignore, update docs"
   - **No `Co-Authored-By` trailer.** No mention of Claude or AI in the message.
8. **History rewrite** — Run `git filter-repo` (Section 4). This removes both the large blobs AND the co-author trailers from historical commits. **Required before any GitHub push.**
9. **Verify** — Run the full checklist from Section 0f, plus:
   - `du -sh .git/` should drop from 1.2 GB to ~10-20 MB
   - `git log --all --format='%B' | grep -ci 'claude\|anthropic'` returns `0`

## 8. Summary of Savings

| Item | Current Size in Git |
|------|-------------------|
| `orbits/**/*.pt` | ~1,500 MB |
| `initializations/**/*.pt` | ~557 MB |
| `Dynamical-System.ai` | 1.1 MB |
| `FINDINGS/*.npy` | 0.7 MB |
| **Total removed** | **~2,059 MB** |
| `.git/` after filter-repo | ~10-20 MB (down from 1.2 GB) |
