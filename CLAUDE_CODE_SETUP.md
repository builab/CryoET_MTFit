# Getting Started with Claude Code

Setup guide for installing and configuring Claude Code on the CryoET_MTFit repo.
Using the MTFit tool itself is covered separately.

## 1. Prerequisites

- **VS Code** installed.
- A **Claude account** with access to Claude Code — either a Claude.ai
  subscription that includes Code access, or an Anthropic Console/API account
  with billing set up.

## 2. Install the extension

1. Open VS Code.
2. Go to the Extensions panel (the icon in the left sidebar, or `Cmd+Shift+X`
   on Mac).
3. Search for **"Claude Code"** and install the official extension from
   Anthropic.
4. Once installed, you'll be prompted to sign in — this opens a browser window
   tied to your Claude account. Approve it there.

## 3. Open this project

1. In VS Code: **File → Open Folder…** and select the `CryoET_MTFit` folder.
2. Open the Claude Code panel (there's a sidebar icon for it once the
   extension is installed, or use the command palette: `Cmd+Shift+P` →
   "Claude Code: Open").
3. You now have a chat interface scoped to this repo — Claude can read every
   file in it, run terminal commands, and edit code directly.

## 4. Permission modes — decide this up front

By default, Claude asks before running commands or editing files, and you
approve or deny each one. There's a setting that controls how much gets
auto-approved instead of asked about every time.

Worth deciding deliberately for this project specifically, since a lot of the
actual work here involves:
- Rebuilding and reinstalling the ChimeraX bundle (`devel build` /
  `devel install` commands)
- Running Python scripts against real `.star` files
- Occasionally editing shell config files or running `git` commands

A reasonable default: let it auto-run read-only things (reading files,
running scripts to inspect data) without asking every time, but keep
confirmation on for anything that pushes to GitHub, deletes files, or touches
things outside this repo. You can adjust this at any time from the settings
menu in the Claude Code panel.

## 5. How the interaction actually works

It's a chat, not a fixed menu:

- Describe what you want in plain language — a bug, a feature, "explain what
  this function does," "why is X happening."
- Claude reads the relevant code, explains its plan for anything nontrivial,
  and makes the change directly (you'll see a diff before/as it happens).
- You can interrupt or redirect at any point — you don't have to wait for it
  to finish before saying "actually, do it this other way."
- For anything ambiguous or where there's a real design choice to make (not
  just a bug fix), it should ask you rather than guess — if it doesn't and
  you notice it guessed wrong, just say so.

## 6. Prompts that work well (and don't waste tokens)

**Making a small, scoped change:**
> "In `tool.py`, change just X. Don't touch how Y or Z work."

Name the file and what should stay untouched up front. Saves a round trip —
Claude won't guess how far to extend the change, and you won't have to
correct scope creep afterward.

**Protecting existing behavior:**
> "This should only affect [feature]. If it would change how [other feature]
> works, stop and ask me first."

The single most useful pattern for this repo specifically — Fit, Clean,
Connect, Predict, and Twist all share the same underlying files, so it's easy
for a "small" fix in one to quietly touch another. Say the blast radius you
expect out loud.

**Not sure a change is safe:**
> "Before you change this, tell me what else depends on it."

Cheaper than finding out afterward that a small fix touched five other
things.

**You already know the fix, just want it done:**
> "Yes, go ahead."

Once you've discussed and confirmed an approach, a short go-ahead is enough —
no need to restate the plan.

**Something looks off, not sure why:**
> "This doesn't look right — what would you check first?"

Lets Claude investigate before proposing a fix, instead of guessing at a fix
for a problem it hasn't looked at yet.

**General efficiency:**
- Point at a specific file/function if you know it, instead of describing
  only the symptom — saves the time spent re-finding it.
- Continuing something from an earlier session: a one-line reminder is
  enough ("continuing the Twist polarity fix from before") — memory already
  has the details, no need to re-explain.
- Open-ended questions ("what should we do about X?") get a short
  recommendation back, not an essay. Ask explicitly if you want the full
  analysis.

## 7. Persistent memory

Claude automatically keeps notes across sessions in this project — things
like decisions that were made, feedback on how you like things done, and
outstanding TODOs. This means you don't have to re-explain context every time
you open a new chat. If something it "remembers" turns out to be stale or
wrong (e.g. a file got renamed, a decision changed), just correct it and it
updates its notes.

## 8. What's next

Once you're comfortable with the basics above, see:

- **`CLAUDE.md`** (repo root) — project-specific operating notes: how to
  rebuild/reinstall the ChimeraX bundle, environment gotchas, where project
  history is tracked. Claude reads this automatically every session; you can
  read it too as a reference.
- **The MTFit interface guide** *(coming soon)* — how to actually use the
  ChimeraX plugin itself, independent of Claude Code.
