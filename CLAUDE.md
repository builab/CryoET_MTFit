# CLAUDE.md

Operating notes for Claude Code in this repo. Read this before making changes —
it captures gotchas that otherwise get rediscovered the hard way every session.

## What this project is

`CryoET_MTFit` builds a ChimeraX plugin (`ChimeraX-MTFit`) for fitting and
refining microtubule/cilia particle picks in cryo-ET tomograms. Core pipeline:
**Fit → Clean → Connect → Predict/Twist**.

- Fit: initial curve fitting and clustering into tubes (assigns `rlnHelicalTubeID`)
- Clean: removes overlapping/duplicate tube fragments
- Connect: joins broken tube segments via trajectory extrapolation
- Predict: assigns angles via template matching — the right choice for **cilia**
- Twist: assigns a synthetic per-particle Rot ramp — the right choice for **MT**
  (microtubules are near-rotationally-symmetric, so Predict's per-particle Rot
  is unreliable for them; see "Angle conventions" below)

Code layout:
- `ChimeraX-MTFit/src/tool.py` — the GUI (ToolInstance, Qt panel)
- `ChimeraX-MTFit/src/mt_fit.py` — CLI entry point, one subcommand per step
- `ChimeraX-MTFit/src/utils/` — core logic (`fit.py`, `clean.py`, `connect.py`,
  `predict.py`, `sort.py`, `io.py`)
- `ChimeraX-MTFit/src/__init__.py` — bundle entry point, auto-installs deps

## Data model & flow

A `.star` file is a table of particles, one row per picked point along a
filament. The columns that matter:

- `rlnCoordinateX/Y/Z` — 3D position in the tomogram (pixels)
- `rlnAngleRot/Tilt/Psi` — RELION ZYZ Euler angles describing 3D orientation
- `rlnHelicalTubeID` — which tube/filament a particle belongs to
- `rlnTomoName`, `rlnImagePixelSize` — bookkeeping
- `rlnLCCmax` (raw picks only) — template-matching confidence score

How a file actually transforms as it moves through the pipeline:

1. **Raw picks** (from external template-matching software): scattered
   points with real — if sometimes noisy — 3D orientation from genuine
   cross-correlation search. No `rlnHelicalTubeID` yet.
2. **Fit**: clusters nearby points into tubes, fits a polynomial curve
   through each, and resamples at a fixed step size. This is where
   `rlnHelicalTubeID` first appears — and where fresh `rlnAngleTilt`/`Psi`
   get fabricated from pure coordinate-tangent geometry for every resampled
   point (`rlnAngleRot` set to 0), since resampled points don't correspond
   1:1 to the original raw picks anymore. `rlnLCCmax` and other picking-only
   columns are dropped here.
3. **Clean**: removes duplicate/overlapping tube fragments (compares tubes
   pairwise, deletes the shorter one where they overlap).
4. **Connect**: merges genuinely broken segments of the same tube, re-running
   Fit's own resample on just the merged tube.
5. **Predict**: maps real orientation from the raw picks (used as their own
   "template") onto the fitted/connected tube via nearest-neighbor averaging.
   For cilia, this is the final, trusted angle assignment. For MT, only the
   *polarity* signal from this step is trusted — Rot itself is not reliable
   for a near-rotationally-symmetric microtubule (see "Angle conventions").
6. **Twist** (MT only): adds a synthetic per-particle `rlnAngleRot` ramp on
   top of whatever's already there, modeling the real biological twist
   between protofilament dimers — optionally polarity-corrected via a
   disposable internal Predict pass (never saved, used only to decide the
   ramp's sign per tube).

The GUI's Manual Tube Join and the `join` CLI subcommand operate directly on
this same model: merging two `rlnHelicalTubeID` groups and re-running Fit's
resample on just the merged tube, leaving every other tube's rows untouched.

## Build & install workflow

After **any** change to `ChimeraX-MTFit/src/`, it must be rebuilt and
reinstalled into ChimeraX before it takes effect:

```bash
cd ChimeraX-MTFit
rm -rf build
/Applications/ChimeraX-*.app/Contents/MacOS/ChimeraX --nogui --exit --cmd "devel build . ; exit"
/Applications/ChimeraX-*.app/Contents/MacOS/ChimeraX --nogui --exit --cmd "devel install . ; exit"
```

Gotchas:
- **Always `rm -rf build` first.** A `git checkout` (or anything else that
  rewrites files without changing their content) can bump `build/lib`'s
  mtimes newer than `src/`, which makes `devel build`'s mtime-based check
  silently skip recompiling changed files. Deleting `build/` forces a full,
  correct rebuild every time.
- **Don't chain `devel build ; devel install` in one `--cmd` string.** They
  race on the same `build/bdist` directory and the second step can fail
  intermittently. Run them as two separate ChimeraX invocations.
- **ChimeraX must be restarted** after reinstalling. A running session keeps
  the old bundle code loaded in memory — rebuilding on disk does nothing for
  a session that's already open.
- Verify sync before testing: `diff -q src/tool.py build/lib/chimerax/mtfit/tool.py`
  (repeat for any other changed files) should print nothing.

## Environment

- **ChimeraX has its own bundled Python**, entirely separate from Anaconda,
  system Python, or this repo's `.venv`. It lives inside
  `ChimeraX-*.app/Contents/Library/Frameworks/Python.framework/...`, with a
  second site-packages directory at
  `~/Library/Application Support/ChimeraX/<version>/lib/python/site-packages/`
  where the bundle itself and its auto-installed dependencies live. ChimeraX
  also ships its own `scipy`/`matplotlib` for internal use, separate from
  what MTFit installs.
- Dependencies auto-install via `_ensure_dependencies()` in `__init__.py`,
  **one package at a time** (not a combined `pip install` call) — `copick`
  pulls in `cryptography`, which fails to build from source on machines
  without a Rust/OpenSSL toolchain, and used to silently block every other
  package in the same batch from installing too, including `scikit-learn`.
  If dependency-related errors resurface (e.g. `No module named 'sklearn'`),
  check this function first before assuming it's something else.
- This repo's own `.venv` was previously symlinked into Anaconda; it's now
  rebuilt against python.org's Python 3.11, independent of Anaconda.

## A stale duplicate to watch for

There are **two** `utils/` packages in this repo:
- `ChimeraX-MTFit/src/utils/` — the real, actively-used bundle code.
- `utils/` at the repo root — an older, stale duplicate used only by the
  standalone scripts in `scripts/`.

Depending on working directory and `PYTHONPATH`, it's easy to accidentally
import the wrong one when testing outside ChimeraX. If a fix doesn't seem to
take effect during manual testing, check which `utils/` actually got
imported before assuming the fix itself is wrong.

## Where project context lives

- **`Microtubule Picking.md`** (repo root) — the PI's running weekly log,
  newest entries at top. This is the primary source of truth for what's been
  decided, deferred, or is currently in flux. **It is intentionally not
  committed to git** (local notes only, per the user's explicit request) —
  don't add it to a commit unless asked.
- **`TODO.md`** (repo root, tracked in git) — an older, version-tagged
  changelog/TODO list. Less actively maintained than the weekly log but has
  useful historical context (e.g. known deferred items like microtubule
  polarity detection).
- Claude's own persistent memory (outside this repo) tracks session-to-session
  decisions, fixes, and PI feedback in more detail than either file above —
  check it for recent context before re-deriving something from scratch.

## Git conventions

- Only commit/push when explicitly asked.
- Stage specific files, not `git add -A` — this repo accumulates scratch
  test outputs (`.star` files, screenshots) in `example/` and elsewhere that
  should not be swept into commits.
- Keep `build/` and `dist/` in sync with `src/` in the same commit (they're
  tracked despite being build artifacts, so changes need `git add -f`).
- Write commit messages around the *why*, not a line-by-line diff summary.

## Testing workflow

Claude cannot click through the ChimeraX GUI or view the 3D render directly.
The standard loop for anything GUI/visual:

1. Make the fix, rebuild, reinstall.
2. Tell the user to restart ChimeraX and retest.
3. Ask for the actual output file, not just a description — comparing two
   `.star` files numerically is far more reliable than eyeballing a 3D
   render or trusting a verbal description of "it looks wrong."
4. Use `scripts/compare_rot_smoothness.py before.star after.star --angle
   <column>` for angle-smoothness comparisons (reports per-tube consecutive-
   particle jump, the actual wobble metric, plus a separate raw value-shift
   check — don't conflate the two).

When a bug report doesn't match the code's expected behavior, verify the
*actual installed* bundle matches `src/` before doing anything else — several
bugs this project has hit turned out to be a stale build, not a real code
issue.

## Angle/orientation conventions

Worth getting right if touching `predict.py`, `connect.py`, or `fit.py`:

- **`fit.py`'s `resample()`** computes `rlnAngleTilt`/`rlnAnglePsi` from pure
  coordinate-tangent geometry (which way consecutive fitted points move) —
  a self-invented scheme for fabricating plausible angles on freshly-generated
  points. This is **not** the same convention that genuine template-matching
  output follows.
- **Genuine RELION-convention angles** (e.g. raw template-matching picks):
  verified numerically against ArtiaX's actual `RELIONEulerRotation` code —
  `rlnAngleTilt`/`rlnAnglePsi` together fully determine which way a
  particle's long axis points; **`rlnAngleRot` has zero effect on
  direction** (confirmed by holding Tilt/Psi fixed and varying Rot across
  0/30/90/180 — identical resulting direction vector every time). Rot is
  purely the roll around that axis.
- Don't invert one convention's angles using the other's formula — they are
  not interchangeable, and mixing them produces confident-looking garbage
  (this happened twice while building the polarity-aware Twist feature;
  the tell was a suspiciously lopsided result, like 9-out-of-10 or 0-out-of-10
  tubes needing the same correction, instead of a plausible mixed split).
