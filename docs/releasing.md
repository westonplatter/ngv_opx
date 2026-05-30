# Releasing

This project ships four artifacts from one repository:

| Artifact         | Registry  | Source dir        |
| ---------------- | --------- | ----------------- |
| `ngv-opx-core`   | crates.io | `crates/core`     |
| `ngv-opx-gpu`    | crates.io | `crates/gpu`      |
| `ngv-opx` (PyPI) | PyPI      | `bindings/python` |
| `@ngv/opx`       | npm       | `bindings/wasm`   |

There are **three release lanes**. Pick the smallest one that fits the change.

---

## Lane 1 — Core release (the default)

Use this whenever the Rust core/API changes, or when you want everything bumped
together. `release-please` owns this lane and keeps all manifests in lockstep.

```bash
# 1. Merge normal conventional commits (feat:, fix:, perf:, ...) to main.
# 2. release-please opens/updates a "Release PR" with the next version + CHANGELOG.
# 3. Merge the Release PR.
# 4. release-please tags vX.Y.Z and the trigger-publish job dispatches:
#       publish-crate.yml   → crates.io (core + gpu)
#       publish-python.yml  → PyPI
#       publish-npm.yml     → npm
```

The `vX.Y.Z` tag publishes **all four artifacts at the same version**.

> Note: tags created by `release-please` use `GITHUB_TOKEN`, and GitHub does not
> fire `on: push` workflows for those tags (loop prevention). That is why
> `release-please.yml` dispatches the publish workflows explicitly via
> `gh workflow run --ref <tag>` rather than relying on the tag-push trigger.

---

## Lane 2 — Python package-only release

Use this for binding-only changes that do **not** need a core release: wheel
metadata, packaging fixes, type stubs, docstrings, README.

```bash
# 1. Open a PR bumping the Python binding to the new version, e.g. 0.1.2:
#      bindings/python/pyproject.toml   project.version  = "0.1.2"
#      bindings/python/Cargo.toml       package.version  = "0.1.2"
#    (Leave crates/* and bindings/wasm untouched.)
# 2. Merge after tests pass.
# 3. Actions → release-python-package → Run workflow:
#      version = 0.1.2
#      dry_run = true     # validate + build first
# 4. Re-run with dry_run = false to tag python-v0.1.2 and publish to PyPI only.
```

Locally validate the bump before opening the PR:

```bash
python3 scripts/check-release-versions.py --package python --version 0.1.2
```

This lane publishes **only PyPI**. crates.io and npm are untouched.

---

## Lane 3 — npm package-only release

Use this for binding-only changes that do **not** need a core release: JS init
behavior, exports, type definitions, README.

```bash
# 1. Open a PR bumping the wasm binding to the new version, e.g. 0.1.2:
#      bindings/wasm/package.json   version          = "0.1.2"
#      bindings/wasm/Cargo.toml     package.version  = "0.1.2"
#    (Leave crates/* and bindings/python untouched.)
# 2. Merge after tests pass.
# 3. Actions → release-npm-package → Run workflow:
#      version = 0.1.2
#      dry_run = true     # validate + build + node parity tests first
# 4. Re-run with dry_run = false to tag npm-v0.1.2 and publish to npm only.
```

Locally:

```bash
python3 scripts/check-release-versions.py --package npm --version 0.1.2
```

This lane publishes **only npm**. crates.io and PyPI are untouched.

---

## Versioning rules and the lockstep reconciliation

Tags are scoped by lane:

- Core: `vX.Y.Z`
- Python: `python-vX.Y.Z`
- npm: `npm-vX.Y.Z`

Package-only bumps are **patch increments by default**. A minor bump is fine for
a binding-only additive helper; a major bump for a breaking package layout or
init-contract change. These require human judgment in the bump PR.

### The one rule that keeps the lanes from colliding

`release-please` rewrites **every** binding manifest (`pyproject.toml`,
`package.json`, both binding `Cargo.toml`s) to the core version on each core
release — that is what keeps a core `vX.Y.Z` able to publish all four artifacts.
Because the package-only lanes share those same version lines, the lanes must
obey one invariant:

> **The next core release must be strictly greater than any package-only
> version already published.**

Concretely: if core is `0.1.1` and you ship Python `python-v0.1.2` and
`python-v0.1.3` independently, the next core release must be **≥ 0.1.4**. If you
let a core release land at `0.1.2` after Python already shipped `0.1.2`, the
PyPI publish step fails on a duplicate version.

Practical guidance:

- Keep package-only versions to small patch runs between core releases.
- When you next cut a core release, make sure it clears the highest package-only
  version already out (release-please will normally do this automatically since
  it bumps the shared counter; just don't hand-pick a lower number).
- `scripts/check-release-versions.py --package core --version X.Y.Z` confirms all
  manifests are aligned for a shared release.

---

## What each guard protects against

- `publish-crate.yml` accepts only `v*` tags → a `python-v*` / `npm-v*` tag can
  never republish the Rust crates.
- `publish-python.yml` accepts `v*` and `python-v*`; `publish-npm.yml` accepts
  `v*` and `npm-v*`. Each publish step is additionally gated on running from a
  tag ref.
- The package-only workflows refuse to tag if the scoped tag already exists or
  if the committed manifest versions do not match the requested version, and a
  `dry_run` exercises validation + build/test without tagging or publishing.

---

## Open questions (current answers)

- **Do package-only tags create GitHub Releases?** No. The annotated tag plus the
  PyPI/npm release page is sufficient at the current cadence. Revisit if we want
  changelogs surfaced on GitHub.
- **Where do package-only release notes live?** In the bump PR description for now;
  no automated changelog until volume justifies it.
- **Can package-only versions get ahead of core by more than a patch?** Allowed
  but discouraged — see the reconciliation rule above. Stay within the current
  core minor.
- **Should release-please manage multiple package entries instead?** Not yet. The
  manual lanes are simpler than a multi-component release-please config for the
  current cadence; revisit if binding-only releases become frequent.
