# Release Per Language Plan

Date: 2026-05-30

## Context

PR 15 changed the release flow so `release-please` remains the source of truth for the canonical project release. When a release is created, `.github/workflows/release-please.yml` explicitly dispatches:

- `publish-crate.yml`
- `publish-python.yml`
- `publish-npm.yml`

This should stay. The explicit dispatch is the right fix for GitHub's recursion behavior: tags created by `GITHUB_TOKEN` do not trigger normal tag workflows, while `workflow_dispatch` does.

The missing capability is package-only releases. The Rust core, Python package, and npm package currently move together through the `crates/core` release-please package and the shared `vX.Y.Z` tag. That is good for core/API releases, but too rigid for small binding-only improvements such as packaging fixes, README corrections, type definitions, JS init behavior, or Python wheel metadata.

## Goals

- Keep the PR 15 release behavior for canonical core releases.
- Allow Python and npm packages to publish patch-level increments independently.
- Make package-only releases intentional, auditable, and hard to trigger accidentally.
- Avoid publishing crates.io packages for binding-only changes.
- Keep version metadata consistent inside each binding package.

## Non-Goals

- Do not replace `release-please` for the main project release flow.
- Do not publish Python/npm packages from arbitrary commits without a reviewed version bump.
- Do not introduce prerelease or nightly package channels in this pass.
- Do not change the public package names: `ngv-opx` on PyPI and `@ngv/opx` on npm.

## Proposed Release Model

Use three release lanes:

1. Core release lane
   - Current PR 15 behavior.
   - Triggered by merging the release-please release PR.
   - Creates the normal shared `vX.Y.Z` tag.
   - Publishes crates.io, PyPI, and npm from the same tag.

2. Python package lane
   - Manual, package-only release.
   - Creates and publishes from a Python-specific tag, for example `python-v0.1.2`.
   - Publishes only `bindings/python`.
   - Bumps only Python package metadata:
     - `bindings/python/pyproject.toml` `project.version`
     - `bindings/python/Cargo.toml` `package.version`

3. npm package lane
   - Manual, package-only release.
   - Creates and publishes from an npm-specific tag, for example `npm-v0.1.2`.
   - Publishes only `bindings/wasm`.
   - Bumps only npm/wasm package metadata:
     - `bindings/wasm/package.json` `version`
     - `bindings/wasm/Cargo.toml` `package.version`

Package-only versions should remain semver-compatible with the underlying core they embed. For example, if core is `0.1.1`, a Python packaging fix can release `ngv-opx==0.1.2` without changing `ngv-opx-core`, as long as the package dependency remains path-based in-repo for build time and the published artifact embeds the same Rust code.

## Versioning Rules

Use the shared `vX.Y.Z` tag only for core releases.

Use scoped tags for language packages:

- Python: `python-vX.Y.Z`
- npm: `npm-vX.Y.Z`

Patch increments are the default for binding-only work:

- Python packaging fix: `0.1.1` to `0.1.2`
- npm README/types/init fix: `0.1.1` to `0.1.2`

Minor or major package-only increments should be allowed, but require human judgment. Examples:

- Minor: a new JS-only helper API that does not require Rust core changes.
- Major: a breaking package layout or initialization contract change.

## Workflow Changes

### 1. Keep `release-please.yml` As The Core Lane

Keep the PR 15 dispatch behavior:

- `release-please` runs on pushes to `main`.
- When it creates a release, `trigger-publish` dispatches all three publish workflows at the shared tag.
- The publish workflows keep their `startsWith(github.ref, 'refs/tags/')` publish guard.

Tighten comments and naming so this lane is clearly documented as the canonical core release path.

### 2. Add Explicit Package Release Workflows

Add two manual workflows:

- `.github/workflows/release-python-package.yml`
- `.github/workflows/release-npm-package.yml`

Each workflow should accept:

- `version`: required semver string, no leading `v`
- `dry_run`: optional boolean, default `true`

Each workflow should:

1. Validate that `version` is valid semver.
2. Validate that the target scoped tag does not already exist.
3. Validate that the current package manifest versions match `version`.
4. Run the package tests/build.
5. If `dry_run == false`, create the scoped tag on the selected commit.
6. Dispatch the existing package publish workflow using `gh workflow run ... --ref <scoped-tag>`.

This keeps version bumps reviewed through a normal PR, while the final publish remains a manual operator action.

### 3. Teach Existing Publish Workflows About Scoped Tags

Update publish triggers and publish guards:

Python:

- Current shared release tag: `v[0-9]+.[0-9]+.[0-9]+`
- New package-only tag: `python-v[0-9]+.[0-9]+.[0-9]+`

npm:

- Current shared release tag: `v[0-9]+.[0-9]+.[0-9]+`
- New package-only tag: `npm-v[0-9]+.[0-9]+.[0-9]+`

The publish jobs should continue to publish only when running on a tag, but should narrow the accepted tag pattern per workflow. For example:

- `publish-python.yml` publishes on `refs/tags/v*` or `refs/tags/python-v*`.
- `publish-npm.yml` publishes on `refs/tags/v*` or `refs/tags/npm-v*`.
- `publish-crate.yml` publishes only on `refs/tags/v*`.

This prevents a package-only tag from publishing the Rust crates.

### 4. Add A Version Consistency Check

Add a small script, for example `scripts/check-release-versions.py`, that can validate:

- Python package version equals the requested version:
  - `bindings/python/pyproject.toml`
  - `bindings/python/Cargo.toml`
- npm package version equals the requested version:
  - `bindings/wasm/package.json`
  - `bindings/wasm/Cargo.toml`
- Core release versions remain aligned for shared releases:
  - workspace version
  - core crate version
  - gpu crate version
  - Python binding Cargo dependency versions, when explicit
  - wasm binding Cargo dependency versions, when explicit

Use this script in:

- package release workflows before creating tags
- CI, if practical, for PR-time feedback

### 5. Document Operator Commands

Add a release docs page with concrete commands:

Python package-only release:

```bash
# 1. Open a PR bumping bindings/python versions to 0.1.2.
# 2. Merge after tests pass.
# 3. Run release-python-package.yml with:
#    version=0.1.2
#    dry_run=false
```

npm package-only release:

```bash
# 1. Open a PR bumping bindings/wasm versions to 0.1.2.
# 2. Merge after tests pass.
# 3. Run release-npm-package.yml with:
#    version=0.1.2
#    dry_run=false
```

Core release:

```bash
# 1. Merge normal conventional commits to main.
# 2. Let release-please open/update the release PR.
# 3. Merge the release PR.
# 4. release-please creates vX.Y.Z and dispatches crate, Python, and npm publish workflows.
```

## Implementation Steps

1. Restore or update the release workflow files from `main` in the working branch if needed.
2. Add scoped tag patterns to `publish-python.yml` and `publish-npm.yml`.
3. Keep `publish-crate.yml` restricted to shared `vX.Y.Z` tags.
4. Add `release-python-package.yml` with dry-run validation and tag creation.
5. Add `release-npm-package.yml` with dry-run validation and tag creation.
6. Add a version consistency script.
7. Add release documentation with operator examples.
8. Run workflow syntax checks and local version-check tests.
9. Test dry-run manual workflow runs from a branch before enabling real publishes.

## Acceptance Criteria

- Merging a release-please release PR still publishes all three artifacts from a shared `vX.Y.Z` release.
- Running the Python package release workflow with `version=0.1.2` creates `python-v0.1.2` and publishes only PyPI artifacts.
- Running the npm package release workflow with `version=0.1.2` creates `npm-v0.1.2` and publishes only npm artifacts.
- `python-v*` and `npm-v*` tags never trigger crates.io publishing.
- A package-only release fails before tag creation if manifest versions do not match the requested version.
- A dry run exercises validation/build/test steps without creating tags or publishing.

## Open Questions

- Should package-only tags also create GitHub Releases, or are tags plus PyPI/npm release pages sufficient?
- Should package-only release notes be maintained manually, generated from commit ranges, or skipped until volume justifies automation?
- Should package-only versions be allowed to get ahead of the core version by more than patch increments?
- Should `release-please` eventually manage multiple package entries, or is manual package-only release enough for the current cadence?
