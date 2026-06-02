#!/usr/bin/env python3
"""Validate version consistency across ngv-opx package manifests.

Used by the package-only release workflows before a scoped tag is created, and
available locally / in CI for PR-time feedback.

Modes:
  python  validate the Python binding manifests equal --version
  npm     validate the npm/wasm binding manifests equal --version
  core    validate all crates + bindings are aligned to --version (lockstep
          core release sanity check)

Without --version the script reports the current versions and verifies internal
consistency for the chosen mode (e.g. that pyproject and Cargo.toml agree).

Exit code 0 on success, 1 on any mismatch.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent

SEMVER = re.compile(r"^\d+\.\d+\.\d+$")

try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:  # pragma: no cover - fallback for older runners
    tomllib = None


def _load_toml(rel: str) -> dict:
    path = ROOT / rel
    if tomllib is not None:
        with open(path, "rb") as fh:
            return tomllib.load(fh)
    # Minimal fallback: only needs to find version-like keys via regex. We keep
    # this dependency-free so the script runs anywhere; tomllib is preferred.
    raise RuntimeError(
        "tomllib unavailable (need Python 3.11+); cannot parse %s" % rel
    )


def _load_json(rel: str) -> dict:
    with open(ROOT / rel) as fh:
        return json.load(fh)


def _dep_version(table: dict, name: str) -> str | None:
    deps = table.get("dependencies", {})
    entry = deps.get(name)
    if isinstance(entry, dict):
        return entry.get("version")
    return None


def _collect(label: str, actual, expected, errors: list[str]) -> None:
    if expected is not None and actual != expected:
        errors.append(f"  {label}: {actual!r} != expected {expected!r}")
    else:
        print(f"  {label}: {actual}")


def check_python(expected: str | None, errors: list[str]) -> None:
    print("Python binding versions:")
    py = _load_toml("bindings/python/pyproject.toml")
    cargo = _load_toml("bindings/python/Cargo.toml")
    pyver = py["project"]["version"]
    cargover = cargo["package"]["version"]
    _collect("bindings/python/pyproject.toml project.version", pyver, expected, errors)
    _collect("bindings/python/Cargo.toml package.version", cargover, expected, errors)
    # The two manifests must always agree with each other even if --version is
    # omitted (a package builds one artifact; one version).
    if pyver != cargover:
        errors.append(
            f"  pyproject ({pyver}) and Cargo.toml ({cargover}) disagree"
        )


def check_npm(expected: str | None, errors: list[str]) -> None:
    print("npm/wasm binding versions:")
    pkg = _load_json("bindings/wasm/package.json")
    cargo = _load_toml("bindings/wasm/Cargo.toml")
    jsver = pkg["version"]
    cargover = cargo["package"]["version"]
    _collect("bindings/wasm/package.json version", jsver, expected, errors)
    _collect("bindings/wasm/Cargo.toml package.version", cargover, expected, errors)
    if jsver != cargover:
        errors.append(
            f"  package.json ({jsver}) and Cargo.toml ({cargover}) disagree"
        )


def check_core(expected: str | None, errors: list[str]) -> None:
    print("Core / lockstep versions:")
    root = _load_toml("Cargo.toml")
    core = _load_toml("crates/core/Cargo.toml")
    gpu = _load_toml("crates/gpu/Cargo.toml")
    py = _load_toml("bindings/python/Cargo.toml")
    wasm = _load_toml("bindings/wasm/Cargo.toml")

    _collect("workspace.package.version", root["workspace"]["package"]["version"], expected, errors)
    _collect("crates/core package.version", core["package"]["version"], expected, errors)
    _collect("crates/gpu package.version", gpu["package"]["version"], expected, errors)
    _collect("crates/gpu dep ngv-opx-core", _dep_version(gpu, "ngv-opx-core"), expected, errors)
    _collect("bindings/python package.version", py["package"]["version"], expected, errors)
    _collect("bindings/python dep ngv-opx-core", _dep_version(py, "ngv-opx-core"), expected, errors)
    _collect("bindings/python dep ngv-opx-gpu", _dep_version(py, "ngv-opx-gpu"), expected, errors)
    _collect("bindings/wasm package.version", wasm["package"]["version"], expected, errors)
    _collect("bindings/wasm dep ngv-opx-core", _dep_version(wasm, "ngv-opx-core"), expected, errors)
    # Bindings that publish to PyPI / npm carry their own user-facing version too.
    check_python(expected, errors)
    check_npm(expected, errors)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--package", choices=["python", "npm", "core"], required=True)
    parser.add_argument(
        "--version",
        help="expected semver (no leading 'v'); if omitted, only internal consistency is checked",
    )
    args = parser.parse_args()

    if args.version is not None and not SEMVER.match(args.version):
        print(f"error: --version {args.version!r} is not a bare semver (X.Y.Z)", file=sys.stderr)
        return 2

    errors: list[str] = []
    if args.package == "python":
        check_python(args.version, errors)
    elif args.package == "npm":
        check_npm(args.version, errors)
    else:
        check_core(args.version, errors)

    if errors:
        print("\nVersion check FAILED:", file=sys.stderr)
        for err in errors:
            print(err, file=sys.stderr)
        return 1

    print("\nVersion check OK.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
