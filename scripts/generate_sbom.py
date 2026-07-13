#!/usr/bin/env python3
"""Generate a lightweight CycloneDX-compatible JSON dependency inventory."""

from __future__ import annotations

import argparse
import json
import sys
from importlib import metadata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

def build_sbom() -> dict:
    from app.version import __version__

    components = []
    for distribution in sorted(metadata.distributions(), key=lambda item: (item.metadata.get("Name") or "").lower()):
        name = distribution.metadata.get("Name")
        if not name:
            continue
        license_name = distribution.metadata.get("License") or "UNKNOWN"
        components.append(
            {
                "type": "library",
                "name": name,
                "version": distribution.version,
                "licenses": [{"license": {"name": license_name}}],
                "purl": f"pkg:pypi/{name.lower().replace('_', '-')}@{distribution.version}",
            }
        )
    return {
        "bomFormat": "CycloneDX",
        "specVersion": "1.5",
        "version": 1,
        "metadata": {"component": {"type": "application", "name": "agu-basketball", "version": __version__}},
        "components": components,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(build_sbom(), ensure_ascii=False, indent=2), encoding="utf-8")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
