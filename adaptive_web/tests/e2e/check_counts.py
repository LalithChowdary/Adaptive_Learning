#!/usr/bin/env python3
"""Enforce E2E 4-tier test count gates, F01..F14 coverage, and unique test names."""

from __future__ import annotations

import ast
from pathlib import Path
import re
import sys
import unittest

E2E_DIR = Path(__file__).resolve().parent
FEATURES = [f"F{i:02d}" for i in range(1, 15)]
N_FEATURES = len(FEATURES)

TIER_FILES = {
    "tier0": E2E_DIR / "test_tier0_infra_smoke.py",
    "tier1": E2E_DIR / "test_tier1_features.py",
    "tier2": E2E_DIR / "test_tier2_boundaries.py",
    "tier3": E2E_DIR / "test_tier3_pairwise.py",
    "tier4": E2E_DIR / "test_tier4_scenarios.py",
}


def extract_test_methods(path: Path) -> tuple[list[tuple[str, str]], dict[str, list[str]]]:
    """Return ([(class_name, method_name), ...], {feature_id: [method_name, ...]})."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    pairs: list[tuple[str, str]] = []
    by_feature: dict[str, list[str]] = {f: [] for f in FEATURES}

    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            m = re.match(r"^Test(F\d{2})", node.name)
            feat = m.group(1) if m else None
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name.startswith("test_"):
                    pairs.append((node.name, item.name))
                    if feat in by_feature:
                        by_feature[feat].append(item.name)
    return pairs, by_feature


def main() -> int:
    errors: list[str] = []
    all_method_names: list[tuple[str, str, str]] = []
    tier_counts: dict[str, int] = {}

    for tier, path in TIER_FILES.items():
        if not path.exists():
            errors.append(f"Missing tier file: {path}")
            tier_counts[tier] = 0
            continue
        pairs, by_feature = extract_test_methods(path)
        tier_counts[tier] = len(pairs)
        for cls_name, meth_name in pairs:
            all_method_names.append((path.name, cls_name, meth_name))

        if tier in ("tier1", "tier2"):
            for feat in FEATURES:
                cnt = len(by_feature[feat])
                if cnt < 5:
                    errors.append(f"{tier} {feat}: expected >= 5 tests, found {cnt}")

    # Threshold gates
    if tier_counts["tier0"] < 4:
        errors.append(f"tier0: expected >= 4 smoke tests, found {tier_counts['tier0']}")
    if tier_counts["tier1"] < 5 * N_FEATURES:
        errors.append(f"tier1: expected >= {5 * N_FEATURES} tests, found {tier_counts['tier1']}")
    if tier_counts["tier2"] < 5 * N_FEATURES:
        errors.append(f"tier2: expected >= {5 * N_FEATURES} tests, found {tier_counts['tier2']}")
    if tier_counts["tier3"] < N_FEATURES:
        errors.append(f"tier3: expected >= {N_FEATURES} tests, found {tier_counts['tier3']}")
    min_tier4 = max(5, N_FEATURES // 2)
    if tier_counts["tier4"] < min_tier4:
        errors.append(f"tier4: expected >= {min_tier4} tests, found {tier_counts['tier4']}")

    # Global unique test method name check
    seen: dict[str, str] = {}
    for fname, cls_name, meth_name in all_method_names:
        loc = f"{fname}:{cls_name}.{meth_name}"
        if meth_name in seen:
            errors.append(f"Duplicate test method name {meth_name!r}: {seen[meth_name]} vs {loc}")
        else:
            seen[meth_name] = loc

    # Verify unittest discover collects exact expected count
    loader = unittest.TestLoader()
    suite = loader.discover(str(E2E_DIR), top_level_dir=str(E2E_DIR))
    discovered = suite.countTestCases()
    expected_total = sum(tier_counts.values())
    if discovered != expected_total:
        errors.append(
            f"unittest discover count ({discovered}) != AST test count ({expected_total}); check loader errors"
        )

    print(
        f"E2E Count Audit (N={N_FEATURES}): "
        f"Tier0={tier_counts['tier0']} Tier1={tier_counts['tier1']} "
        f"Tier2={tier_counts['tier2']} Tier3={tier_counts['tier3']} "
        f"Tier4={tier_counts['tier4']} | Total={expected_total} Discovered={discovered}"
    )
    if errors:
        for err in errors:
            print(f"ERROR: {err}", file=sys.stderr)
        return 1
    print("OK: All E2E tier counts, F01..F14 minimums, and uniqueness gates passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
