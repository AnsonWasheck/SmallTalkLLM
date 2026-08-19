#!/usr/bin/env python3
"""Verify trainable parameter counts for every experimental configuration.

Exit code 1 if any config disagrees with its documented target (report, don't patch).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from smalltalk.config import CONFIG_DIR, ModelConfig, all_model_configs, load_model_config
from smalltalk.params import (
    analytic_breakdown,
    check_config,
    deployed_bytes,
    fits_budget,
    max_params_for_budget,
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("configs", nargs="*", help="config paths or names; default = all")
    ap.add_argument("--breakdown", action="store_true", help="print per-component costs")
    ap.add_argument("--json", dest="as_json", action="store_true")
    ap.add_argument("--no-empirical", action="store_true", help="formula only (fast)")
    ap.add_argument("--bytes", dest="show_bytes", action="store_true",
                    help="report shipped artifact size at each precision")
    ap.add_argument("--budget-mb", type=float, default=None,
                    help="flag configs that exceed this deployed size (e.g. 4)")
    ap.add_argument("--precision", default="int8",
                    help="precision used for the --budget-mb check")
    args = ap.parse_args()

    cfgs: list[ModelConfig]
    if args.configs:
        cfgs = [load_model_config(c) for c in args.configs]
    else:
        cfgs = all_model_configs()
        if not cfgs:
            print(f"no configs found in {CONFIG_DIR / 'model'}", file=sys.stderr)
            return 1
    cfgs.sort(key=lambda c: analytic_breakdown(c).total)

    checks = [check_config(c, empirical=not args.no_empirical) for c in cfgs]

    if args.as_json:
        print(
            json.dumps(
                [
                    {
                        "name": c.name,
                        "params": c.empirical,
                        "analytic": c.analytic,
                        "expected": c.expected,
                        "ok": c.ok,
                    }
                    for c in checks
                ],
                indent=2,
            )
        )
    else:
        print("=" * 72)
        print("smalltalk-ai parameter verification")
        print("=" * 72)
        for cfg, chk in zip(cfgs, checks):
            print(chk.report())
            if args.breakdown:
                for label, n in analytic_breakdown(cfg).as_rows():
                    print(f"{'':>18}{label:<24} {n:>12,}")
            print("-" * 72)
        bad = [c.name for c in checks if not c.ok]
        print("ALL CONFIGS MATCH TARGETS" if not bad else f"OFF-TARGET: {', '.join(bad)}")

        if args.show_bytes or args.budget_mb:
            print("\nShipped artifact size (weights + tokenizer + metadata, no optimizer state)")
            precisions = ["fp32", "bf16", "int8", "int4"]
            print(f"{'config':>16} {'params':>11} " + "".join(f"{p:>10}" for p in precisions))
            for cfg in cfgs:
                row = f"{cfg.name:>16} {analytic_breakdown(cfg).total:>11,} "
                for p in precisions:
                    row += f"{deployed_bytes(cfg, p) / 1048576:>9.2f}M"
                print(row)
            if args.budget_mb:
                lim = max_params_for_budget(args.budget_mb, args.precision)
                print(f"\nBudget {args.budget_mb:g} MB at {args.precision}: "
                      f"room for {lim:,} params")
                for cfg in cfgs:
                    ok = fits_budget(cfg, args.budget_mb, args.precision)
                    size = deployed_bytes(cfg, args.precision) / 1048576
                    print(f"  {'FITS  ' if ok else 'OVER  '}{cfg.name:>16} {size:6.2f} MB")
    return 0 if all(c.ok for c in checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
