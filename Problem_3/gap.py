#!/usr/bin/env python3
"""gap.py

Run the gap experiment required by Problem 3:
Compare MILP (exact) vs LP relaxation (lower bound) for several perturbation radii.

IMPORTANT:
- This script does NOT modify any of the original assignment files.
- By default it imports:
    * MILP formulation from `mip_assignment.py` (module name: mip_assignment)
    * LP formulation from `lp.py` (module name: lp)
  You can override via CLI flags.

We run each formulation in a separate spawned process so we can enforce a per-run wall-clock timeout.
"""

import argparse
import csv
import importlib
import multiprocessing as mp
import os
import sys
import traceback
from typing import Any, Dict, List, Optional


S_LIST = [0.0, 0.001, 0.003, 0.01, 0.03, 0.1]


def _ensure_local_imports_work():
    """Make sure the directory containing this script is on sys.path.

    This is important when multiprocessing uses the 'spawn' start method.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)


def _worker(module_name: str, data_file: str, perturbation: float, queue: "mp.Queue"):
    """Run one full verification build + solve and return objective values.

    We do NOT call verify() because verify() prints extra things and (for MILP) may contain
    MIP-only asserts. Instead we reconstruct the same steps and return the raw optimal
    objective values.
    """
    try:
        _ensure_local_imports_work()

        # Silence the very verbose per-objective prints from solve_objectives().
        import contextlib
        import io

        mod = importlib.import_module(module_name)

        with contextlib.redirect_stdout(io.StringIO()):
            model, x_test, groundtruth_label = mod.load_model_and_data(data_file)
            v = mod.Verifier()

            variables = v.create_inputs(x_test, perturbation)
            weights, biases = mod.extract_model_weight(model)

            num_layers = len(weights)
            for i in range(num_layers):
                variables = list(v.add_linear_layer(variables, weights[i], biases[i]))
                if i != num_layers - 1:
                    lbs, _ = v.solve_objectives(variables, direction="minimization")
                    ubs, _ = v.solve_objectives(variables, direction="maximization")
                    variables = v.add_relu_layer(variables, lbs, ubs)

            objectives, target_labels = v.get_verification_objectives(variables, groundtruth_label)
            optimal_objs, _ = v.solve_objectives(objectives, direction="minimization")

        queue.put(
            {
                "ok": True,
                "optimal_objs": optimal_objs,
                "target_labels": target_labels,
                "groundtruth_label": int(groundtruth_label),
            }
        )

    except BaseException as e:
        queue.put(
            {
                "ok": False,
                "error": repr(e),
                "traceback": traceback.format_exc(),
            }
        )


def run_one(module_name: str, data_file: str, perturbation: float, time_limit: float) -> Dict[str, Any]:
    """Run one module (MILP or LP) with timeout.

    Returns a dict:
      - {ok: True, ...} on success
      - {ok: False, timeout: True} on timeout
      - {ok: False, error: ..., traceback: ...} on failure
    """
    ctx = mp.get_context("spawn")
    q: "mp.Queue" = ctx.Queue()
    p = ctx.Process(target=_worker, args=(module_name, data_file, perturbation, q), daemon=True)

    p.start()
    p.join(timeout=None if time_limit <= 0 else time_limit)

    if p.is_alive():
        p.terminate()
        p.join()
        return {"ok": False, "timeout": True}

    if q.empty():
        # Process exited without putting anything in the queue.
        return {"ok": False, "error": f"worker exited with code {p.exitcode} and produced no result"}

    return q.get()


def _fmt(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, str):
        return v
    try:
        return f"{float(v):.10g}"
    except Exception:
        return str(v)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_file", type=str, help="e.g. data1.pth")
    parser.add_argument("--time_limit", type=float, default=600.0, help="seconds per run (MILP or LP). 0 = no limit")
    parser.add_argument("--out_csv", type=str, default="gap_results.csv")
    parser.add_argument(
        "--milp_module",
        type=str,
        default="mip_assignment",
        help="Python module name for MILP (default: mip_assignment, i.e., mip_assignment.py)",
    )
    parser.add_argument(
        "--lp_module",
        type=str,
        default="lp",
        help="Python module name for LP (default: lp, i.e., lp.py)",
    )
    args = parser.parse_args()

    _ensure_local_imports_work()

    rows: List[Dict[str, Any]] = []

    print("Running gap experiment...")
    print(f"data_file = {args.data_file}")
    print(f"time_limit per run = {args.time_limit} sec")
    print(f"MILP module = {args.milp_module}")
    print(f"LP module   = {args.lp_module}")
    print(f"s list = {S_LIST}")
    print("")

    for s in S_LIST:
        print(f"s = {s}")

        milp_res = run_one(args.milp_module, args.data_file, s, args.time_limit)
        lp_res = run_one(args.lp_module, args.data_file, s, args.time_limit)

        # Handle hard errors early (better than silently writing garbage timeouts).
        if not milp_res.get("ok", False) and not milp_res.get("timeout", False) and "error" in milp_res:
            print("  MILP: ERROR")
            print(" ", milp_res["error"])
            if "traceback" in milp_res:
                print(milp_res["traceback"])
            raise SystemExit(1)

        if not lp_res.get("ok", False) and not lp_res.get("timeout", False) and "error" in lp_res:
            print("  LP: ERROR")
            print(" ", lp_res["error"])
            if "traceback" in lp_res:
                print(lp_res["traceback"])
            raise SystemExit(1)

        print("  MILP:", "done" if milp_res.get("ok", False) else "timeout")
        print("  LP:  ", "done" if lp_res.get("ok", False) else "timeout")

        # If both timed out, skip.
        if not milp_res.get("ok", False) and milp_res.get("timeout", False) and not lp_res.get("ok", False) and lp_res.get("timeout", False):
            print("  Both MILP and LP timed out; skipping detailed rows for this s.")
            print("")
            continue

        # Use whichever finished to fetch metadata.
        meta = milp_res if milp_res.get("ok", False) else lp_res
        target_labels = meta["target_labels"]
        c = meta["groundtruth_label"]

        milp_objs = milp_res.get("optimal_objs") if milp_res.get("ok", False) else None
        lp_objs = lp_res.get("optimal_objs") if lp_res.get("ok", False) else None

        per_s_rows: List[Dict[str, Any]] = []

        for i, r in enumerate(target_labels):
            if r == c:
                continue

            milp_val = None if milp_objs is None else milp_objs[i]
            lp_val = None if lp_objs is None else lp_objs[i]
            gap = None

            if milp_val is not None and lp_val is not None and not isinstance(milp_val, str) and not isinstance(lp_val, str):
                try:
                    gap = float(milp_val) - float(lp_val)
                except Exception:
                    gap = None

            row = {
                "s": s,
                "groundtruth": c,
                "target": r,
                "milp_min_yc_minus_yr": milp_val if milp_val is not None else "timeout",
                "lp_lower_bound_yc_minus_yr": lp_val if lp_val is not None else "timeout",
                "gap_milp_minus_lp": gap if gap is not None else "timeout",
            }
            rows.append(row)
            per_s_rows.append(row)

        gaps_by_target = [f"{row['target']}:{_fmt(row['gap_milp_minus_lp'])}" for row in per_s_rows]
        print("  gaps (target:gap) =", ", ".join(gaps_by_target))
        print("")

    with open(args.out_csv, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "s",
                "groundtruth",
                "target",
                "milp_min_yc_minus_yr",
                "lp_lower_bound_yc_minus_yr",
                "gap_milp_minus_lp",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "s": row["s"],
                    "groundtruth": row["groundtruth"],
                    "target": row["target"],
                    "milp_min_yc_minus_yr": _fmt(row["milp_min_yc_minus_yr"]),
                    "lp_lower_bound_yc_minus_yr": _fmt(row["lp_lower_bound_yc_minus_yr"]),
                    "gap_milp_minus_lp": _fmt(row["gap_milp_minus_lp"]),
                }
            )

    print(f"Wrote {len(rows)} rows to {args.out_csv}")


if __name__ == "__main__":
    main()