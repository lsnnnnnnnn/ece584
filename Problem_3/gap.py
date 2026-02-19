import argparse
import csv
import importlib
import multiprocessing as mp
import os
import sys
import traceback
from typing import Any, Dict, List


S = [0.0, 0.001, 0.003, 0.01, 0.03, 0.1]


def add_path() -> None:
    d = os.path.dirname(os.path.abspath(__file__))
    if d not in sys.path:
        sys.path.insert(0, d)


def work(mod_name: str, data_file: str, eps: float, q: "mp.Queue") -> None:
    try:
        add_path()
        import contextlib
        import io

        m = importlib.import_module(mod_name)

        # silence solver/model prints
        with contextlib.redirect_stdout(io.StringIO()):
            model, x, y = m.load_model_and_data(data_file)
            v = m.Verifier()

            vars_ = v.create_inputs(x, eps)
            W, b = m.extract_model_weight(model)

            L = len(W)
            for i in range(L):
                vars_ = list(v.add_linear_layer(vars_, W[i], b[i]))
                if i != L - 1:
                    lbs, _ = v.solve_objectives(vars_, direction="minimization")
                    ubs, _ = v.solve_objectives(vars_, direction="maximization")
                    vars_ = v.add_relu_layer(vars_, lbs, ubs)

            objs, tgts = v.get_verification_objectives(vars_, y)
            vals, _ = v.solve_objectives(objs, direction="minimization")

        q.put({"ok": True, "objs": vals, "tgts": tgts, "y": int(y)})

    except BaseException as e:
        q.put({"ok": False, "err": repr(e), "tb": traceback.format_exc()})


def run(mod_name: str, data_file: str, eps: float, tlim: float) -> Dict[str, Any]:
    ctx = mp.get_context("spawn")
    q: "mp.Queue" = ctx.Queue()
    p = ctx.Process(target=work, args=(mod_name, data_file, eps, q), daemon=True)

    p.start()
    p.join(timeout=None if tlim <= 0 else tlim)

    if p.is_alive():
        p.terminate()
        p.join()
        return {"ok": False, "timeout": True}

    if q.empty():
        return {"ok": False, "err": f"worker exited with code {p.exitcode} and produced no result"}

    return q.get()


def fmt(x: Any) -> str:
    if x is None:
        return "timeout"
    if isinstance(x, str):
        return x
    try:
        return f"{float(x):.10g}"
    except Exception:
        return str(x)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("data_file", type=str, help="e.g. data1.pth")
    ap.add_argument("--time_limit", type=float, default=600.0, help="seconds per run. 0 = no limit")
    ap.add_argument("--out_csv", type=str, default="gap_results.csv")
    ap.add_argument("--milp_module", type=str, default="mip_assignment")
    ap.add_argument("--lp_module", type=str, default="lp")
    args = ap.parse_args()

    add_path()

    rows: List[Dict[str, Any]] = []

    print("Gap experiment (MILP vs LP)")
    print(f"data_file = {args.data_file}")
    print(f"time_limit = {args.time_limit} sec")
    print(f"MILP mod  = {args.milp_module}")
    print(f"LP mod    = {args.lp_module}")
    print(f"S         = {S}\n")

    for eps in S:
        print(f"s = {eps}")

        milp = run(args.milp_module, args.data_file, eps, args.time_limit)
        lp = run(args.lp_module, args.data_file, eps, args.time_limit)

        # hard errors (non-timeout)
        for name, res in (("MILP", milp), ("LP", lp)):
            if (not res.get("ok", False)) and (not res.get("timeout", False)) and ("err" in res):
                print(f"  {name}: ERROR {res['err']}")
                if "tb" in res:
                    print(res["tb"])
                raise SystemExit(1)

        print("  MILP:", "done" if milp.get("ok", False) else "timeout")
        print("  LP:  ", "done" if lp.get("ok", False) else "timeout")

        if milp.get("timeout") and lp.get("timeout"):
            print("  Both timed out.\n")
            continue

        meta = milp if milp.get("ok", False) else lp
        tgts = meta["tgts"]
        y = meta["y"]

        milp_objs = milp["objs"] if milp.get("ok", False) else None
        lp_objs = lp["objs"] if lp.get("ok", False) else None

        per_eps = []
        for i, r in enumerate(tgts):
            if r == y:
                continue

            a = None if milp_objs is None else milp_objs[i]
            b = None if lp_objs is None else lp_objs[i]

            gap = None
            if a is not None and b is not None and not isinstance(a, str) and not isinstance(b, str):
                try:
                    gap = float(a) - float(b)
                except Exception:
                    gap = None

            row = {
                "s": eps,
                "groundtruth": y,
                "target": r,
                "milp_min_yc_minus_yr": a,
                "lp_lower_bound_yc_minus_yr": b,
                "gap_milp_minus_lp": gap,
            }
            rows.append(row)
            per_eps.append(row)

        print("  gaps (target:gap) =", ", ".join(f"{r['target']}:{fmt(r['gap_milp_minus_lp'])}" for r in per_eps))
        print("")

    with open(args.out_csv, "w", newline="") as f:
        w = csv.DictWriter(
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
        w.writeheader()
        for r in rows:
            w.writerow(
                {
                    "s": fmt(r["s"]),
                    "groundtruth": r["groundtruth"],
                    "target": r["target"],
                    "milp_min_yc_minus_yr": fmt(r["milp_min_yc_minus_yr"]),
                    "lp_lower_bound_yc_minus_yr": fmt(r["lp_lower_bound_yc_minus_yr"]),
                    "gap_milp_minus_lp": fmt(r["gap_milp_minus_lp"]),
                }
            )

    print(f"Wrote {len(rows)} rows to {args.out_csv}")


if __name__ == "__main__":
    main()
