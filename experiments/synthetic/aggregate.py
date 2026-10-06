"""Aggregate the per-run JSON records of the synthetic benchmark into one row per latent dimension N.

    python experiments/synthetic/aggregate.py --results results/synthetic --out results/synthetic.csv

Every column also appears, under the same name, in
experiments/synthetic/paper_results.csv, the table behind the paper's figure
for this benchmark, so the two files can be compared directly. "baseline" is
the unrotated LeJEPA representation and "posthoc" is its DSReg rotation.
Standard deviations are sample standard deviations (ddof=1) across seeds.
"""

import argparse
import json
from pathlib import Path

import pandas as pd


def load_records(results_dir):
    """Read every JSON record written by run.py in results_dir."""
    rows = []
    for path in sorted(Path(results_dir).glob("*.json")):
        rec = json.loads(path.read_text())
        rows.append(
            {
                "N": int(rec["N"]),
                "seed": int(rec["seed"]),
                "baseline_mcc": float(rec["lejepa_mcc"]),
                "posthoc_mcc": float(rec["dsreg_mcc"]),
                "baseline_r2_hz": float(rec["lejepa_r2"]),
            }
        )
    return pd.DataFrame(rows)


def summarize(rows, dims):
    selected = rows[rows["N"].isin(dims)]
    return (
        selected.groupby("N", as_index=False)
        .agg(
            n=("seed", "count"),
            seeds=("seed", lambda values: ",".join(str(int(v)) for v in sorted(values))),
            baseline_mcc_mean=("baseline_mcc", "mean"),
            baseline_mcc_std=("baseline_mcc", "std"),
            posthoc_mcc_mean=("posthoc_mcc", "mean"),
            posthoc_mcc_std=("posthoc_mcc", "std"),
            baseline_r2_mean=("baseline_r2_hz", "mean"),
            baseline_r2_min=("baseline_r2_hz", "min"),
        )
        .sort_values("N")
    )


def main():
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0], formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--results", default="results/synthetic", help="directory of per-run JSON files")
    parser.add_argument("--out", default="results/synthetic.csv", help="output CSV")
    parser.add_argument("--dims", type=int, nargs="+", default=[4, 6, 8, 10, 12, 14], help="latent dimensions N to keep")
    args = parser.parse_args()

    rows = load_records(args.results)
    if rows.empty:
        raise SystemExit(f"no JSON records in {args.results}")
    summary = summarize(rows, args.dims)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.out, index=False)
    print(summary.to_string(index=False))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
