"""Plot latent MCC against latent dimension for the synthetic benchmark, from the aggregated CSV.

    python experiments/synthetic/plot.py --csv results/synthetic.csv --out results/synthetic.pdf

The figure has the size and style of the one in the paper. To draw it from the
paper's own table, pass --csv experiments/synthetic/paper_results.csv.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

C_LEJEPA = "#8C8C8C"
C_DSREG = "#0B6E69"


def main():
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0], formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--csv", default="results/synthetic.csv", help="table written by aggregate.py")
    parser.add_argument("--out", default="results/synthetic.pdf", help="output figure")
    args = parser.parse_args()

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 7.2,
        "axes.labelsize": 7.4,
        "xtick.labelsize": 6.9,
        "ytick.labelsize": 6.9,
        "axes.linewidth": 0.7,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    df = pd.read_csv(args.csv).sort_values("N")
    fig = plt.figure(figsize=(1.594, 1.209))
    ax = fig.add_subplot(fig.add_gridspec(1, 1, left=0.245, right=0.97, top=0.965, bottom=0.30)[0, 0])
    ax.errorbar(df["N"], df["posthoc_mcc_mean"], yerr=df["posthoc_mcc_std"], color=C_DSREG,
                marker="o", markersize=2.6, linewidth=1.3, elinewidth=0.7, capsize=1.5, capthick=0.7, zorder=3)
    ax.errorbar(df["N"], df["baseline_mcc_mean"], yerr=df["baseline_mcc_std"], color=C_LEJEPA,
                marker="s", markersize=2.4, linewidth=1.3, linestyle=(0, (3, 2)), elinewidth=0.7,
                capsize=1.5, capthick=0.7, zorder=2)
    ax.text(9.0, 0.905, "DSReg", color=C_DSREG, fontsize=7.2, fontweight="bold", ha="center")
    ax.text(11.2, 0.520, "LeJEPA", color=C_LEJEPA, fontsize=7.2, fontweight="bold", ha="center", va="top")
    ax.set_xlabel("Latent dimension $N$")
    ax.set_ylabel("Latent MCC")
    ax.set_xticks([4, 6, 8, 10, 12, 14])
    ax.set_ylim(0.38, 1.05)
    ax.set_yticks([0.4, 0.6, 0.8, 1.0])
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines["left"].set_color("#747B84")
    ax.spines["bottom"].set_color("#747B84")
    ax.tick_params(colors="#222831", length=2.2, pad=1.2)
    ax.grid(axis="y", color="#DDDDDD", lw=0.5, zorder=0)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
