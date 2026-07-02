from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


OUT_DIR = Path(__file__).resolve().parent

MODELS = ["AMNet", "LocalMapper", "SAMMNet"]
SPLITS = ["Random", "EC-number"]

DATA = {
    "average": {
        "values": {
            "Random": [95.54, 81.35, 78.77],
            "EC-number": [70.13, 73.64, 71.57],
        },
        "ylabel": "Average (%)",
        "filename": "average_bar",
        "legend": True,
    },
    "top1": {
        "values": {
            "Random": [77.93, 78.91, 77.17],
            "EC-number": [48.89, 71.10, 71.57],
        },
        "ylabel": "Top 1% (%)",
        "filename": "top1_bar",
        "legend": False,
    },
}

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Nimbus Roman", "Times", "DejaVu Serif"],
        "font.size": 26,
        "axes.labelsize": 26,
        "xtick.labelsize": 26,
        "ytick.labelsize": 26,
        "legend.fontsize": 24,
        "axes.linewidth": 1.2,
        "xtick.major.width": 1.2,
        "ytick.major.width": 1.2,
        "xtick.major.size": 6,
        "ytick.major.size": 6,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "axes.unicode_minus": False,
    }
)


def plot_metric(values, ylabel, filename, legend=False):
    x = np.arange(len(MODELS))
    width = 0.34
    colors = ["#1868B2", "#DE582B"]

    fig, ax = plt.subplots(figsize=(8.8, 6.2), dpi=300)

    for idx, split in enumerate(SPLITS):
        offset = (idx - 0.5) * width
        ax.bar(
            x + offset,
            values[split],
            width=width,
            label=split,
            color=colors[idx],
            edgecolor="black",
            linewidth=1.0,
        )

    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(MODELS)
    ax.set_ylim(0, 100)
    ax.set_yticks(np.arange(0, 101, 20))

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out")
    ax.grid(False)

    if legend:
        ax.legend(
            frameon=False,
            loc="upper right",
            bbox_to_anchor=(1.0, 1.08),
            handlelength=1.2,
            handletextpad=0.4,
        )

    fig.tight_layout()
    fig.savefig(OUT_DIR / f"{filename}.pdf", bbox_inches="tight")
    fig.savefig(OUT_DIR / f"{filename}.png", bbox_inches="tight", dpi=600)
    plt.close(fig)


def main():
    for config in DATA.values():
        plot_metric(**config)


if __name__ == "__main__":
    main()
