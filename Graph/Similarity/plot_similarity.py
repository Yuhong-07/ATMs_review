import os
os.environ.setdefault("MPLCONFIGDIR", f"/tmp/mplconfig-{os.getuid()}")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem import AllChem
from rdkit import DataStructs

from sklearn.manifold import TSNE

RDLogger.DisableLog("rdApp.*")


# ============================================================
# Global style
# ============================================================

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Nimbus Roman", "Times", "DejaVu Serif"],
        "font.size": 26,
        "axes.labelsize": 26,
        "axes.titlesize": 26,
        "xtick.labelsize": 26,
        "ytick.labelsize": 26,
        "legend.fontsize": 26,
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


# ============================================================
# Configuration
# ============================================================

RANDOM_SEED = 42
RADIUS = 2
NBITS = 2048

# Sampling
TSNE_SAMPLE_TRAIN = 4000
TSNE_SAMPLE_TEST = None   # None = use all test data

# Data and output
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join(SCRIPT_DIR, "dataset")
OUT_DIR = SCRIPT_DIR
CACHE_DIR = os.path.join(OUT_DIR, "fingerprint_cache")

# Display cropping:
# only changes the displayed axis range, not the t-SNE computation itself
DISPLAY_CROP_PERCENTILE = 1.0

TRAIN_COLOR = "#0C4E9B"
TEST_COLOR = "#C72228"
POINT_ALPHA = 1.0


DATASETS = [
    (
        "a",
        "Random split",
        "random",
        os.path.join(DATA_ROOT, "random", "train.csv"),
        os.path.join(DATA_ROOT, "random", "test.csv"),
        False,    # no legend
        "train", # train on top
    ),
    (
        "b",
        "EC-number split",
        "EC_split_N2",
        os.path.join(DATA_ROOT, "EC_split_N2", "train.csv"),
        os.path.join(DATA_ROOT, "EC_split_N2", "test.csv"),
        True,   # with legend
        "test",  # test on top
    ),
]


# ============================================================
# Reaction parsing
# ============================================================

def split_reaction_smiles(rxn_smiles: str):
    if not isinstance(rxn_smiles, str):
        return None, None, None

    rxn_smiles = rxn_smiles.strip()

    if ">>" in rxn_smiles:
        parts = rxn_smiles.split(">>")
        if len(parts) != 2:
            return None, None, None
        return parts[0], "", parts[1]

    parts = rxn_smiles.split(">")
    if len(parts) == 3:
        return parts[0], parts[1], parts[2]

    return None, None, None


def detect_reaction_column(df: pd.DataFrame):
    for col in [
        "reactions",
        "rxn_smiles",
        "reaction",
        "smiles",
        "unmapped",
        "canonical_rxn",
        "reaction_smiles",
    ]:
        if col in df.columns:
            return col
    return None


# ============================================================
# Fingerprint generation
# ============================================================

def smiles_to_morgan_array(smiles: str, radius=2, nBits=2048):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits)
    arr = np.zeros((nBits,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def reaction_fp_struct(rxn_smiles: str, radius=2, nBits=2048):
    """
    Struct fingerprint:
        struct_fp = reactant_fp != product_fp

    This records which Morgan fingerprint bits changed
    from reactants to products.
    """
    reactants, agents, products = split_reaction_smiles(rxn_smiles)

    if reactants is None or products is None:
        return None

    r_fp = smiles_to_morgan_array(reactants, radius=radius, nBits=nBits)
    p_fp = smiles_to_morgan_array(products, radius=radius, nBits=nBits)

    if r_fp is None or p_fp is None:
        return None

    struct_fp = (r_fp != p_fp).astype(np.float32)

    # skip uninformative all-zero change vectors
    if struct_fp.sum() == 0:
        return None

    return struct_fp


# ============================================================
# Data loading / caching
# ============================================================

def sample_array(X, sample_size, seed):
    if sample_size is None or len(X) <= sample_size:
        return X

    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), size=sample_size, replace=False)
    return X[idx]


def make_fps_from_csv(csv_path):
    df = pd.read_csv(csv_path)

    col = detect_reaction_column(df)
    if col is None:
        raise ValueError(f"Cannot find reaction column in {csv_path}")

    fps = []
    for rxn in df[col].tolist():
        fp = reaction_fp_struct(rxn, radius=RADIUS, nBits=NBITS)
        if fp is not None:
            fps.append(fp)

    if len(fps) == 0:
        raise ValueError(f"No valid fingerprints from {csv_path}")

    return np.vstack(fps).astype(np.float32)


def load_or_make_fps(dataset_key, split_name, csv_path):
    os.makedirs(CACHE_DIR, exist_ok=True)

    cache_path = os.path.join(
        CACHE_DIR,
        f"{dataset_key}_{split_name}_struct_fp_radius{RADIUS}_nbits{NBITS}.npz",
    )

    if os.path.exists(cache_path):
        data = np.load(cache_path)
        X = data["fps"].astype(np.float32)
        print(f"Loaded cached fingerprints: {cache_path} | shape={X.shape}")
        return X

    print(f"Computing struct fingerprints: {dataset_key} {split_name}")
    X = make_fps_from_csv(csv_path)

    np.savez_compressed(cache_path, fps=X)
    print(f"Saved cached fingerprints: {cache_path} | shape={X.shape}")

    return X


def load_dataset_fps(dataset_key, train_csv, test_csv):
    train_fps = load_or_make_fps(dataset_key, "train", train_csv)
    test_fps = load_or_make_fps(dataset_key, "test", test_csv)
    return train_fps, test_fps


# ============================================================
# t-SNE
# ============================================================

def run_tsne_struct_cosine(train_fps, test_fps):
    train_s = sample_array(train_fps, TSNE_SAMPLE_TRAIN, RANDOM_SEED)
    test_s = sample_array(test_fps, TSNE_SAMPLE_TEST, RANDOM_SEED + 1)

    X = np.vstack([train_s, test_s]).astype(np.float32)
    labels = np.array(["train"] * len(train_s) + ["test"] * len(test_s), dtype=object)

    print("After sampling:")
    print("  train_s shape:", train_s.shape)
    print("  test_s shape :", test_s.shape)
    print("  train count  :", np.sum(labels == "train"))
    print("  test count   :", np.sum(labels == "test"))

    # Normalize for cosine distance
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    X = X / norms

    tsne = TSNE(
        n_components=2,
        metric="cosine",
        perplexity=30,
        learning_rate="auto",
        init="random",
        random_state=RANDOM_SEED,
        method="barnes_hut",
        angle=0.5,
        max_iter=1000,   # if sklearn version errors, replace with n_iter=1000
        verbose=1,
    )

    emb = tsne.fit_transform(X)

    return emb, labels


# ============================================================
# Plot helpers
# ============================================================

def set_display_region(ax, emb, crop_percentile=1.0):
    """
    Only changes displayed area, not the embedding itself.
    Used to hide remote outliers visually.
    """
    x_low, x_high = np.percentile(emb[:, 0], [crop_percentile, 100 - crop_percentile])
    y_low, y_high = np.percentile(emb[:, 1], [crop_percentile, 100 - crop_percentile])

    x_pad = 0.05 * (x_high - x_low)
    y_pad = 0.05 * (y_high - y_low)

    ax.set_xlim(x_low - x_pad, x_high + x_pad)
    ax.set_ylim(y_low - y_pad, y_high + y_pad)


def style_axes(ax):
    """
    Keep border and axis titles, but remove axis numbers.
    """
    # axis titles more aligned with the algorithm
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")

    # remove numeric ticks and tick labels
    ax.set_xticks([])
    ax.set_yticks([])

    # keep border
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.2)


def plot_single_panel(
    panel_letter,
    title,
    dataset_key,
    emb,
    labels,
    add_legend,
    top_layer,
):
    os.makedirs(OUT_DIR, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 6), dpi=600)

    train_mask = labels == "train"
    test_mask = labels == "test"

    train_n = int(train_mask.sum())
    test_n = int(test_mask.sum())

    # Different layer order:
    # random: train on top
    # EC:     test on top
    if top_layer == "train":
        # draw test first, then train
        ax.scatter(
            emb[test_mask, 0],
            emb[test_mask, 1],
            s=45,
            alpha=POINT_ALPHA,
            c=TEST_COLOR,
            linewidths=0,
            label=f"test",
        )
        ax.scatter(
            emb[train_mask, 0],
            emb[train_mask, 1],
            s=25,
            alpha=POINT_ALPHA,
            c=TRAIN_COLOR,
            linewidths=0,
            label=f"train",
        )
    else:
        # draw train first, then test
        ax.scatter(
            emb[train_mask, 0],
            emb[train_mask, 1],
            s=30,
            alpha=POINT_ALPHA,
            c=TRAIN_COLOR,
            linewidths=0,
            label=f"train",
        )
        ax.scatter(
            emb[test_mask, 0],
            emb[test_mask, 1],
            s=30,
            alpha=POINT_ALPHA,
            c=TEST_COLOR,
            linewidths=0,
            label=f"test",
        )

    set_display_region(ax, emb, crop_percentile=DISPLAY_CROP_PERCENTILE)
    style_axes(ax)

    ax.set_title(title, fontsize=26, pad=16)

    # # panel letter
    # ax.text(
    #     -0.12,
    #     1.02,
    #     panel_letter,
    #     transform=ax.transAxes,
    #     fontsize=26,
    #     fontweight="bold",
    #     ha="left",
    #     va="bottom",
    # )

    # only the requested panel has legend
    if add_legend:
        legend = ax.legend(
            frameon=True,
            loc="lower right",
            markerscale=1.7,
            handlelength=0.8,
            handletextpad=0.3,
            borderpad=0.25,
            labelspacing=0.25,
            borderaxespad=0.35,
        )
        legend.get_frame().set_facecolor("white")
        legend.get_frame().set_edgecolor("black")
        legend.get_frame().set_linewidth(1.0)
        legend.get_frame().set_alpha(1.0)

    fig.tight_layout()

    out_base = os.path.join(OUT_DIR, f"tsne_{dataset_key}_struct_cosine")
    fig.savefig(f"{out_base}.png", bbox_inches="tight", dpi=600)
    fig.savefig(f"{out_base}.pdf", bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {out_base}.png")
    print(f"Saved: {out_base}.pdf")


# ============================================================
# Main
# ============================================================

def main():
    print("Output dir:", OUT_DIR)
    print("Cache dir:", CACHE_DIR)
    print("Fingerprint: struct")
    print("Distance: cosine")
    print(f"TSNE_SAMPLE_TRAIN = {TSNE_SAMPLE_TRAIN}")
    print("TSNE_SAMPLE_TEST = all")

    for panel_letter, title, dataset_key, train_csv, test_csv, add_legend, top_layer in DATASETS:
        print("\n" + "=" * 80)
        print(f"Dataset: {title}")
        print("=" * 80)

        train_fps, test_fps = load_dataset_fps(dataset_key, train_csv, test_csv)
        print(f"Valid struct fingerprints: train={len(train_fps)}, test={len(test_fps)}")

        emb, labels = run_tsne_struct_cosine(train_fps, test_fps)

        plot_single_panel(
            panel_letter=panel_letter,
            title=title,
            dataset_key=dataset_key,
            emb=emb,
            labels=labels,
            add_legend=add_legend,
            top_layer=top_layer,
        )


if __name__ == "__main__":
    main()