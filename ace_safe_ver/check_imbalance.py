"""
Statistics and bar plot for the filtered SAFE pairs dataset,
by dataset_name and endpoint.
"""
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

DATA_PATH = Path(__file__).resolve().parent / "pairs_safe_filtered.csv"
OUT_PLOT_PATH = Path(__file__).resolve().parent / "pairs_safe_filtered_stats_barplot.png"


def main():
    print(f"Loading: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    if "dataset_name" not in df.columns or "endpoint" not in df.columns:
        raise ValueError("CSV must have 'dataset_name' and 'endpoint' columns.")

    n_total = len(df)
    print("\n" + "=" * 60)
    print("OVERALL")
    print("=" * 60)
    print(f"Total pairs: {n_total}")
    print(f"Unique dataset_name: {df['dataset_name'].nunique()}")
    print(f"Unique endpoint: {df['endpoint'].nunique()}")

    # By (dataset_name, endpoint)
    by_both = df.groupby(["dataset_name", "endpoint"], dropna=False).size().reset_index(name="count")
    by_both = by_both.sort_values("count", ascending=False)

    print("\n" + "=" * 60)
    print("BY DATASET_NAME AND ENDPOINT")
    print("=" * 60)
    print(by_both.to_string(index=False))

    # By dataset_name only
    by_dataset = df.groupby("dataset_name", dropna=False).size().reset_index(name="count")
    by_dataset = by_dataset.sort_values("count", ascending=False)
    print("\n" + "=" * 60)
    print("BY DATASET_NAME")
    print("=" * 60)
    print(by_dataset.to_string(index=False))

    # By endpoint only
    by_endpoint = df.groupby("endpoint", dropna=False).size().reset_index(name="count")
    by_endpoint = by_endpoint.sort_values("count", ascending=False)
    print("\n" + "=" * 60)
    print("BY ENDPOINT")
    print("=" * 60)
    print(by_endpoint.to_string(index=False))

    # Bar plot: pair count by (dataset_name, endpoint)
    # Use plain Python types to avoid matplotlib deepcopy RecursionError (e.g. with pandas/Python 3.14)
    plot_df = by_both.head(30)
    n_bars = len(plot_df)
    counts = [int(plot_df["count"].iloc[i]) for i in range(n_bars)]
    labels = [
        str(plot_df["dataset_name"].iloc[i]) + "\n" + str(plot_df["endpoint"].iloc[i])
        for i in range(n_bars)
    ]

    # fig, ax = plt.subplots(figsize=(12, 6))
    # x = list(range(n_bars))
    # ax.bar(x, counts, color="steelblue", edgecolor="navy", alpha=0.85)
    # ax.set_xticks(x)
    # ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    # ax.set_ylabel("Number of pairs", fontsize=11)
    # ax.set_xlabel("Dataset × Endpoint", fontsize=11)
    # ax.set_title("Filtered SAFE pairs: count by dataset and endpoint", fontsize=12)
    # ax.spines["top"].set_visible(False)
    # ax.spines["right"].set_visible(False)
    # plt.tight_layout()
    # plt.savefig(OUT_PLOT_PATH, dpi=150, bbox_inches="tight")
    # plt.close()
    # print(f"\nBar plot saved: {OUT_PLOT_PATH}")


if __name__ == "__main__":
    main()
