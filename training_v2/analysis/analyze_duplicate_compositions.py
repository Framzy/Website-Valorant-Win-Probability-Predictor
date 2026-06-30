"""
Analyze Duplicate Compositions
==============================

Purpose
-------
Analyze duplicate team compositions before removing
the Total Maps Played >= 2 filter.

This script is only for research and validation.
It is NOT part of the training pipeline.
"""

from collections import Counter

import pandas as pd

from training_v2.config import DATASET_PATH
from training_v2.training_utils import aggregate_matches


def load_dataset() -> pd.DataFrame:
    """
    Load raw dataset.
    """

    print("=" * 60)
    print("LOAD DATASET")
    print("=" * 60)

    df = pd.read_csv(DATASET_PATH)

    print(f"[INFO] Shape : {df.shape}")

    return df

def aggregate_matches_without_filter(
    df: pd.DataFrame
) -> pd.DataFrame:
    """
    Aggregate matches without
    Total Maps Played filtering.
    """

    print("\n" + "=" * 60)
    print("AGGREGATE MATCHES (WITHOUT FILTER)")
    print("=" * 60)

    grouped = (
        df.groupby(
            [
                "Tournament",
                "Stage",
                "Match Type",
                "Year",
                "Map",
                "Team",
            ],
            as_index=False,
        )
        .agg(
            {
                "Agent": list,
                "Total Wins By Map": "first",
                "Total Loss By Map": "first",
                "Total Maps Played": "first",
            }
        )
    )

    print(f"[INFO] Total Composition : {len(grouped)}")

    return grouped

def build_composition_key(row) -> tuple:
    """
    Build unique composition key.

    Team
    Year
    Map
    Agents
    """

    agents = tuple(sorted(row["Agent"]))

    return (
        row["Year"],
        row["Team"],
        row["Map"],
        agents,
    )


def analyze_duplicates(grouped: pd.DataFrame):

    print("\n" + "=" * 60)
    print("DUPLICATE ANALYSIS")
    print("=" * 60)

    grouped = grouped.copy()

    grouped["Composition Key"] = grouped.apply(
        build_composition_key,
        axis=1,
    )

    counts = Counter(grouped["Composition Key"])

    duplicate_keys = {
        key: value
        for key, value in counts.items()
        if value > 1
    }

    print(f"[INFO] Total Composition : {len(grouped)}")
    print(f"[INFO] Unique Composition : {len(counts)}")
    print(f"[INFO] Duplicate Composition : {len(duplicate_keys)}")

    if not duplicate_keys:
        print("\nNo duplicate compositions found.")
        return

    print("\nTop 10 Duplicate Composition\n")

    top_duplicates = sorted(
        duplicate_keys.items(),
        key=lambda x: x[1],
        reverse=True,
    )[:10]

    for key, occurrence in top_duplicates:

        year, team, map_name, agents = key

        print("-" * 60)
        print(f"Year        : {year}")
        print(f"Team        : {team}")
        print(f"Map         : {map_name}")
        print(f"Occurrence  : {occurrence}")
        print("Composition :")

        for agent in agents:
            print(f"  - {agent}")


def main():

    df = load_dataset()

    # ---------------------------------
    # WITH FILTER
    # ---------------------------------

    print("\n" + "=" * 60)
    print("WITH FILTER")
    print("=" * 60)

    grouped_filter = aggregate_matches(df)

    analyze_duplicates(grouped_filter)

    # ---------------------------------
    # WITHOUT FILTER
    # ---------------------------------

    print("\n" + "=" * 60)
    print("WITHOUT FILTER")
    print("=" * 60)

    grouped_no_filter = aggregate_matches_without_filter(df)

    analyze_duplicates(grouped_no_filter)


if __name__ == "__main__":
    main()