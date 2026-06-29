"""
Build Team Dataset V2
=====================
"""

from pathlib import Path

import pandas as pd

from training_v2.config import DATASET_PATH
from training_v2.config import DATASET_DIR
from training_v2.training_utils import aggregate_matches
from training_v2.preprocessing.composition_reconstruction import (
    analyze_roles,
    calculate_role_distribution,
    allocate_role_slots,
    select_agents,
)


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


def aggregate_dataset(
    df: pd.DataFrame
) -> pd.DataFrame:
    """
    Aggregate raw dataset into compositions.
    """

    print("\n" + "=" * 60)
    print("AGGREGATE DATASET")
    print("=" * 60)

    grouped = aggregate_matches(df)

    print(f"[INFO] Total Composition : {len(grouped)}")

    return grouped


def split_dataset(
    grouped: pd.DataFrame
):
    """
    Split composition into

    - exact 5 agents
    - more than 5 agents
    """

    exact_five = grouped[
        grouped["Agent"].apply(len) == 5
    ].copy()

    greater_than_five = grouped[
        grouped["Agent"].apply(len) > 5
    ].copy()

    print("\n" + "=" * 60)
    print("SPLIT DATASET")
    print("=" * 60)

    print(f"[INFO] Exactly 5 Agent : {len(exact_five)}")

    print(f"[INFO] >5 Agent        : {len(greater_than_five)}")

    return exact_five, greater_than_five

def build_agent_played_frequency(
    raw_df: pd.DataFrame,
    team: str,
    map_name: str,
    year: int | None = None,
) -> dict:
    """
    Build agent played frequency.

    Agent Played =
        Total Wins By Map +
        Total Loss By Map
    """

    mask = (
        (raw_df["Team"] == team)
        &
        (raw_df["Map"] == map_name)
    )

    if year is not None:
        mask &= (
            raw_df["Year"] == year
        )

    subset = raw_df.loc[mask]

    subset = subset.copy()

    subset["Agent Played"] = (
        subset["Total Wins By Map"]
        +
        subset["Total Loss By Map"]
    )

    frequency = (
        subset
        .groupby("Agent")["Agent Played"]
        .sum()
        .sort_values(ascending=False)
        .to_dict()
    )

    return {
        agent.lower(): int(total)
        for agent, total in frequency.items()
    }
    
def reconstruct_composition(
    frequency: dict[str, int]
) -> list[str]:
    """
    Reconstruct composition from
    historical agent frequency.
    """

    analysis = analyze_roles(
        frequency
    )

    distribution = calculate_role_distribution(
        analysis
    )

    slots = allocate_role_slots(
        distribution
    )

    result = select_agents(
        distribution,
        slots
    )

    return sorted(
        result["composition"]
    )

def reconstruct_dataset(
    raw_df: pd.DataFrame,
    greater_than_five: pd.DataFrame,
) -> pd.DataFrame:
    """
    Reconstruct every composition
    containing more than five agents.
    """

    reconstructed = greater_than_five.copy()

    reconstructed_agents = []

    print("\n" + "=" * 60)
    print("RECONSTRUCT DATASET")
    print("=" * 60)

    total = len(reconstructed)

    for index, row in reconstructed.iterrows():

        frequency = build_agent_played_frequency(

            raw_df=raw_df,

            team=row["Team"],

            map_name=row["Map"],

            year=row["Year"]

        )

        composition = reconstruct_composition(
            frequency
        )

        reconstructed_agents.append(
            composition
        )

        if (len(reconstructed_agents) % 100) == 0:

            print(
                f"[INFO] Processed "
                f"{len(reconstructed_agents)}/{total}"
            )

    reconstructed["Agent"] = reconstructed_agents
    
    return reconstructed

def merge_dataset(
    exact_five: pd.DataFrame,
    reconstructed: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge original and reconstructed compositions.
    """

    print("\n" + "=" * 60)
    print("MERGE DATASET")
    print("=" * 60)

    dataset = pd.concat(

        [

            exact_five,

            reconstructed

        ],

        ignore_index=True

    )

    print(

        f"[INFO] Total Composition : {len(dataset)}"

    )

    return dataset

def calculate_winrate(
    dataset: pd.DataFrame
) -> pd.DataFrame:
    """
    Calculate composition winrate.
    """

    dataset = dataset.copy()

    dataset["Winrate"] = (

        dataset["Total Wins By Map"]

        /

        (

            dataset["Total Wins By Map"]

            +

            dataset["Total Loss By Map"]

        )

    )

    print("\n" + "=" * 60)
    print("CALCULATE WINRATE")
    print("=" * 60)

    print(

        f"[INFO] Mean Winrate : "

        f"{dataset['Winrate'].mean():.4f}"

    )

    return dataset

OUTPUT_DATASET = (

    DATASET_DIR

    /

    "valorant_dataset_team_v2.csv"

)

def save_dataset(
    dataset: pd.DataFrame
):
    """
    Save Team Dataset V2.
    """

    print("\n" + "=" * 60)
    print("SAVE DATASET")
    print("=" * 60)

    dataset.to_csv(

        OUTPUT_DATASET,

        index=False

    )

    print(

        f"[INFO] Saved : "

        f"{OUTPUT_DATASET.name}"

    )
def main():

    df = load_dataset()

    grouped = aggregate_dataset(df)

    exact_five, greater_than_five = split_dataset(grouped)

    reconstructed = reconstruct_dataset(

        raw_df=df,

        greater_than_five=greater_than_five,

    )

    dataset = merge_dataset(

        exact_five,

        reconstructed,

    )

    dataset = calculate_winrate(
        dataset
    )

    save_dataset(
        dataset
    )
    


if __name__ == "__main__":
    main()