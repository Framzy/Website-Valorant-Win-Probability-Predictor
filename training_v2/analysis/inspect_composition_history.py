from training_v2.config import DATASET_PATH
import pandas as pd

TEAM = "Gen.G"
MAP_NAME = "Lotus"

COMPOSITION = {
    "fade",
    "killjoy",
    "omen",
    "raze",
    "viper"
}

df = pd.read_csv(DATASET_PATH)

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

grouped["Agent Set"] = grouped["Agent"].apply(
    lambda x: set(a.lower() for a in x)
)

result = grouped[
    (grouped["Team"] == TEAM)
    &
    (grouped["Map"] == MAP_NAME)
    &
    (grouped["Agent Set"] == COMPOSITION)
]

print(result[
    [
        "Tournament",
        "Stage",
        "Match Type",
        "Year",
        "Total Maps Played",
        "Total Wins By Map",
        "Total Loss By Map",
    ]
])