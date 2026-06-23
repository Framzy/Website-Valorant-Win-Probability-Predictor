from collections import Counter
import pandas as pd

from config import DATASET_PATH

print("[INFO] Pre-loading historical data...")

try:
    df_hist = pd.read_csv(DATASET_PATH)
    df_hist["match_id"] = (
        df_hist["Tournament"] + "_"
        + df_hist["Stage"] + "_"
        + df_hist["Match Type"] + "_"
        + df_hist["Map"] + "_"
        + df_hist["Team"]
    )

    HIST_DATA_LOADED = True
    print("[INFO] Historical data loaded.")

except Exception as e:
    HIST_DATA_LOADED = False
    print(f"[WARN] Could not load historical data: {e}")


def get_team_map_stats(team, map_):

    most_common_combo = "Tidak ditemukan"
    historical_agents_set = set()

    if not HIST_DATA_LOADED:
        return most_common_combo, historical_agents_set

    try:
        df_filtered = df_hist[
            (df_hist["Team"].str.lower() == team.lower())
            &
            (df_hist["Map"].str.lower() == map_.lower())
        ]
        combos = []

        for _, g in df_filtered.groupby("match_id"):
            if len(g) == 5:
                combo = tuple(
                    sorted(
                        g["Agent"].str.lower()
                    )
                )
                combos.append(combo)
                historical_agents_set.update(combo)

        if combos:
            most_common = (
                Counter(combos)
                .most_common(1)[0][0]
            )
            most_common_combo = (
                ", ".join(most_common).title()
            )

    except Exception:
        most_common_combo = "Error memuat combo"

    return (
        most_common_combo,
        historical_agents_set
    )