import pandas as pd
from pathlib import Path

# =========================
# PATH
# =========================

BASE_DIR = Path(__file__).resolve().parent

DATASET_PATH = BASE_DIR / "dataset" / "valorant_dataset_all.csv"
OUTPUT_PATH = BASE_DIR / "dataset" / "historical_compositions.csv"



# =========================
# ROLE MAP
# =========================

AGENT_ROLE_MAP = {
    **{a: "duelist" for a in [
        "iso", "jett", "raze", "reyna",
        "yoru", "neon", "phoenix", "waylay"
    ]},

    **{a: "initiator" for a in [
        "skye", "sova", "breach",
        "fade", "kayo", "gekko", "tejo"
    ]},

    **{a: "controller" for a in [
        "brimstone", "omen", "viper",
        "astra", "harbor", "clove"
    ]},

    **{a: "sentinel" for a in [
        "killjoy", "cypher", "sage",
        "chamber", "deadlock", "vyse"
    ]},

    # Agent baru
    "miks": "initiator",
    "veto": "controller",
}

# =========================
# LOAD DATA
# =========================

print("[INFO] Loading dataset...")

df = pd.read_csv(DATASET_PATH)

print(df.columns.tolist())
print(df.head())

df["Agent"] = df["Agent"].str.lower()

print(f"[INFO] Total Rows: {len(df)}")

# =========================
# BUILD COMPOSITIONS
# =========================

records = []

group_cols = [
    "Tournament",
    "Stage",
    "Match Type",
    "Map",
    "Team"
]

groups = df.groupby(group_cols)
print()
print("===== GROUP SIZE CHECK =====")

group_sizes = groups.size()

print(group_sizes.describe())

print()
print(group_sizes.value_counts().sort_index())

print("============================")

print(f"[INFO] Total Groups: {len(groups)}")

print()

print("===== MAP PLAYED DISTRIBUTION =====")

print(
    df_grouped["Total Played"]
    .value_counts()
    .sort_index()
)

print("==============================")

for keys, group in groups:

    tournament, stage, match_type, map_name, team = keys

    total_maps = int(group["Total Maps Played"].max())

    if total_maps <= 0:
        continue

    role_agents = {
        "duelist": [],
        "initiator": [],
        "controller": [],
        "sentinel": []
    }

    # =====================
    # Agent Usage Score
    # =====================

    for _, row in group.iterrows():

        agent = row["Agent"]

        if agent not in AGENT_ROLE_MAP:
            continue

        role = AGENT_ROLE_MAP[agent]

        usage_score = (
            row["Total Wins By Map"] +
            row["Total Loss By Map"]
        )

        role_agents[role].append(
            (agent, usage_score)
        )

    # Sort usage tertinggi
    for role in role_agents:
        role_agents[role].sort(
            key=lambda x: x[1],
            reverse=True
        )

    composition = []

    # =====================
    # Mandatory Roles
    # =====================

    for role in [
        "controller",
        "initiator",
        "duelist"
    ]:
        if role_agents[role]:
            composition.append(
                role_agents[role][0][0]
            )

    # =====================
    # Flex Pool
    # =====================

    remaining = []

    for role in role_agents:
        for agent, score in role_agents[role]:
            if agent not in composition:
                remaining.append(
                    (agent, score, role)
                )

    remaining.sort(
        key=lambda x: x[1],
        reverse=True
    )

    sentinel_count = 0

    for agent in composition:
        if AGENT_ROLE_MAP[agent] == "sentinel":
            sentinel_count += 1

    for agent, score, role in remaining:

        if len(composition) >= 5:
            break

        if role == "sentinel" and sentinel_count >= 2:
            continue

        composition.append(agent)

        if role == "sentinel":
            sentinel_count += 1

    if len(composition) != 5:
        continue

    winrate = (
        group["Total Wins By Map"].sum()
        /
        (
            group["Total Wins By Map"].sum()
            +
            group["Total Loss By Map"].sum()
        )
    )

    records.append({
        "Tournament": tournament,
        "Stage": stage,
        "Match Type": match_type,
        "Map": map_name,
        "Team": team,
        "Composition": ",".join(sorted(composition)),
        "Winrate": round(winrate, 4),
        "Total Maps Played": total_maps
    })

# =========================
# SAVE
# =========================


result_df = pd.DataFrame(records)

print()
print("[INFO] Preview:")
print(result_df.head())

print(result_df["Winrate"].describe())

print(
    result_df["Winrate"]
    .value_counts()
    .sort_index()
)

print()
print(f"[INFO] Total Compositions: {len(result_df)}")

result_df.to_csv(
    OUTPUT_PATH,
    index=False
)

print()
print("[INFO] Saved:")
print(OUTPUT_PATH)