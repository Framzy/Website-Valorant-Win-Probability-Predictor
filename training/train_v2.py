# train_team_v2.py

import pandas as pd
import numpy as np
import joblib

from pathlib import Path

from sklearn.preprocessing import (
    OneHotEncoder,
    MultiLabelBinarizer,
    StandardScaler
)

from sklearn.model_selection import (
    train_test_split,
    KFold,
    cross_val_score
)

from sklearn.metrics import mean_absolute_error

from xgboost import XGBRegressor

BASE_DIR = Path(__file__).resolve().parent

DATASET_PATH = BASE_DIR / "dataset" / "valorant_dataset_all.csv"

MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

AGENT_ROLE_MAP = {
    # Duelist
    "iso": "duelist",
    "jett": "duelist",
    "raze": "duelist",
    "reyna": "duelist",
    "yoru": "duelist",
    "neon": "duelist",
    "phoenix": "duelist",
    "waylay": "duelist",

    # Initiator
    "skye": "initiator",
    "sova": "initiator",
    "breach": "initiator",
    "fade": "initiator",
    "kayo": "initiator",
    "gekko": "initiator",
    "tejo": "initiator",

    # Controller
    "brimstone": "controller",
    "omen": "controller",
    "viper": "controller",
    "astra": "controller",
    "harbor": "controller",
    "clove": "controller",    
    "miks": "controller",


    # Sentinel
    "killjoy": "sentinel",
    "cypher": "sentinel",
    "sage": "sentinel",
    "chamber": "sentinel",
    "deadlock": "sentinel",
    "vyse": "sentinel",
    "veto": "sentinel",

}

ROLE_ORDER = [
    "duelist",
    "initiator",
    "controller",
    "sentinel"
]


def get_role_vector(agents):
    count = {role: 0 for role in ROLE_ORDER}

    for agent in agents:
        role = AGENT_ROLE_MAP.get(agent)

        if role:
            count[role] += 1

    return [count[r] for r in ROLE_ORDER]

print("[INFO] Loading dataset...")

df = pd.read_csv(DATASET_PATH)

df["Agent"] = df["Agent"].str.lower()

print(df.head())

print("[INFO] Aggregating data...")

group_cols = [
    "Tournament",
    "Stage",
    "Match Type",
    "Map",
    "Team"
]

df_grouped = df.groupby(group_cols).agg({
    "Agent": lambda x: list(x),
    "Total Wins By Map": "max",
    "Total Loss By Map": "max",
    "Total Maps Played": "max",
    "Year": "first"
}).reset_index()

print(
    f"[INFO] Before Filter: {len(df_grouped)}"
)

df_grouped = df_grouped[
    df_grouped["Total Maps Played"] >= 2
].reset_index(drop=True)

print(
    f"[INFO] After Filter: {len(df_grouped)}"
)

df_grouped["Winrate"] = (
    df_grouped["Total Wins By Map"]
    /
    (
        df_grouped["Total Wins By Map"]
        +
        df_grouped["Total Loss By Map"]
    )
)

print(df_grouped["Winrate"].describe())

team_ohe = OneHotEncoder(
    sparse_output=False,
    handle_unknown="ignore"
)

team_encoded = team_ohe.fit_transform(
    df_grouped[["Team"]]
)

map_ohe = OneHotEncoder(
    sparse_output=False,
    handle_unknown="ignore"
)

map_encoded = map_ohe.fit_transform(
    df_grouped[["Map"]]
)

year_ohe = OneHotEncoder(
    sparse_output=False,
    handle_unknown="ignore"
)

year_encoded = year_ohe.fit_transform(
    df_grouped[["Year"]]
)

mlb = MultiLabelBinarizer()

agent_matrix = mlb.fit_transform(
    df_grouped["Agent"]
)

role_matrix = np.vstack(
    df_grouped["Agent"].apply(
        get_role_vector
    )
)

role_scaler = StandardScaler()

role_scaled = role_scaler.fit_transform(
    role_matrix
)

X = np.hstack([
    year_encoded,
    team_encoded,
    map_encoded,
    agent_matrix,
    role_scaled
])

y = df_grouped["Winrate"].values

print()

print("[INFO] X Shape:", X.shape)
print("[INFO] y Shape:", y.shape)

print(
    "[INFO] Sample Ratio:",
    round(X.shape[0] / X.shape[1], 2)
)

print("\n===== AGENT COVERAGE =====")

agent_counter = {}

for agents in df_grouped["Agent"]:
    for agent in agents:
        agent_counter[agent] = agent_counter.get(agent, 0) + 1

coverage = (
    pd.Series(agent_counter)
    .sort_values()
)

print(coverage)

print("==========================\n")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = XGBRegressor(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

print("[INFO] Training XGBoost...")

model.fit(
    X_train,
    y_train
)

print("\n===== CROSS VALIDATION =====")

cv = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

scores = cross_val_score(
    model,
    X,
    y,
    cv=cv,
    scoring="neg_mean_absolute_error"
)

print("MAE per Fold:")
print(-scores)

print(f"\nMean MAE : {-scores.mean():.4f}")
print(f"Std MAE  : {scores.std():.4f}")

print("============================\n")

preds = model.predict(X_test)

mae = mean_absolute_error(
    y_test,
    preds
)

print()
print("[INFO] MAE:", round(mae, 4))

print("\n===== FEATURE IMPORTANCE =====")

feature_importance = pd.DataFrame({
    "Feature Index": np.arange(len(model.feature_importances_)),
    "Importance": model.feature_importances_
})

feature_importance = feature_importance.sort_values(
    "Importance",
    ascending=False
)

print(feature_importance.head(20))

print("==============================\n")

print()
print("[INFO] Saving artifacts...")

joblib.dump(
    model,
    MODEL_DIR / "xgb_model.pkl"
)

joblib.dump(
    team_ohe,
    MODEL_DIR / "team_ohe.pkl"
)

joblib.dump(
    map_ohe,
    MODEL_DIR / "map_ohe.pkl"
)

joblib.dump(
    year_ohe,
    MODEL_DIR / "year_ohe.pkl"
)

joblib.dump(
    mlb,
    MODEL_DIR / "mlb.pkl"
)

joblib.dump(
    role_scaler,
    MODEL_DIR / "role_scaler.pkl"
)

joblib.dump(
    AGENT_ROLE_MAP,
    MODEL_DIR / "agent_role_map.pkl"
)

print("[INFO] Done!")