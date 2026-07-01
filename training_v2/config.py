"""
Configuration for Valorant Predictor V2 Training
"""

from pathlib import Path

# ==========================================================
# PATHS
# ==========================================================

BASE_DIR = Path(__file__).resolve().parent

DATASET_PATH = (
    BASE_DIR
    / "dataset"
    / "valorant_dataset_all.csv"
)
DATASET_DIR = (
    BASE_DIR
    / "dataset"
)
TEAM_DATASET_PATH = (
    BASE_DIR
    / "dataset"
    / "valorant_dataset_team_v2.csv"
)

MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# ==========================================================
# RANDOM
# ==========================================================

RANDOM_STATE = 42

# ==========================================================
# DATASET
# ==========================================================


TRAIN_TEST_SPLIT = 0.20

CV_FOLDS = 5

FEATURE_COLUMNS = [
    "Team",
    "Map",
    "Year",
    "Agent",
]

TARGET_COLUMN = "Winrate"

# ==========================================================
# XGBOOST
# ==========================================================

XGB_PARAMS = {

    "n_estimators": 300,

    "max_depth": 6,

    "learning_rate": 0.05,

    "subsample": 0.8,

    "colsample_bytree": 0.8,

    "random_state": RANDOM_STATE

}

# ==========================================================
# ROLE ORDER
# ==========================================================

ROLE_ORDER = [
    "duelist",
    "initiator",
    "controller",
    "sentinel"
]

# ==========================================================
# AGENT ROLE MAP
# ==========================================================

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
    "breach": "initiator",
    "fade": "initiator",
    "gekko": "initiator",
    "kayo": "initiator",
    "skye": "initiator",
    "sova": "initiator",
    "tejo": "initiator",

    # Controller
    "astra": "controller",
    "brimstone": "controller",
    "clove": "controller",
    "harbor": "controller",
    "miks": "controller",
    "omen": "controller",
    "viper": "controller",

    # Sentinel
    "chamber": "sentinel",
    "cypher": "sentinel",
    "deadlock": "sentinel",
    "killjoy": "sentinel",
    "sage": "sentinel",
    "veto": "sentinel",
    "vyse": "sentinel"

}