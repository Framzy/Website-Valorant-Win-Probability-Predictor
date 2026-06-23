from pathlib import Path

# BASE DIRECTORY

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"

# DATASET

DATASET_PATH = BASE_DIR / "dataset/valorant_dataset_all.csv"

# MODEL

MODEL_PATH = MODEL_DIR / "jst_model.keras"

# SCALER

ROLE_SCALER_PATH = MODEL_DIR / "role_scaler.pkl"
EXTRA_SCALER_PATH = MODEL_DIR / "extra_scaler.pkl"

# ENCODER

TEAM_OHE_PATH = MODEL_DIR / "team_ohe.pkl"
MAP_OHE_PATH = MODEL_DIR / "map_ohe.pkl"
MLB_PATH = MODEL_DIR / "mlb.pkl"

# AGENT ROLE

AGENT_ROLE_MAP_PATH = MODEL_DIR / "agent_role_map.pkl"
AGENT_ROLE_MAP_GENERAL_PATH = MODEL_DIR / "agent_role_map_general.pkl"

# ROLE STATISTICS

ROLE_MEAN_DICT_PATH = MODEL_DIR / "role_mean_dict.pkl"
ROLE_MAP_MEAN_DICT_PATH = MODEL_DIR / "role_map_mean_dict.pkl"

# TEAM STATISTICS

TEAM_WR_DICT_PATH = MODEL_DIR / "team_wr_dict.pkl"
TEAM_SAMPLE_COUNT_PATH = MODEL_DIR / "team_sample_count.pkl"

# GAUGE

GAUGE_THRESHOLDS_PATH = MODEL_DIR / "gauge_thresholds.pkl"

# GENERAL PREDICTOR

AGENT_MAP_WEIGHTED_PATH = MODEL_DIR / "agent_map_weighted.pkl"
MAP_MAX_COUNT_PATH = MODEL_DIR / "map_max_count.pkl"
ROLE_POPULAR_AGENTS_PATH = MODEL_DIR / "role_popular_agents.pkl"
POPULAR_COMPS_PATH = MODEL_DIR / "popular_comps.pkl"
CASUAL_THRESHOLDS_PATH = MODEL_DIR / "casual_thresholds.pkl"