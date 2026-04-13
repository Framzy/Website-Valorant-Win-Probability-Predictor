# === train.py ===
import pandas as pd
import numpy as np
import random
import tensorflow as tf
import joblib
from sklearn.preprocessing import OneHotEncoder, MultiLabelBinarizer, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from collections import defaultdict

# Set seed
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

# Load data
print("[INFO] Loading data...")
df = pd.read_csv("valorant_dataset_all.csv")
df["Total Played"] = df["Total Wins By Map"] + df["Total Loss By Map"]
df = df[df["Total Played"] >= 5].reset_index(drop=True)

# Agent-role mapping
AGENT_ROLE_MAP = {
    **{a: "duelist" for a in ["iso", "jett", "raze", "reyna", "yoru", "neon", "phoenix", "waylay"]},
    **{a: "initiator" for a in ["skye", "sova", "breach", "fade", "kayo", "gekko", "tejo"]},
    **{a: "controller" for a in ["brimstone", "omen", "viper", "astra", "harbor", "clove"]},
    **{a: "sentinel" for a in ["killjoy", "cypher", "sage", "chamber", "deadlock", "vyse"]},
}
ROLE_ORDER = ["duelist", "initiator", "controller", "sentinel"]

# Role vector
def get_role_vector(agents):
    count = {r: 0 for r in ROLE_ORDER}
    for a in agents:
        role = AGENT_ROLE_MAP.get(a)
        if role:
            count[role] += 1
    return [count[r] for r in ROLE_ORDER]

# Aggregate per match
print("[INFO] Aggregating data by match...")
group_cols = ["Tournament", "Stage", "Match Type", "Map", "Team"]
df_grouped = df.groupby(group_cols).agg({
    "Agent": lambda x: list(x),
    "Total Wins By Map": "first",
    "Total Loss By Map": "first"
}).reset_index()
df_grouped["Winrate"] = df_grouped["Total Wins By Map"] / (
    df_grouped["Total Wins By Map"] + df_grouped["Total Loss By Map"])
    
# Role historis tim
tim_role_hist = df_grouped.groupby("Team")["Agent"].apply(
    lambda rows: np.mean([get_role_vector(ag) for ag in rows], axis=0)).to_dict()

# Encode team, map, agent
team_ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
map_ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
team_encoded = team_ohe.fit_transform(df_grouped[["Team"]])
map_encoded = map_ohe.fit_transform(df_grouped[["Map"]])
unique_agents = sorted({a for lst in df_grouped["Agent"] for a in lst})
mlb = MultiLabelBinarizer(classes=unique_agents)
agent_matrix = mlb.fit_transform(df_grouped["Agent"])

# Role matrix
role_matrix = np.vstack(df_grouped["Agent"].apply(get_role_vector))

# Cosine similarity
def cosine_sim(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)

role_sim_scores = [cosine_sim(get_role_vector(row["Agent"]), tim_role_hist.get(row["Team"], [1.25]*4))
                   for _, row in df_grouped.iterrows()]
role_sim_scores = np.array(role_sim_scores).reshape(-1, 1)

# Gabung fitur
X_raw = np.hstack([team_encoded, map_encoded, agent_matrix]).astype(np.float32)
role_scaler = StandardScaler()
role_scaled = role_scaler.fit_transform(role_matrix)
X = np.hstack([X_raw, role_scaled, role_sim_scores])
y = df_grouped["Winrate"].values.astype(np.float32)

# Train model
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
model = Sequential([
    Input(shape=(X.shape[1],)),
    Dense(256, activation="relu"), BatchNormalization(), Dropout(0.4),
    Dense(128, activation="relu"), Dropout(0.3),
    Dense(64, activation="relu"), Dropout(0.2),
    Dense(1, activation="linear")
])
model.compile(optimizer="adam", loss="mse", metrics=["mae"])
es = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
model.fit(Xtr, ytr, validation_split=0.15, epochs=150, batch_size=16, callbacks=[es], verbose=1)

# Evaluasi
print("[INFO] Evaluating model...")
y_pred = model.predict(Xte).flatten()
print(f"[INFO] MAE test: {mean_absolute_error(yte, y_pred):.4f}")

# Save
model.save("jst_model.keras")
joblib.dump(role_scaler, "role_scaler.pkl")
joblib.dump(team_ohe, "team_ohe.pkl")
joblib.dump(map_ohe, "map_ohe.pkl")
joblib.dump(mlb, "mlb.pkl")
joblib.dump(tim_role_hist, "role_mean_dict.pkl")
joblib.dump(AGENT_ROLE_MAP, "agent_role_map.pkl")