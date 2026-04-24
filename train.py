# === train.py ===
import pandas as pd
import numpy as np
import random
import tensorflow as tf
import joblib
from sklearn.preprocessing import OneHotEncoder, MultiLabelBinarizer, StandardScaler
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_absolute_error
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
from collections import defaultdict

# Set seed
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

# Load data
print("[INFO] Loading data...")
df = pd.read_csv("valorant_dataset_all.csv")

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

# Filter: hanya match dengan tepat 5 agent (komposisi valid)
df_grouped["agent_count"] = df_grouped["Agent"].apply(len)
print(f"[INFO] Before agent filter: {len(df_grouped)} matches")
df_grouped = df_grouped[df_grouped["agent_count"] == 5].reset_index(drop=True)
print(f"[INFO] After agent filter (==5): {len(df_grouped)} matches")

# Filter: total played >= 5
df_grouped["Total Played"] = df_grouped["Total Wins By Map"] + df_grouped["Total Loss By Map"]
df_grouped = df_grouped[df_grouped["Total Played"] >= 5].reset_index(drop=True)
print(f"[INFO] After total played filter (>=5): {len(df_grouped)} matches")

# Winrate target
df_grouped["Winrate"] = df_grouped["Total Wins By Map"] / df_grouped["Total Played"]

# === FITUR BARU: Team overall winrate ===
team_wr = df_grouped.groupby("Team")["Winrate"].mean().to_dict()
df_grouped["team_overall_wr"] = df_grouped["Team"].map(team_wr)

# === Role historis per TIM+MAP (lebih akurat dari per-Tim saja) ===
# Key: ("Team Name", "Map Name") → rata-rata role vector
tim_map_role_hist = df_grouped.groupby(["Team", "Map"]).apply(
    lambda rows: np.mean([get_role_vector(ag) for ag in rows["Agent"]], axis=0)
).to_dict()

# Fallback: per-Tim saja (dipakai jika kombinasi Tim+Map tidak ada di data)
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

# Cosine similarity — gunakan Tim+Map key, fallback ke Tim saja
def cosine_sim(a, b):
    a = np.array(a, dtype=np.float64)
    b = np.array(b, dtype=np.float64)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)

def get_role_ref(team, map_name):
    """Ambil referensi role historis: Tim+Map dulu, fallback ke Tim."""
    key = (team, map_name)
    if key in tim_map_role_hist:
        return tim_map_role_hist[key]
    return tim_role_hist.get(team, [0.25]*4)

role_sim_scores = [cosine_sim(get_role_vector(row["Agent"]), get_role_ref(row["Team"], row["Map"]))
                   for _, row in df_grouped.iterrows()]
role_sim_scores = np.array(role_sim_scores).reshape(-1, 1)

# Fitur tambahan
team_overall_wr = df_grouped["team_overall_wr"].values.reshape(-1, 1)
total_played = df_grouped["Total Played"].values.reshape(-1, 1)

# Gabung fitur
X_raw = np.hstack([team_encoded, map_encoded, agent_matrix]).astype(np.float32)
role_scaler = StandardScaler()
role_scaled = role_scaler.fit_transform(role_matrix)

# Extra features scaler
extra_features = np.hstack([team_overall_wr, total_played])
extra_scaler = StandardScaler()
extra_scaled = extra_scaler.fit_transform(extra_features)

X = np.hstack([X_raw, role_scaled, role_sim_scores, extra_scaled]).astype(np.float32)
y = df_grouped["Winrate"].values.astype(np.float32)

print(f"[INFO] Feature matrix shape: {X.shape}")
print(f"[INFO] Target shape: {y.shape}")
print(f"[INFO] Sample:Feature ratio = {X.shape[0]}:{X.shape[1]} = {X.shape[0]/X.shape[1]:.2f}:1")

# === 5-Fold Cross Validation untuk evaluasi reliable ===
print("\n[INFO] Running 5-Fold Cross Validation...")
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_mae_scores = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
    X_cv_train, X_cv_val = X[train_idx], X[val_idx]
    y_cv_train, y_cv_val = y[train_idx], y[val_idx]
    
    cv_model = Sequential([
        Input(shape=(X.shape[1],)),
        Dense(32, activation="relu", kernel_regularizer=l2(0.01)),
        BatchNormalization(),
        Dropout(0.3),
        Dense(16, activation="relu", kernel_regularizer=l2(0.01)),
        Dropout(0.2),
        Dense(1, activation="sigmoid")
    ])
    cv_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                     loss="mse", metrics=["mae"])
    cv_model.fit(X_cv_train, y_cv_train, epochs=300, batch_size=8,
                 validation_data=(X_cv_val, y_cv_val),
                 callbacks=[EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True)],
                 verbose=0)
    
    y_cv_pred = cv_model.predict(X_cv_val, verbose=0).flatten()
    fold_mae = mean_absolute_error(y_cv_val, y_cv_pred)
    cv_mae_scores.append(fold_mae)
    print(f"  Fold {fold+1}: MAE = {fold_mae:.4f}")

print(f"\n[INFO] 5-Fold CV MAE: {np.mean(cv_mae_scores):.4f} +/- {np.std(cv_mae_scores):.4f}")

# === Train final model on all data with validation split ===
print("\n[INFO] Training final model...")
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)

model = Sequential([
    Input(shape=(X.shape[1],)),
    Dense(32, activation="relu", kernel_regularizer=l2(0.01)),
    BatchNormalization(),
    Dropout(0.3),
    Dense(16, activation="relu", kernel_regularizer=l2(0.01)),
    Dropout(0.2),
    Dense(1, activation="sigmoid")
])
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="mse",
    metrics=["mae"]
)

es = EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True)
lr_scheduler = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=8, min_lr=1e-5, verbose=1)

model.fit(Xtr, ytr, validation_split=0.15, epochs=300, batch_size=8,
          callbacks=[es, lr_scheduler], verbose=1)

# Evaluasi
print("\n[INFO] Evaluating model...")
y_pred = model.predict(Xte, verbose=0).flatten()
test_mae = mean_absolute_error(yte, y_pred)
print(f"[INFO] MAE test: {test_mae:.4f}")
print(f"[INFO] Prediction range: [{y_pred.min():.4f}, {y_pred.max():.4f}]")
print(f"[INFO] Actual range:     [{yte.min():.4f}, {yte.max():.4f}]")

# Count tim per data historis (untuk confidence calculation nanti)
team_sample_count = df_grouped["Team"].value_counts().to_dict()

# ===== GAUGE THRESHOLDS (Projection Layer) =====
# Jalankan model di seluruh data training untuk dapat distribusi prediksi sesungguhnya.
# P25/P50/P75 menjadi batas zona gauge — bukan hardcode 35/50/65.
print("\n[INFO] Calculating gauge thresholds from full dataset predictions...")
all_preds = model.predict(X, verbose=0).flatten()
gauge_thresholds = {
    "p25": float(np.percentile(all_preds, 25)),   # bottom 25% → Merah
    "p50": float(np.percentile(all_preds, 50)),   # median     → Oranye/Kuning boundary
    "p75": float(np.percentile(all_preds, 75)),   # top 25%   → Hijau
    "min": float(all_preds.min()),
    "max": float(all_preds.max()),
}
print(f"[INFO] Gauge zones → Merah < {gauge_thresholds['p25']*100:.1f}% | "
      f"Oranye < {gauge_thresholds['p50']*100:.1f}% | "
      f"Kuning < {gauge_thresholds['p75']*100:.1f}% | "
      f"Hijau ≥ {gauge_thresholds['p75']*100:.1f}%")

# Save
print("\n[INFO] Saving model and artifacts...")
model.save("jst_model.keras")
joblib.dump(role_scaler, "role_scaler.pkl")
joblib.dump(extra_scaler, "extra_scaler.pkl")
joblib.dump(team_ohe, "team_ohe.pkl")
joblib.dump(map_ohe, "map_ohe.pkl")
joblib.dump(mlb, "mlb.pkl")
joblib.dump(tim_role_hist, "role_mean_dict.pkl")
joblib.dump(tim_map_role_hist, "role_map_mean_dict.pkl")
joblib.dump(AGENT_ROLE_MAP, "agent_role_map.pkl")
joblib.dump(team_wr, "team_wr_dict.pkl")
joblib.dump(team_sample_count, "team_sample_count.pkl")
joblib.dump(gauge_thresholds, "gauge_thresholds.pkl")  # ← NEW: projection layer
print("[INFO] Done! All artifacts saved.")