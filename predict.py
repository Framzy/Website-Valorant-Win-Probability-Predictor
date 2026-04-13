# === predict.py ===
import joblib
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

model         = load_model("jst_model.keras")
role_scaler   = joblib.load("role_scaler.pkl")
team_ohe      = joblib.load("team_ohe.pkl")
map_ohe       = joblib.load("map_ohe.pkl")
mlb           = joblib.load("mlb.pkl")
AGENT_ROLE_MAP= joblib.load("agent_role_map.pkl")
role_mean_dict = joblib.load("role_mean_dict.pkl")
global_best_combos= joblib.load("global_best_combos.pkl")


ROLE_ORDER = ["duelist", "initiator", "controller", "sentinel"]

def get_role_vector(agents, normalize=True):
    cnt = {r: 0 for r in ROLE_ORDER}
    for ag in agents:
        role = AGENT_ROLE_MAP.get(ag)
        if role in cnt:
            cnt[role] += 1
    vec = np.array([cnt[r] for r in ROLE_ORDER])
    return vec / 5 if normalize else vec

def cosine_sim(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)

def prepare_input(team, map_name, agents):
    df = pd.DataFrame({"Team": [team], "Map": [map_name]})
    team_vec = team_ohe.transform(df[["Team"]])
    map_vec  = map_ohe.transform(df[["Map"]])
    agent_vec = mlb.transform([agents])
    role_norm = get_role_vector(agents).reshape(1, -1)
    role_raw  = get_role_vector(agents, normalize=False)
    sim_score = cosine_sim(get_role_vector(agents), role_mean_dict.get(team, [0.25]*4))
    sim_score = np.array([[sim_score]])
    x = np.hstack([team_vec, map_vec, agent_vec, role_norm, sim_score]).astype(np.float32)
    return x, role_raw, sim_score[0,0]

def describe_composition(vec_raw):
    dominant = [ROLE_ORDER[i] for i, v in enumerate(vec_raw) if v >= 2]
    if not dominant:
        return "Balanced"
    return f"Dominant roles: {', '.join(dominant)}"

if __name__ == "__main__":
    print("=== Prediksi Winrate Berdasarkan Komposisi Agent ===")
    team = input("Nama Team: ").strip()
    map_ = input("Nama Map: ").strip()
    agents = [a.strip().lower() for a in input("5 Agent (pisah koma): ").split(',')]

    if len(agents)!=5 or any(a not in AGENT_ROLE_MAP for a in agents):
        print("[ERROR] Input agent tidak valid.")
    elif team not in role_mean_dict:
        print("[ERROR] Tim tidak ditemukan dalam data historis.")
    else:
        x, role_raw_vec, sim_score = prepare_input(team, map_, agents)
        pred = float(model.predict(x)[0,0])
        pred = np.clip(pred, 0, 1)

        # Penalti progresif
        penalty = 0
        max_same_role = max(role_raw_vec)
        if max_same_role == 3:
            penalty += 0.1
        elif max_same_role == 4:
            penalty += 0.2
        elif max_same_role == 5:
            penalty += 0.3

        if role_raw_vec[ROLE_ORDER.index("sentinel")] == 0:
            penalty += 0.1
        if role_raw_vec[ROLE_ORDER.index("controller")] == 0:
            penalty += 0.1
        if role_raw_vec.sum() != 5:
            penalty += 0.05

        adjusted_pred = np.clip(pred - penalty, 0, 1)
        comp_desc = describe_composition(role_raw_vec)

        print(f"\nPrediksi Winrate (sebelum penalti): {pred:.4f}")
        print(f"Prediksi Winrate (setelah penalti): {adjusted_pred:.4f}")
        print(f"Komposisi Role: {comp_desc}")
        print(f"Skor Kecocokan dengan Pola Historis Tim: {sim_score:.4f}")

        if sim_score > 0.8:
            print("[INFO] Komposisi sangat cocok dengan pola historis.")
        elif sim_score > 0.5:
            print("[INFO] Komposisi cukup relevan, namun tidak identik.")
        else:
            print("[INFO] Komposisi berbeda jauh dari pola tim.")
            
            print("\n=== Global Top 3 Kombinasi Terbaik untuk Map ini ===")
                
        print("\nKombinasi Historis Terbaik Tim di Map Ini:")
        try:
            df_hist = pd.read_csv("valorant_dataset_all.csv")
            df_hist["match_id"] = (
                df_hist['Tournament'] + "_" + df_hist['Stage'] + "_" +
                df_hist['Match Type'] + "_" + df_hist['Map'] + "_" + df_hist['Team']
            )
            # Filter hanya yang sesuai team dan map
            df_filtered = df_hist[
                (df_hist["Team"].str.lower() == team.lower()) &
                (df_hist["Map"].str.lower() == map_.lower())
            ]

            # Ambil kombinasi agent per match
            combos = []
            for _, group in df_filtered.groupby("match_id"):
                if len(group) == 5:
                    combo = tuple(sorted(group["Agent"].str.lower()))
                    combos.append(combo)

            if combos:
                from collections import Counter
                counter = Counter(combos)
                most_common_combo = counter.most_common(1)[0][0]
                print(f"→ Kombinasi: {', '.join(most_common_combo).title()}")
            else:
                print("⚠️ Tidak ditemukan kombinasi historis untuk team & map ini.")
        except Exception as e:
            print(f"[ERROR] Tidak dapat memuat data historis: {e}")