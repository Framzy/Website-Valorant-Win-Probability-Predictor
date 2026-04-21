# === predict.py ===
import joblib
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

model          = load_model("jst_model.keras")
role_scaler    = joblib.load("role_scaler.pkl")
extra_scaler   = joblib.load("extra_scaler.pkl")
team_ohe       = joblib.load("team_ohe.pkl")
map_ohe        = joblib.load("map_ohe.pkl")
mlb            = joblib.load("mlb.pkl")
AGENT_ROLE_MAP = joblib.load("agent_role_map.pkl")
role_mean_dict     = joblib.load("role_mean_dict.pkl")      # per-Tim (fallback)
role_map_mean_dict = joblib.load("role_map_mean_dict.pkl")  # per-Tim+Map (akurat)
team_wr_dict   = joblib.load("team_wr_dict.pkl")
team_sample_count = joblib.load("team_sample_count.pkl")

ROLE_ORDER = ["duelist", "initiator", "controller", "sentinel"]

def get_role_vector(agents, as_array=True):
    """Hitung jumlah agent per role."""
    cnt = {r: 0 for r in ROLE_ORDER}
    for ag in agents:
        role = AGENT_ROLE_MAP.get(ag)
        if role in cnt:
            cnt[role] += 1
    vec = [cnt[r] for r in ROLE_ORDER]
    return np.array(vec) if as_array else vec

def cosine_sim(a, b):
    """Cosine similarity antara dua vector."""
    a = np.array(a, dtype=np.float64)
    b = np.array(b, dtype=np.float64)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)

def calculate_penalty_details(role_raw_vec):
    """
    Hitung penalti untuk komposisi agent anomali.
    Returns: (total_penalty, list of {reason, value} dicts)
    Digunakan di predict.py CLI dan app.py web — SATU SUMBER KEBENARAN.
    """
    details = []
    max_same_role = max(role_raw_vec)
    
    # Cari role dominan
    dominant_role = ROLE_ORDER[role_raw_vec.tolist().index(max_same_role)] if hasattr(role_raw_vec, 'tolist') else ROLE_ORDER[list(role_raw_vec).index(max_same_role)]
    
    # Penalti progresif untuk role stacking (terlalu banyak role yang sama)
    if max_same_role == 3:
        details.append({"reason": f"3 agent role sama ({dominant_role})", "value": -10})
    elif max_same_role == 4:
        details.append({"reason": f"4 agent role sama ({dominant_role})", "value": -20})
    elif max_same_role == 5:
        details.append({"reason": f"5 agent role sama ({dominant_role})", "value": -35})
    
    # Penalti jika tidak ada controller (smoke sangat penting)
    if role_raw_vec[ROLE_ORDER.index("controller")] == 0:
        details.append({"reason": "Tidak ada controller (smoke)", "value": -10})
    
    # Penalti jika tidak ada initiator (info gathering penting)
    if role_raw_vec[ROLE_ORDER.index("initiator")] == 0:
        details.append({"reason": "Tidak ada initiator (info)", "value": -5})
    
    total = sum(d["value"] for d in details) / 100.0  # convert to 0-1 scale
    return abs(total), details

def calculate_penalty(role_raw_vec):
    """Backward-compatible wrapper — returns only total penalty."""
    total, _ = calculate_penalty_details(role_raw_vec)
    return total

def get_role_ref(team, map_name):
    """Ambil referensi role historis: Tim+Map dulu, fallback ke Tim saja."""
    key = (team, map_name)
    if key in role_map_mean_dict:
        return role_map_mean_dict[key]
    return role_mean_dict.get(team, [0.25]*4)

def calculate_confidence(team, map_name, sim_score):
    """
    Hitung confidence factor (0-1) berdasarkan:
    - Jumlah data historis tim
    - Cosine similarity score (Tim+Map specific)
    """
    # Berapa banyak data historis tim (max clamp di 12)
    n_samples = team_sample_count.get(team, 0)
    data_confidence = min(n_samples / 12.0, 1.0)

    # Cosine similarity sebagai proxy kecocokan pola
    sim_confidence = max(0, min(sim_score, 1.0))

    # Gabungkan: 60% data confidence, 40% similarity confidence
    confidence = 0.6 * data_confidence + 0.4 * sim_confidence
    return confidence

def moderate_prediction(raw_pred, confidence_factor):
    """
    Blend prediksi model dengan prior 50% berdasarkan confidence.
    Semakin rendah confidence, semakin dekat ke 50%.
    """
    prior = 0.5
    return prior + (raw_pred - prior) * confidence_factor

def prepare_input(team, map_name, agents):
    """Siapkan input vector untuk model (konsisten dengan train.py)."""
    df = pd.DataFrame({"Team": [team], "Map": [map_name]})
    team_vec = team_ohe.transform(df[["Team"]])
    map_vec  = map_ohe.transform(df[["Map"]])
    agent_vec = mlb.transform([agents])

    # Role vector — gunakan role_scaler (SAMA seperti training)
    role_raw = get_role_vector(agents)
    role_scaled = role_scaler.transform(role_raw.reshape(1, -1))

    # Cosine similarity — gunakan Tim+Map reference (AKURAT, fallback ke Tim)
    ref_vec = get_role_ref(team, map_name)
    sim_score = cosine_sim(role_raw, ref_vec)
    sim_arr = np.array([[sim_score]])

    # Fitur tambahan: team overall winrate & total_played (estimasi median)
    team_wr = team_wr_dict.get(team, 0.5)
    total_played_est = 5.0
    extra_raw = np.array([[team_wr, total_played_est]])
    extra_scaled = extra_scaler.transform(extra_raw)

    x = np.hstack([team_vec, map_vec, agent_vec, role_scaled, sim_arr, extra_scaled]).astype(np.float32)
    return x, role_raw, sim_score

def describe_composition(vec_raw):
    """Deskripsi komposisi role."""
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
        pred = float(model.predict(x, verbose=0)[0,0])
        # sigmoid sudah memastikan 0-1, tapi clip untuk safety
        pred = np.clip(pred, 0, 1)

        # Penalti untuk komposisi anomali
        penalty = calculate_penalty(role_raw_vec)
        adjusted_pred = np.clip(pred - penalty, 0, 1)
        
        # Moderasi berdasarkan confidence
        confidence = calculate_confidence(team, map_, sim_score)
        moderated_pred = moderate_prediction(adjusted_pred, confidence)
        
        comp_desc = describe_composition(role_raw_vec)

        print(f"\nPrediksi Raw Model: {pred:.4f} ({pred*100:.2f}%)")
        print(f"Penalti Komposisi: -{penalty:.4f}")
        print(f"Prediksi Setelah Penalti: {adjusted_pred:.4f} ({adjusted_pred*100:.2f}%)")
        print(f"Confidence Factor: {confidence:.4f}")
        print(f"Prediksi Final (Moderated): {moderated_pred:.4f} ({moderated_pred*100:.2f}%)")
        print(f"Komposisi Role: {comp_desc}")
        print(f"Skor Kecocokan dengan Pola Historis Tim: {sim_score:.4f}")

        if sim_score > 0.8:
            print("[INFO] Komposisi sangat cocok dengan pola historis.")
        elif sim_score > 0.5:
            print("[INFO] Komposisi cukup relevan, namun tidak identik.")
        else:
            print("[INFO] Komposisi berbeda jauh dari pola tim.")
            
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