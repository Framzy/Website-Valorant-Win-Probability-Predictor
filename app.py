from flask import Flask, render_template, request, jsonify
from predict import (
    prepare_input, describe_composition, model,
    calculate_penalty_details, calculate_agent_mismatch_penalty,
    calculate_confidence, moderate_prediction,
    AGENT_ROLE_MAP, role_mean_dict, ROLE_ORDER
)
import numpy as np
import pandas as pd

# Pre-load data historis sekali saat startup (bukan setiap request)
print("[INFO] Pre-loading historical data...")
try:
    df_hist = pd.read_csv("valorant_dataset_all.csv")
    df_hist["match_id"] = (
        df_hist['Tournament'] + "_" + df_hist['Stage'] + "_" +
        df_hist['Match Type'] + "_" + df_hist['Map'] + "_" + df_hist['Team']
    )
    HIST_DATA_LOADED = True
    print("[INFO] Historical data loaded successfully.")
except Exception as e:
    HIST_DATA_LOADED = False
    print(f"[WARN] Could not load historical data: {e}")

app = Flask(
    __name__,
    static_folder='app',       # ← serve semua file statis dari folder 'app'
    static_url_path='',        # ← supaya CSS/JS/JSON bisa diakses di root URL
    template_folder='app'      # ← cari index.html di folder 'app'
)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    team = data['team']
    map_ = data['map']
    agents = [a.lower() for a in data['agents']]

    # validasi singkat
    if len(agents) != 5 or any(a not in AGENT_ROLE_MAP for a in agents):
        return jsonify(error="Input agent tidak valid"), 400
    if team not in role_mean_dict:
        return jsonify(error="Team tidak ditemukan"), 400

    x, role_vec, sim_score = prepare_input(team, map_, agents)
    pred = float(model.predict(x, verbose=0)[0,0])
    pred = np.clip(pred, 0, 1)

    # ===== Role composition penalty =====
    role_penalty, penalty_details = calculate_penalty_details(role_vec)
    comp = describe_composition(role_vec)

    # ===== Historical data: combo + agent pool =====
    most_common_combo = "Tidak ditemukan"
    historical_agents_set = set()  # semua agent pernah dipakai tim di map ini

    if HIST_DATA_LOADED:
        try:
            df_filtered = df_hist[
                (df_hist["Team"].str.lower() == team.lower()) &
                (df_hist["Map"].str.lower() == map_.lower())
            ]
            combos = []
            for _, g in df_filtered.groupby("match_id"):
                if len(g) == 5:
                    combo = tuple(sorted(g["Agent"].str.lower()))
                    combos.append(combo)
                    historical_agents_set.update(combo)

            if combos:
                from collections import Counter
                most_common = Counter(combos).most_common(1)[0][0]
                most_common_combo = ", ".join(most_common).title()
        except Exception:
            most_common_combo = "Error memuat combo"

    # ===== Agent mismatch penalty (minor, -2% per agent di luar rotasi) =====
    mismatch_penalty, mismatch_details = calculate_agent_mismatch_penalty(
        agents, historical_agents_set
    )

    # ===== Gabungkan penalti & moderasi =====
    total_penalty = np.clip(role_penalty + mismatch_penalty, 0, 1)
    all_penalty_details = penalty_details + mismatch_details

    adj = np.clip(pred - total_penalty, 0, 1)
    confidence = calculate_confidence(team, map_, sim_score)
    moderated = moderate_prediction(adj, confidence)

    return jsonify(
        pred=round(pred*100, 2),
        adjusted_pred=round(moderated*100, 2),
        comp_desc=comp,
        sim_score=round(sim_score*100, 2),
        confidence=round(confidence*100, 2),
        most_common_combo=most_common_combo,
        penalty_details=all_penalty_details,
        total_penalty=round(total_penalty*100, 2)
    )
    

if __name__ == '__main__':
    app.run(debug=True)
