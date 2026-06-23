from flask import Flask, render_template, request, jsonify
from predict import (
    prepare_input, describe_composition, model,
    calculate_penalty_details, calculate_agent_mismatch_penalty,
    calculate_confidence, moderate_prediction,
    AGENT_ROLE_MAP, role_mean_dict
)

from predict_general import (
    calculate_casual_score, describe_composition as describe_composition_gen,
    AGENT_ROLE_MAP as AGENT_ROLE_MAP_GEN
)

from historical_service import (
    get_team_map_stats
)

import numpy as np
import joblib

from config import GAUGE_THRESHOLDS_PATH


# Load gauge thresholds (projection layer)
try:
    _gt = joblib.load(GAUGE_THRESHOLDS_PATH)
    GAUGE_THRESHOLDS = {
        "p25": round(_gt["p25"] * 100, 1),
        "p50": round(_gt["p50"] * 100, 1),
        "p75": round(_gt["p75"] * 100, 1),
    }
    print(f"[INFO] Gauge thresholds loaded: {GAUGE_THRESHOLDS}")
except Exception:
    GAUGE_THRESHOLDS = {"p25": 35.0, "p50": 50.0, "p75": 65.0}
    print("[WARN] gauge_thresholds.pkl not found, using defaults.")

app = Flask(
    __name__,
    static_folder='./../frontend/app',       # ← serve semua file statis dari folder 'app'
    static_url_path='',        # ← supaya CSS/JS/JSON bisa diakses di root URL
    template_folder='./../frontend/app'      # ← cari index.html di folder 'app'
)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    
    if not data:
        return jsonify(
            error="Request JSON tidak valid"
        ), 400

    team = data.get("team")
    map_ = data.get("map")
    agents = data.get("agents")

    if not team or not map_ or not agents:
        return jsonify(
            error="Data tidak lengkap"
        ), 400

    agents = [a.lower() for a in agents]

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
    most_common_combo, historical_agents_set = get_team_map_stats(
        team,
        map_
    )

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
        total_penalty=round(total_penalty*100, 2),
        gauge_thresholds=GAUGE_THRESHOLDS
    )


@app.route('/predict_general', methods=['POST'])
def predict_general_route():
    data = request.get_json()
    
    if not data:
        return jsonify(
            error="Request JSON tidak valid"
        ), 400
    
    map_ = data.get("map")
    agents = data.get("agents")

    if not map_ or not agents:
        return jsonify(
            error="Data tidak lengkap"
        ), 400

    agents = [a.lower() for a in agents]

    # Validasi
    if len(agents) != 5 or any(a not in AGENT_ROLE_MAP_GEN for a in agents):
        return jsonify(error="Input agent tidak valid"), 400

    # Hitung skor heuristik
    result = calculate_casual_score(map_, agents)
    comp = describe_composition_gen(agents)

    return jsonify(
        adjusted_pred=result['score'],
        base_score=result['base_score'],
        comp_desc=comp,
        agent_details=result['agent_details'],
        penalty_details=result['penalty_details'],
        total_penalty=result['total_penalty'],
        popular_comps=result['popular_comps'],
        map_agent_count=result['map_agent_count'],
        gauge_thresholds=result['thresholds']
    )

@app.errorhandler(Exception)
def handle_error(error):
    print(f"[ERROR] {error}")

    return jsonify(
        error="Terjadi kesalahan pada server"
    ), 500
if __name__ == '__main__':
    app.run(debug=True)
