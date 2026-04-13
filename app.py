from flask import Flask, render_template, request, jsonify
from predict import prepare_input, describe_composition, model
import numpy as np
import pandas as pd

# Load mappings dari predict.py
from predict import AGENT_ROLE_MAP, role_mean_dict, ROLE_ORDER

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
    pred = float(model.predict(x)[0,0])
    pred = np.clip(pred, 0, 1)

    # penalti sederhana (sama dengan di predict.py)
    penalty = 0
    mx = max(role_vec)
    if mx == 3: penalty += 0.2
    if mx == 4: penalty += 0.3
    if mx == 5: penalty += 0.4
    if role_vec[ROLE_ORDER.index("controller")] == 0: penalty += 0.1
    if role_vec.sum() != 5: penalty += 0.05

    adj = np.clip(pred - penalty, 0, 1)
    comp = describe_composition(role_vec)

    # Hitung Most Common Combo
    try:
        df_hist = pd.read_csv("valorant_dataset_all.csv")
        df_hist["match_id"] = (
            df_hist['Tournament'] + "_" + df_hist['Stage'] + "_" +
            df_hist['Match Type'] + "_" + df_hist['Map'] + "_" + df_hist['Team']
        )
        df_filtered = df_hist[
            (df_hist["Team"].str.lower() == team.lower()) &
            (df_hist["Map"].str.lower() == map_.lower())
        ]

        combos = []
        for _, g in df_filtered.groupby("match_id"):
            if len(g) == 5:
                combo = tuple(sorted(g["Agent"].str.lower()))
                combos.append(combo)

        if combos:
            from collections import Counter
            most_common = Counter(combos).most_common(1)[0][0]
            most_common_combo = ", ".join(most_common).title()
        else:
            most_common_combo = "Tidak ditemukan"
    except Exception:
        most_common_combo = "Error memuat combo"


    return jsonify(
        pred=round(pred*100,2),
        adjusted_pred=round(adj*100,2),
        comp_desc=comp,
        sim_score=round(sim_score*100,2),
        most_common_combo=most_common_combo
    )
    

if __name__ == '__main__':
    app.run(debug=True)
