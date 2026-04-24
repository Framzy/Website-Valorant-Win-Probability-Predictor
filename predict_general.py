# === predict_general.py ===
# Scoring heuristik untuk Casual mode
# Menghitung weighted score berdasarkan data historis agent-map
import joblib
import math

# Load pre-computed data
agent_map_weighted  = joblib.load("agent_map_weighted.pkl")
map_max_count       = joblib.load("map_max_count.pkl")
role_popular_agents = joblib.load("role_popular_agents.pkl")
popular_comps       = joblib.load("popular_comps.pkl")
casual_thresholds   = joblib.load("casual_thresholds.pkl")
AGENT_ROLE_MAP      = joblib.load("agent_role_map_general.pkl")

ROLE_ORDER = ["duelist", "initiator", "controller", "sentinel"]

# Minimum pick count agar agent dianggap "familiar" di map
MIN_FAMILIAR_COUNT = 3


def get_agent_score(agent, map_name):
    """
    Ambil weighted score untuk agent di map tertentu.
    Return (weighted_score, count, avg_wr) atau (0, 0, 0) jika tidak ada data.
    """
    data = agent_map_weighted.get((agent, map_name))
    if data:
        return data['weighted'], data['count'], data['avg_wr']
    return 0.0, 0, 0.0


def check_agent_familiarity(agent, map_name):
    """
    Cek apakah agent familiar di map ini.
    Return: (is_familiar, reason)
      - is_familiar=True  → agent pernah dipick >= MIN_FAMILIAR_COUNT di map
      - is_familiar=False → agent tidak pernah/jarang dipick
    """
    data = agent_map_weighted.get((agent, map_name))
    if data and data['count'] >= MIN_FAMILIAR_COUNT:
        return True, None
    return False, data['count'] if data else 0


def check_role_alternative(agent, map_name):
    """
    Cek apakah ada agent lain di role yang sama yang populer di map ini.
    Return: (has_role_presence, top_agent_name)
    """
    role = AGENT_ROLE_MAP.get(agent)
    if not role:
        return False, None
    popular = role_popular_agents.get((role, map_name), [])
    # Ada agent di role ini yang familiar?
    for pop_agent, count in popular:
        if count >= MIN_FAMILIAR_COUNT:
            return True, pop_agent
    return False, None


def calculate_casual_score(map_name, agents):
    """
    Hitung skor komposisi untuk Casual mode.

    Return dict:
      - score: final weighted score (0-1 scale, tapi max ~0.50)
      - agent_details: per-agent breakdown
      - penalty_details: list of penalty reasons
      - total_penalty: total penalty value
      - popular_comps: top 3 popular comps untuk map
      - map_data_count: total match data untuk map ini
    """
    agent_scores = []
    agent_details = []
    penalty_details = []
    total_penalty = 0.0

    # Role count untuk role-composition check
    role_count = {r: 0 for r in ROLE_ORDER}

    for ag in agents:
        ws, count, avg_wr = get_agent_score(ag, map_name)
        role = AGENT_ROLE_MAP.get(ag, "unknown")
        if role in role_count:
            role_count[role] += 1

        is_familiar, pick_count = check_agent_familiarity(ag, map_name)
        agent_name = ag.title()

        if is_familiar:
            # Agent sering dipick di map ini → langsung pakai score-nya
            agent_scores.append(ws)
            agent_details.append({
                "agent": agent_name,
                "role": role,
                "score": round(ws * 100, 1),
                "count": count,
                "status": "popular"
            })
        else:
            # Agent tidak familiar → cek apakah role-nya ada yang populer
            has_role, top_agent = check_role_alternative(ag, map_name)

            if has_role:
                # Role-nya ada di map, tapi agent ini jarang → penalti kecil
                # Gunakan score dari top agent di role yang sama, tapi dikurangi
                top_ws, _, _ = get_agent_score(top_agent, map_name)
                reduced_score = top_ws * 0.7  # 70% dari top agent score
                agent_scores.append(reduced_score)

                penalty_val = round((top_ws - reduced_score) * 100, 1)
                total_penalty += (top_ws - reduced_score)

                penalty_details.append({
                    "reason": f"{agent_name} jarang dipick di map ini "
                              f"(dipick {pick_count}x, biasanya {top_agent.title()})",
                    "value": f"-{penalty_val}"
                })
                agent_details.append({
                    "agent": agent_name,
                    "role": role,
                    "score": round(reduced_score * 100, 1),
                    "count": pick_count,
                    "status": "rare"
                })
            else:
                # Role tidak ada di map ATAU agent benar-benar asing
                # Penalti lebih besar
                fallback_score = 0.15  # skor fallback rendah
                agent_scores.append(fallback_score)

                penalty_details.append({
                    "reason": f"{agent_name} ({role}) sangat jarang digunakan di map ini",
                    "value": f"-{round((0.4 - fallback_score) * 100, 1)}"
                })
                agent_details.append({
                    "agent": agent_name,
                    "role": role,
                    "score": round(fallback_score * 100, 1),
                    "count": 0,
                    "status": "unknown"
                })

    # === Role Composition Penalties ===
    # Tanpa controller
    if role_count.get("controller", 0) == 0:
        pen = 0.03
        total_penalty += pen
        penalty_details.append({
            "reason": "Tidak ada Controller dalam komposisi",
            "value": f"-{pen*100:.0f}"
        })

    # Tanpa initiator
    if role_count.get("initiator", 0) == 0:
        pen = 0.02
        total_penalty += pen
        penalty_details.append({
            "reason": "Tidak ada Initiator dalam komposisi",
            "value": f"-{pen*100:.0f}"
        })

    # Role stacking (3+ agent di role yang sama)
    for role_name in ROLE_ORDER:
        cnt = role_count.get(role_name, 0)
        if cnt >= 3:
            pen = 0.03 * (cnt - 2)
            total_penalty += pen
            penalty_details.append({
                "reason": f"{cnt}x {role_name.title()} (role stacking)",
                "value": f"-{pen*100:.0f}"
            })

    # === Final Score ===
    import numpy as np
    base_score = np.mean(agent_scores) if agent_scores else 0.0
    final_score = max(0.0, base_score - total_penalty)

    # Map data count
    map_data = sum(1 for (a, m) in agent_map_weighted if m == map_name)

    # Popular comps
    pop_comps = get_popular_comps(map_name)

    return {
        "score": round(final_score * 100, 2),
        "base_score": round(base_score * 100, 2),
        "agent_details": agent_details,
        "penalty_details": penalty_details,
        "total_penalty": round(total_penalty * 100, 2),
        "popular_comps": pop_comps,
        "map_agent_count": map_data,
        "thresholds": {
            "p25": round(casual_thresholds['p25'] * 100, 1),
            "p50": round(casual_thresholds['p50'] * 100, 1),
            "p75": round(casual_thresholds['p75'] * 100, 1),
        }
    }


def get_popular_comps(map_name):
    """Ambil top 3 popular comps untuk map tertentu."""
    comps = popular_comps.get(map_name, [])
    formatted = []
    for c in comps:
        formatted.append({
            "agents": ", ".join(a.title() for a in c["agents"]),
            "count": c["count"],
            "avg_wr": c["avg_wr"]
        })
    return formatted


def describe_composition(agents):
    """Deskripsi komposisi role dari agents yang dipilih."""
    role_cnt = {r: 0 for r in ROLE_ORDER}
    for ag in agents:
        role = AGENT_ROLE_MAP.get(ag)
        if role in role_cnt:
            role_cnt[role] += 1

    parts = []
    for r in ROLE_ORDER:
        if role_cnt[r] > 0:
            parts.append(f"{role_cnt[r]} {r.title()}")
    return " / ".join(parts) if parts else "Unknown"
