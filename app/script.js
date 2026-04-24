const BASE = "http://127.0.0.1:5000";
const maxChecked = 5;
let checkedOrder = [];
let agentDataCache = [];

// Projection layer: threshold dari data aktual model (di-update dari response server)
// Default fallback jika server belum kirim thresholds
let gaugeThresholds = { p25: 35.0, p50: 50.0, p75: 65.0 };

async function loadData() {
  const res = await fetch(`${BASE}/data.json`);
  const { teams, maps, agents } = await res.json();
  agentDataCache = agents;

  // Populate team <select>
  const teamSelect = document.getElementById("team");
  const nt = document.getElementById("namaTeam");
  teams.forEach((t) => {
    const opt = new Option(t, t);
    teamSelect.appendChild(opt);
  });
  teamSelect.addEventListener("change", () => (nt.innerText = teamSelect.value));

  // Populate map <select>
  const mapSelect = document.getElementById("map");
  const nm = document.getElementById("namaMap");
  maps.forEach((m) => {
    const opt = new Option(m, m);
    mapSelect.appendChild(opt);
  });
  mapSelect.addEventListener("change", () => (nm.innerText = mapSelect.value));

  // Generate agent cards with role badges
  const agentContainer = document.getElementById("Agent");
  agents.forEach((a) => {
    const wrapper = document.createElement("div");
    wrapper.classList.add("agent-item");
    wrapper.dataset.name = a.name;
    wrapper.dataset.role = a.role;
    wrapper.style.backgroundImage = `url("${a.url}")`;

    // Role badge
    const badge = document.createElement("span");
    badge.classList.add("role-badge", a.role);
    badge.textContent = a.role.charAt(0).toUpperCase() + a.role.slice(1);
    wrapper.appendChild(badge);

    // Agent name
    const h1 = document.createElement("h1");
    h1.textContent = a.name.charAt(0).toUpperCase() + a.name.slice(1);
    wrapper.appendChild(h1);

    // Click handler
    wrapper.addEventListener("click", () => toggleAgent(a.name, wrapper));
    agentContainer.appendChild(wrapper);
  });

  // Role filter tabs
  setupRoleFilter();
}

// ===== Role Filter =====
function setupRoleFilter() {
  const tabs = document.querySelectorAll(".role-tab");
  tabs.forEach((tab) => {
    tab.addEventListener("click", () => {
      // Update active tab
      tabs.forEach((t) => t.classList.remove("active"));
      tab.classList.add("active");

      const role = tab.dataset.role;
      const items = document.querySelectorAll(".agent-item");

      items.forEach((item) => {
        if (role === "all" || item.dataset.role === role) {
          item.classList.remove("role-hidden");
        } else {
          item.classList.add("role-hidden");
        }
      });
    });
  });
}

// ===== Toggle Agent Selection =====
function toggleAgent(name, el) {
  const idx = checkedOrder.indexOf(name);
  if (idx > -1) {
    checkedOrder.splice(idx, 1);
    el.classList.remove("selected");
  } else {
    if (checkedOrder.length >= maxChecked) {
      const first = checkedOrder.shift();
      const firstEl = document.querySelector(`.agent-item[data-name="${first}"]`);
      firstEl.classList.remove("selected");
    }
    checkedOrder.push(name);
    el.classList.add("selected");
  }
  updateSelectedAgents();
  updateAgentCounter();
}

function updateSelectedAgents() {
  document.getElementById("namaAgent").innerText = checkedOrder.length
    ? checkedOrder.map((n) => n.charAt(0).toUpperCase() + n.slice(1)).join(", ")
    : "—";
}

function updateAgentCounter() {
  const counter = document.getElementById("agentCounter");
  counter.textContent = `${checkedOrder.length} / ${maxChecked} Agent Dipilih`;

  if (checkedOrder.length === maxChecked) {
    counter.classList.add("complete");
  } else {
    counter.classList.remove("complete");
  }
}

// ===== Reset All =====
function resetAll() {
  // Reset agents
  checkedOrder = [];
  document.querySelectorAll(".agent-item.selected").forEach((el) => el.classList.remove("selected"));
  updateSelectedAgents();
  updateAgentCounter();

  // Reset dropdowns
  document.getElementById("team").value = "";
  document.getElementById("map").value = "";
  document.getElementById("namaTeam").innerText = "—";
  document.getElementById("namaMap").innerText = "—";

  // Reset role filter to "All"
  const tabs = document.querySelectorAll(".role-tab");
  tabs.forEach((t) => t.classList.remove("active"));
  tabs[0].classList.add("active");
  document.querySelectorAll(".agent-item").forEach((item) => item.classList.remove("role-hidden"));

  // Hide output
  const outputEl = document.getElementById("outputPredict");
  outputEl.classList.remove("visible");

  // Reset gauge
  resetGauge();
}

// ===== Error Popup =====
function showErrorPopup(message) {
  document.getElementById("errorMessage").innerText = message;
  const overlay = document.getElementById("errorOverlay");
  const popup = document.getElementById("errorPopup");
  overlay.style.display = "block";
  popup.style.display = "flex";
  requestAnimationFrame(() => popup.classList.add("show"));
}

function closeErrorPopup() {
  const overlay = document.getElementById("errorOverlay");
  const popup = document.getElementById("errorPopup");
  popup.classList.remove("show");
  setTimeout(() => {
    overlay.style.display = "none";
    popup.style.display = "none";
  }, 250);
}

// ===== Gauge Zone Colors — pakai threshold dari data aktual =====
function getZoneColor(pct) {
  if (pct >= gaugeThresholds.p75) return "#27ae60";  // top 25%   → Hijau
  if (pct >= gaugeThresholds.p50) return "#f39c12";  // 50-75%    → Kuning
  if (pct >= gaugeThresholds.p25) return "#e67e22";  // 25-50%    → Oranye
  return "#e74c3c";                                  // bottom 25% → Merah
}

function getZoneGlow(pct) {
  if (pct >= gaugeThresholds.p75) return "rgba(39, 174, 96, 0.55)";
  if (pct >= gaugeThresholds.p50) return "rgba(243, 156, 18, 0.55)";
  if (pct >= gaugeThresholds.p25) return "rgba(230, 126, 34, 0.55)";
  return "rgba(231, 76, 60, 0.55)";
}

function getGaugeFillRatio(pct) {
  // Fixed zone fill — level visual per kuarter
  if (pct >= gaugeThresholds.p75) return 1.00;   // Hijau  → full
  if (pct >= gaugeThresholds.p50) return 0.75;   // Kuning → 3/4
  if (pct >= gaugeThresholds.p25) return 0.50;   // Oranye → 1/2
  return 0.25;                                   // Merah  → 1/4
}

function getGaugeStatusText(pct) {
  if (pct >= 65) return "Sangat Tinggi";
  if (pct >= 55) return "Tinggi";
  if (pct >= 45) return "Moderat";
  if (pct >= 35) return "Rendah";
  return "Sangat Rendah";
}

function getGaugeSubText(pct) {
  if (pct >= 65) return "Peluang menang besar";
  if (pct >= 55) return "Peluang menang tinggi";
  if (pct >= 45) return "Peluang menang seimbang";
  if (pct >= 35) return "Peluang menang rendah";
  return "Peluang menang kecil";
}

// Semi-circle arc length = π * r = π * 80 ≈ 251.33
const ARC_LENGTH = 251.33;

function animateGauge(targetPct) {
  const fill = document.getElementById("gaugeFill");
  const statusEl = document.getElementById("gaugeStatus");
  const subEl    = document.getElementById("gaugeSubText");

  const pct = Math.max(0, Math.min(100, targetPct));

  const color     = getZoneColor(pct);
  const glow      = getZoneGlow(pct);
  const fillRatio = getGaugeFillRatio(pct);

  fill.style.stroke = color;
  fill.style.filter = `drop-shadow(0 0 10px ${glow})`;
  fill.style.strokeDashoffset = ARC_LENGTH * (1 - fillRatio);

  statusEl.textContent = getGaugeStatusText(pct);
  statusEl.style.color = color;
  subEl.textContent = getGaugeSubText(pct);
}

function resetGauge() {
  const fill = document.getElementById("gaugeFill");
  const statusEl = document.getElementById("gaugeStatus");
  const subEl    = document.getElementById("gaugeSubText");

  fill.style.stroke = "var(--red)";
  fill.style.filter = "drop-shadow(0 0 8px rgba(255, 70, 85, 0.5))";
  fill.style.strokeDashoffset = ARC_LENGTH;  // arc hidden = empty state

  statusEl.textContent = "—";
  statusEl.style.color = "";
  subEl.textContent = "Pilih dan prediksi";
}

// ===== Staggered Reveal =====
function revealResults() {
  const items = document.querySelectorAll(".reveal-item");
  items.forEach((item, i) => {
    item.classList.remove("revealed");
    setTimeout(() => item.classList.add("revealed"), 150 * i);
  });
}

// ===== Render Penalty Details =====
function renderPenaltyDetails(details) {
  const list = document.getElementById("penaltyList");
  list.innerHTML = "";

  if (!details || details.length === 0) {
    const okEl = document.createElement("p");
    okEl.className = "penalty-none penalty-ok";
    okEl.textContent = "✅ Komposisi seimbang — tidak ada penalti";
    list.appendChild(okEl);
    return;
  }

  details.forEach((d) => {
    const item = document.createElement("div");
    item.className = "penalty-item";

    const icon = document.createElement("span");
    icon.className = "penalty-icon";
    icon.textContent = "⚠️";

    const reason = document.createElement("span");
    reason.className = "penalty-reason";
    reason.textContent = d.reason;

    const val = document.createElement("span");
    val.className = "penalty-val";
    val.textContent = `${d.value}%`;

    item.appendChild(icon);
    item.appendChild(reason);
    item.appendChild(val);
    list.appendChild(item);
  });
}

// ===== Main DOMContentLoaded =====
document.addEventListener("DOMContentLoaded", () => {
  loadData();

  // Predict button
  document.getElementById("btnPredict").addEventListener("click", async () => {
    const team = document.getElementById("team").value;
    const map = document.getElementById("map").value;

    // ===== VALIDASI INPUT =====
    if (!team && !map && checkedOrder.length < 5) {
      showErrorPopup("Silakan pilih Team, Map, dan 5 Agent sebelum melakukan prediksi.");
      return;
    }
    if (!team) {
      showErrorPopup("Team belum dipilih!\nSilakan pilih salah satu team terlebih dahulu.");
      return;
    }
    if (!map) {
      showErrorPopup("Map belum dipilih!\nSilakan pilih salah satu map terlebih dahulu.");
      return;
    }
    if (checkedOrder.length < 5) {
      showErrorPopup(`Agent yang dipilih masih kurang!\nDipilih: ${checkedOrder.length}/5 agent. Pilih ${5 - checkedOrder.length} agent lagi.`);
      return;
    }

    // Show loading
    const overlayEl = document.getElementById("overlay");
    const loadingEl = document.getElementById("loading");
    overlayEl.style.display = "block";
    loadingEl.style.display = "flex";

    // Reset output state before new prediction
    resetGauge();
    document.querySelectorAll(".reveal-item").forEach((el) => el.classList.remove("revealed"));

    // Update info labels
    document.getElementById("sim_score_info").innerText = `Kecocokan Dengan ${team}`;
    document.getElementById("kombinasi_agent_info").innerText = `Kombinasi Terbaik ${team} Pada Map ${map}`;

    try {
      const response = await fetch(`${BASE}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ team, map, agents: checkedOrder }),
      });

      const result = await response.json();

      // Hide loading
      overlayEl.style.display = "none";
      loadingEl.style.display = "none";

      if (response.ok) {
        // Show output with animation
        const outputEl = document.getElementById("outputPredict");
        outputEl.classList.add("visible");

        // Update projection layer thresholds dari server (data-calibrated)
        if (result.gauge_thresholds) {
          gaugeThresholds = result.gauge_thresholds;
        }

        // Animate gauge (sekarang pakai threshold dari data)
        setTimeout(() => animateGauge(result.adjusted_pred), 200);

        // Fill stat values
        document.getElementById("pred").innerText = `${result.pred}%`;
        document.getElementById("confidence").innerText = `${result.confidence}%`;
        document.getElementById("sim_score").innerText = `${result.sim_score}%`;
        document.getElementById("comp_desc").innerText = result.comp_desc;
        document.getElementById("most_common_combo").innerText = result.most_common_combo;

        // Render penalty details
        renderPenaltyDetails(result.penalty_details);

        // Staggered reveal
        setTimeout(() => revealResults(), 300);

        // Smooth scroll to output
        setTimeout(() => {
          outputEl.scrollIntoView({ behavior: "smooth", block: "center" });
        }, 600);
      } else {
        showErrorPopup(result.error || "Terjadi kesalahan saat prediksi.");
      }
    } catch (err) {
      overlayEl.style.display = "none";
      loadingEl.style.display = "none";
      showErrorPopup("Tidak dapat terhubung ke server. Pastikan server berjalan.");
    }
  });

  // Reset button
  document.getElementById("btnReset").addEventListener("click", resetAll);
});
