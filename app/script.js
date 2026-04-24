const BASE = "http://127.0.0.1:5000";
const maxChecked = 5;
let checkedOrder = [];
let agentDataCache = [];
let currentMode = "casual"; // default: casual

// Projection layer thresholds (updated from server response)
let gaugeThresholds = { p25: 40.0, p50: 50.0, p75: 60.0 };

async function loadData() {
  const res = await fetch(`${BASE}/data.json`);
  const { teams, maps, agents } = await res.json();
  agentDataCache = agents;

  // Populate team <select> (Pro Team mode only)
  const teamSelect = document.getElementById("team");
  const nt = document.getElementById("namaTeam");
  teams.forEach((t) => {
    const opt = new Option(t, t);
    teamSelect.appendChild(opt);
  });
  teamSelect.addEventListener("change", () => (nt.innerText = teamSelect.value));

  // Populate BOTH map <select> elements
  const mapSelectPro = document.getElementById("map");
  const mapSelectCasual = document.getElementById("mapCasual");
  const nm = document.getElementById("namaMap");
  const nmCasual = document.getElementById("namaMapCasual");

  maps.forEach((m) => {
    mapSelectPro.appendChild(new Option(m, m));
    mapSelectCasual.appendChild(new Option(m, m));
  });
  mapSelectPro.addEventListener("change", () => (nm.innerText = mapSelectPro.value));
  mapSelectCasual.addEventListener("change", () => (nmCasual.innerText = mapSelectCasual.value));

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

// ===== Mode Tab Switching =====
function setupModeTabs() {
  const tabs = document.querySelectorAll(".mode-tab");
  tabs.forEach((tab) => {
    tab.addEventListener("click", () => {
      const mode = tab.dataset.mode;
      if (mode === currentMode) return;

      // Update active tab
      tabs.forEach((t) => t.classList.remove("active"));
      tab.classList.add("active");
      currentMode = mode;

      // Toggle panels
      toggleModePanels(mode);

      // Hide output when switching tabs
      const outputEl = document.getElementById("outputPredict");
      outputEl.classList.remove("visible");
      resetGauge();
    });
  });
}

function toggleModePanels(mode) {
  // Casual panels
  const casualPanels = [
    document.getElementById("formCasual"),
    document.getElementById("pickCasual"),
  ];
  // Pro Team panels
  const proTeamPanels = [
    document.getElementById("formProTeam"),
    document.getElementById("pickProTeam"),
  ];

  if (mode === "casual") {
    casualPanels.forEach((p) => p.classList.add("active"));
    proTeamPanels.forEach((p) => p.classList.remove("active"));
  } else {
    casualPanels.forEach((p) => p.classList.remove("active"));
    proTeamPanels.forEach((p) => p.classList.add("active"));
  }
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
  const text = checkedOrder.length
    ? checkedOrder.map((n) => n.charAt(0).toUpperCase() + n.slice(1)).join(", ")
    : "—";

  // Update both casual and pro team displays
  const agentCasual = document.getElementById("namaAgentCasual");
  const agentPro = document.getElementById("namaAgent");
  if (agentCasual) agentCasual.innerText = text;
  if (agentPro) agentPro.innerText = text;
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

// ===== Reset Functions =====
function resetCasual() {
  // Reset agents
  checkedOrder = [];
  document.querySelectorAll(".agent-item.selected").forEach((el) => el.classList.remove("selected"));
  updateSelectedAgents();
  updateAgentCounter();

  // Reset map
  document.getElementById("mapCasual").value = "";
  document.getElementById("namaMapCasual").innerText = "—";

  // Reset role filter
  resetRoleFilter();

  // Hide output
  hideOutput();
}

function resetProTeam() {
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

  // Reset role filter
  resetRoleFilter();

  // Hide output
  hideOutput();
}

function resetRoleFilter() {
  const tabs = document.querySelectorAll(".role-tab");
  tabs.forEach((t) => t.classList.remove("active"));
  tabs[0].classList.add("active");
  document.querySelectorAll(".agent-item").forEach((item) => item.classList.remove("role-hidden"));
}

function hideOutput() {
  const outputEl = document.getElementById("outputPredict");
  outputEl.classList.remove("visible");
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

// ===== Gauge Zone Colors =====
function getZoneColor(pct) {
  if (pct >= gaugeThresholds.p75) return "#27ae60";
  if (pct >= gaugeThresholds.p50) return "#f39c12";
  if (pct >= gaugeThresholds.p25) return "#e67e22";
  return "#e74c3c";
}

function getZoneGlow(pct) {
  if (pct >= gaugeThresholds.p75) return "rgba(39, 174, 96, 0.55)";
  if (pct >= gaugeThresholds.p50) return "rgba(243, 156, 18, 0.55)";
  if (pct >= gaugeThresholds.p25) return "rgba(230, 126, 34, 0.55)";
  return "rgba(231, 76, 60, 0.55)";
}

function getGaugeFillRatio(pct) {
  if (pct >= gaugeThresholds.p75) return 1.00;
  if (pct >= gaugeThresholds.p50) return 0.75;
  if (pct >= gaugeThresholds.p25) return 0.50;
  return 0.25;
}

function getGaugeStatusText(pct) {
  if (pct >= gaugeThresholds.p75) return "Sangat Tinggi";
  if (pct >= gaugeThresholds.p50) return "Tinggi";
  if (pct >= gaugeThresholds.p25) return "Moderat";
  return "Rendah";
}

function getGaugeSubText(pct) {
  if (pct >= gaugeThresholds.p75) return "Peluang menang besar";
  if (pct >= gaugeThresholds.p50) return "Peluang menang tinggi";
  if (pct >= gaugeThresholds.p25) return "Peluang menang seimbang";
  return "Peluang menang kecil";
}

const ARC_LENGTH = 251.33;

function animateGauge(targetPct) {
  const fill = document.getElementById("gaugeFill");
  const statusEl = document.getElementById("gaugeStatus");
  const subEl = document.getElementById("gaugeSubText");

  const pct = Math.max(0, Math.min(100, targetPct));

  const color = getZoneColor(pct);
  const glow = getZoneGlow(pct);
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
  const subEl = document.getElementById("gaugeSubText");

  fill.style.stroke = "var(--red)";
  fill.style.filter = "drop-shadow(0 0 8px rgba(255, 70, 85, 0.5))";
  fill.style.strokeDashoffset = ARC_LENGTH;

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

// ===== Render Popular Comps (Casual Mode) =====
function renderPopularComps(comps) {
  const list = document.getElementById("popularCompsList");
  list.innerHTML = "";

  if (!comps || comps.length === 0) {
    const empty = document.createElement("p");
    empty.className = "penalty-none";
    empty.textContent = "Tidak ada data komposisi populer untuk map ini.";
    list.appendChild(empty);
    return;
  }

  comps.forEach((comp, i) => {
    const card = document.createElement("div");
    card.className = "popular-comp-card";

    const rank = document.createElement("div");
    rank.className = "popular-comp-rank";
    rank.textContent = `#${i + 1}`;

    const info = document.createElement("div");
    info.className = "popular-comp-info";

    const agents = document.createElement("div");
    agents.className = "popular-comp-agents";
    agents.textContent = comp.agents;

    const meta = document.createElement("div");
    meta.className = "popular-comp-meta";
    meta.textContent = `Dipakai ${comp.count}× oleh tim pro`;

    info.appendChild(agents);
    info.appendChild(meta);

    const wr = document.createElement("div");
    wr.className = "popular-comp-wr";
    wr.textContent = `${comp.avg_wr}%`;

    card.appendChild(rank);
    card.appendChild(info);
    card.appendChild(wr);
    list.appendChild(card);
  });
}

// ===== Show/Hide output sections based on mode =====
function configureOutputForMode(mode) {
  const comboSection = document.getElementById("comboSection");
  const popularSection = document.getElementById("popularCompsSection");

  if (mode === "casual") {
    comboSection.style.display = "none";
    popularSection.style.display = "block";
    // Casual labels
    document.getElementById("gaugeLabel").innerText = "Skor Komposisi Agent";
    document.getElementById("pred_label").innerText = "Skor Final";
    document.getElementById("confidence_label").innerText = "Data di Map";
    document.getElementById("sim_score_info").innerText = "Skor Sebelum Penalti";
  } else {
    comboSection.style.display = "block";
    popularSection.style.display = "none";
    // Pro Team labels
    document.getElementById("gaugeLabel").innerText = "Final Win Probability";
    document.getElementById("pred_label").innerText = "Raw Probability";
    document.getElementById("confidence_label").innerText = "Tingkat Kepercayaan";
    document.getElementById("sim_score_info").innerText = "Skor Kecocokan Historis";
  }
}

// ===== Predict: Casual Mode =====
async function predictCasual() {
  const map = document.getElementById("mapCasual").value;

  // Validasi
  if (!map && checkedOrder.length < 5) {
    showErrorPopup("Silakan pilih Map dan 5 Agent sebelum melakukan prediksi.");
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

  showLoading();
  resetGauge();
  document.querySelectorAll(".reveal-item").forEach((el) => el.classList.remove("revealed"));

  // Update info labels for casual mode
  document.getElementById("sim_score_info").innerText = `Skor Sebelum Penalti`;

  try {
    const response = await fetch(`${BASE}/predict_general`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ map, agents: checkedOrder }),
    });

    const result = await response.json();
    hideLoading();

    if (response.ok) {
      configureOutputForMode("casual");
      const outputEl = document.getElementById("outputPredict");
      outputEl.classList.add("visible");

      if (result.gauge_thresholds) {
        gaugeThresholds = result.gauge_thresholds;
      }

      setTimeout(() => animateGauge(result.adjusted_pred), 200);

      // Stat cards: adapt labels for casual
      document.getElementById("pred").innerText = `${result.adjusted_pred}%`;
      document.getElementById("confidence").innerText = `${result.map_agent_count} agents`;
      document.getElementById("sim_score").innerText = `${result.base_score}%`;
      document.getElementById("comp_desc").innerText = result.comp_desc;

      renderPenaltyDetails(result.penalty_details);
      renderPopularComps(result.popular_comps);

      setTimeout(() => revealResults(), 300);
      setTimeout(() => {
        outputEl.scrollIntoView({ behavior: "smooth", block: "center" });
      }, 600);
    } else {
      showErrorPopup(result.error || "Terjadi kesalahan saat prediksi.");
    }
  } catch (err) {
    hideLoading();
    showErrorPopup("Tidak dapat terhubung ke server. Pastikan server berjalan.");
  }
}

// ===== Predict: Pro Team Mode =====
async function predictProTeam() {
  const team = document.getElementById("team").value;
  const map = document.getElementById("map").value;

  // Validasi
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

  showLoading();
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
    hideLoading();

    if (response.ok) {
      configureOutputForMode("proteam");
      const outputEl = document.getElementById("outputPredict");
      outputEl.classList.add("visible");

      if (result.gauge_thresholds) {
        gaugeThresholds = result.gauge_thresholds;
      }

      setTimeout(() => animateGauge(result.adjusted_pred), 200);

      document.getElementById("pred").innerText = `${result.pred}%`;
      document.getElementById("confidence").innerText = `${result.confidence}%`;
      document.getElementById("sim_score").innerText = `${result.sim_score}%`;
      document.getElementById("comp_desc").innerText = result.comp_desc;
      document.getElementById("most_common_combo").innerText = result.most_common_combo;

      renderPenaltyDetails(result.penalty_details);

      setTimeout(() => revealResults(), 300);
      setTimeout(() => {
        outputEl.scrollIntoView({ behavior: "smooth", block: "center" });
      }, 600);
    } else {
      showErrorPopup(result.error || "Terjadi kesalahan saat prediksi.");
    }
  } catch (err) {
    hideLoading();
    showErrorPopup("Tidak dapat terhubung ke server. Pastikan server berjalan.");
  }
}

// ===== Loading helpers =====
function showLoading() {
  document.getElementById("overlay").style.display = "block";
  document.getElementById("loading").style.display = "flex";
}

function hideLoading() {
  document.getElementById("overlay").style.display = "none";
  document.getElementById("loading").style.display = "none";
}

// ===== Main DOMContentLoaded =====
document.addEventListener("DOMContentLoaded", () => {
  loadData();
  setupModeTabs();

  // Casual predict & reset
  document.getElementById("btnPredictCasual").addEventListener("click", predictCasual);
  document.getElementById("btnResetCasual").addEventListener("click", resetCasual);

  // Pro Team predict & reset
  document.getElementById("btnPredict").addEventListener("click", predictProTeam);
  document.getElementById("btnReset").addEventListener("click", resetProTeam);
});
