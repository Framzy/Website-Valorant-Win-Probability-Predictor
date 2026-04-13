const BASE = "http://127.0.0.1:5000";
const maxChecked = 5;
let checkedOrder = [];

async function loadData() {
  // 1) Ambil data JSON
  const res = await fetch(`${BASE}/data.json`);
  const { teams, maps, agents } = await res.json();

  // 2) Populate team <select>
  const teamSelect = document.getElementById("team");
  const nt = document.getElementById("namaTeam");
  teams.forEach((t) => {
    const opt = new Option(t, t);
    teamSelect.appendChild(opt);
  });
  teamSelect.addEventListener(
    "change",
    () => (nt.innerText = teamSelect.value),
  );

  // 3) Populate map <select>
  const mapSelect = document.getElementById("map");
  const nm = document.getElementById("namaMap");
  maps.forEach((m) => {
    const opt = new Option(m, m);
    mapSelect.appendChild(opt);
  });
  mapSelect.addEventListener("change", () => (nm.innerText = mapSelect.value));

  // 4) Generate kartu Agent sesuai CSS .agent-item
  const agentContainer = document.getElementById("Agent");
  agents.forEach((a) => {
    const wrapper = document.createElement("div");
    wrapper.classList.add("agent-item");
    wrapper.dataset.name = a.name;

    // atur background-image via inline style
    wrapper.style.backgroundImage = `url("${a.url}")`;

    // judul nama agent
    const h1 = document.createElement("h1");
    h1.textContent = a.name.charAt(0).toUpperCase() + a.name.slice(1);
    wrapper.appendChild(h1);

    // klik untuk select / unselect
    wrapper.addEventListener("click", () => toggleAgent(a.name, wrapper));

    agentContainer.appendChild(wrapper);
  });
}

// toggle border & daftar pilihan
function toggleAgent(name, el) {
  const idx = checkedOrder.indexOf(name);
  if (idx > -1) {
    // unselect
    checkedOrder.splice(idx, 1);
    el.classList.remove("selected");
  } else {
    // select baru
    if (checkedOrder.length >= maxChecked) {
      // hapus yang pertama
      const first = checkedOrder.shift();
      const firstEl = document.querySelector(
        `.agent-item[data-name="${first}"]`,
      );
      firstEl.classList.remove("selected");
    }
    checkedOrder.push(name);
    el.classList.add("selected");
  }
  updateSelectedAgents();
}

// tampilkan daftar terpilih
function updateSelectedAgents() {
  document.getElementById("namaAgent").innerText = checkedOrder
    .map((n) => n.charAt(0).toUpperCase() + n.slice(1))
    .join(", ");
}

// ===== Error Popup Functions =====
function showErrorPopup(message) {
  document.getElementById("errorMessage").innerText = message;
  const overlay = document.getElementById("errorOverlay");
  const popup   = document.getElementById("errorPopup");
  overlay.style.display = "block";
  popup.style.display   = "flex";
  requestAnimationFrame(() => popup.classList.add("show"));
}

function closeErrorPopup() {
  const overlay = document.getElementById("errorOverlay");
  const popup   = document.getElementById("errorPopup");
  popup.classList.remove("show");
  setTimeout(() => {
    overlay.style.display = "none";
    popup.style.display   = "none";
  }, 250);
}

document.addEventListener("DOMContentLoaded", () => {
  loadData();

  // tombol prediksi
  document
    .querySelector('button[name="Predict"]')
    .addEventListener("click", async () => {
      const team = document.getElementById("team").value;
      const map  = document.getElementById("map").value;

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
      // ==========================


      // Tampilkan indikator loading
      const overlayEl = document.getElementById("overlay");
      const loadingEl = document.getElementById("loading");
      overlayEl.style.display = "block";
      loadingEl.style.display = "flex";

      // Kosongkan sementara hasil sebelumnya
      document.getElementById("pred").innerText = "...";
      document.getElementById("adjusted_pred").innerText = "...";
      document.getElementById("comp_desc").innerText = "Memproses...";
      document.getElementById("sim_score").innerText = "...";
      document.getElementById("most_common_combo").innerText = "...";

      // Judul info
      document.getElementById("sim_score_info").innerText =
        `Skor Kecocokan Dengan ${team} :`;
      document.getElementById("kombinasi_agent_info").innerText =
        `Kombinasi Terbaik Berdasarkan ${team} Pada Map ${map} :`;

      // Jeda loading buatan
      await new Promise((resolve) => setTimeout(resolve, 1000));

      // Kirim permintaan
      const response = await fetch(`${BASE}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ team, map, agents: checkedOrder }),
      });

      const result = await response.json();

      // Sembunyikan loading
      overlayEl.style.display = "none";
      loadingEl.style.display = "none";

      if (response.ok) {
        document.getElementById("pred").innerText = `${result.pred}%`;
        const adjEl = document.getElementById("adjusted_pred");
        adjEl.innerText =
          result.pred === result.adjusted_pred
            ? `${result.adjusted_pred}%`
            : `${result.adjusted_pred}% (Penalti)`;
        document.getElementById("comp_desc").innerText = result.comp_desc;
        document.getElementById("sim_score").innerText = `${result.sim_score}%`;
        document.getElementById("most_common_combo").innerText =
          result.most_common_combo;
      } else {
        showErrorPopup(result.error || "Terjadi kesalahan saat prediksi.");
      }
    });
});
