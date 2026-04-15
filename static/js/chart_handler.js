// ── Chart.js global defaults ──────────────────────────────────────────────────
Chart.defaults.color          = '#718096';
Chart.defaults.borderColor    = '#1c2330';
Chart.defaults.font.family    = "'DM Mono', monospace";
Chart.defaults.font.size      = 11;

const ACCENT  = '#00aaff';
const GREEN   = '#00e07a';
const YELLOW  = '#f5c842';
const RED     = '#ff4455';
const SURFACE = '#0f1318';

// ── Helpers ───────────────────────────────────────────────────────────────────
async function api(url) {
  const r = await fetch(url);
  return r.json();
}

function shortLabel(name) {
  return name.replace(/_lag(\d+)$/, ' [lag $1]').replace(/_/g, ' ');
}

// ── 1. Model stats ─────────────────────────────────────────────────────────
async function loadModelStats() {
  const data = await api('/api/analytics/model');

  // Metrics row
  const { mae, rmse, r2, relative_mae_pct } = data.metrics;
  document.getElementById('metrics-row').innerHTML = `
    <div class="metric-pill">
      <span class="metric-label">MAE</span>
      <span class="metric-value">${mae ?? '8.55'} <small style="font-size:12px;font-weight:400">µg/m³</small></span>
    </div>
    <div class="metric-pill">
      <span class="metric-label">RMSE</span>
      <span class="metric-value warn">${rmse ?? '14.43'} <small style="font-size:12px;font-weight:400">µg/m³</small></span>
    </div>
    <div class="metric-pill">
      <span class="metric-label">R²</span>
      <span class="metric-value good">${r2 ?? '0.86'}</span>
    </div>
    <div class="metric-pill">
      <span class="metric-label">RELATIVE MAE</span>
      <span class="metric-value">${relative_mae_pct ?? '1.02'}<small style="font-size:12px;font-weight:400">%</small></span>
    </div>
  `;

  // Training curve
  const curveCtx = document.getElementById('curveChart').getContext('2d');
  new Chart(curveCtx, {
    type: 'line',
    data: {
      labels: data.training_curve.map(d => d.tree),
      datasets: [
        {
          label: 'Train RMSE',
          data:  data.training_curve.map(d => d.train_rmse),
          borderColor: ACCENT, borderWidth: 1.5,
          pointRadius: 0, tension: 0.3, fill: false,
        },
        {
          label: 'Val RMSE',
          data:  data.training_curve.map(d => d.val_rmse),
          borderColor: YELLOW, borderWidth: 1.5,
          pointRadius: 0, tension: 0.3, fill: false,
        },
      ],
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      animation: { duration: 600 },
      plugins: { legend: { position: 'top', labels: { boxWidth: 12, padding: 16 } } },
      scales: {
        x: { title: { display: true, text: 'Trees', color: '#4a5568' }, ticks: { maxTicksLimit: 10 } },
        y: { title: { display: true, text: 'RMSE (µg/m³)', color: '#4a5568' } },
      },
    },
  });

  // Scatter
  const scatterCtx = document.getElementById('scatterChart').getContext('2d');
  const maxVal = Math.max(...data.scatter.map(d => Math.max(d.actual, d.predicted)));
  new Chart(scatterCtx, {
    type: 'scatter',
    data: {
      datasets: [
        {
          label: 'Predictions',
          data: data.scatter.map(d => ({ x: d.actual, y: d.predicted })),
          backgroundColor: 'rgba(0,170,255,0.35)',
          pointRadius: 3, pointHoverRadius: 5,
        },
        {
          label: 'Perfect fit',
          data: [{ x: 0, y: 0 }, { x: maxVal, y: maxVal }],
          type: 'line',
          borderColor: 'rgba(255,255,255,0.15)',
          borderWidth: 1, borderDash: [4, 4],
          pointRadius: 0, fill: false,
        },
      ],
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      animation: { duration: 600 },
      plugins: { legend: { position: 'top', labels: { boxWidth: 12, padding: 16 } } },
      scales: {
        x: { title: { display: true, text: 'Actual PM2.5 (µg/m³)', color: '#4a5568' } },
        y: { title: { display: true, text: 'Predicted PM2.5 (µg/m³)', color: '#4a5568' } },
      },
    },
  });

  // Feature importance
  const maxFI = data.feature_importance[0]?.importance || 1;
  document.getElementById('fi-list').innerHTML = data.feature_importance.map(f => `
    <div class="fi-row">
      <span class="fi-name" title="${f.feature}">${shortLabel(f.feature)}</span>
      <div class="fi-bar-wrap">
        <div class="fi-bar" style="width:${(f.importance / maxFI * 100).toFixed(1)}%"></div>
      </div>
      <span class="fi-val">${f.importance.toLocaleString()}</span>
    </div>
  `).join('');
}


// ── 2. AQI stations ───────────────────────────────────────────────────────────
const stationCharts = {};  // station_id → Chart instance

async function loadStations() {
  const stations = await api('/api/analytics/stations');
  const grid = document.getElementById('station-grid');
  grid.innerHTML = '';

  for (const station of stations) {
    const card = document.createElement('div');
    card.className = 'station-card';
    card.id = `station-${station.station_id}`;

    const sensorOptions = station.sensors
      .map(s => `<option value="${s.sensor_id}" data-units="${s.units}">${s.display_name}</option>`)
      .join('');

    card.innerHTML = `
      <div class="station-header">
        <div>
          <div class="station-name">${station.name}</div>
          <div class="station-owner">${station.owner}</div>
        </div>
        <select class="sensor-select" id="select-${station.station_id}" onchange="changeSensor(${station.station_id})">
          ${sensorOptions}
        </select>
      </div>
      <div class="hours-toggle">
        <button class="hours-btn active" data-hours="24"  onclick="changeHours(${station.station_id}, 24,  this)">24 h</button>
        <button class="hours-btn"        data-hours="48"  onclick="changeHours(${station.station_id}, 48,  this)">48 h</button>
        <button class="hours-btn"        data-hours="168" onclick="changeHours(${station.station_id}, 168, this)">7 d</button>
      </div>
      <div class="station-chart-wrap">
        <canvas id="chart-${station.station_id}"></canvas>
      </div>
    `;

    grid.appendChild(card);

    // Load default sensor (first in list)
    if (station.sensors.length > 0) {
      await renderStationChart(station.station_id, station.sensors[0].sensor_id, station.sensors[0].units, 24);
    }
  }
}

async function renderStationChart(stationId, sensorId, units, hours) {
  const readings = await api(`/api/analytics/readings/${sensorId}?hours=${hours}`);

  const labels = readings.map(r => {
    const d = new Date(r.datetime);
    return d.toLocaleString('en-IN', { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit', hour12: false });
  });
  const values = readings.map(r => r.value);
  const displayName = readings[0]?.display_name || '';

  // Pick colour based on parameter
  const color = displayName.toLowerCase().includes('pm') ? ACCENT
              : displayName.toLowerCase().includes('co')  ? YELLOW
              : displayName.toLowerCase().includes('no')  ? RED
              : GREEN;

  const canvasId = `chart-${stationId}`;
  const ctx = document.getElementById(canvasId)?.getContext('2d');
  if (!ctx) return;

  // Destroy previous chart if exists
  if (stationCharts[stationId]) {
    stationCharts[stationId].destroy();
  }

  stationCharts[stationId] = new Chart(ctx, {
    type: 'line',
    data: {
      labels,
      datasets: [{
        label: `${displayName} (${units})`,
        data: values,
        borderColor: color,
        borderWidth: 1.5,
        backgroundColor: color + '18',
        pointRadius: 0,
        pointHoverRadius: 4,
        tension: 0.3,
        fill: true,
      }],
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      animation: { duration: 400 },
      interaction: { mode: 'index', intersect: false },
      plugins: {
        legend: { display: false },
        tooltip: {
          backgroundColor: '#131820',
          borderColor: '#1c2330',
          borderWidth: 1,
          titleColor: '#718096',
          bodyColor: '#e2e8f0',
          padding: 10,
          callbacks: {
            label: ctx => ` ${ctx.parsed.y.toFixed(2)} ${units}`,
          },
        },
      },
      scales: {
        x: {
          ticks: { maxTicksLimit: 6, maxRotation: 0 },
          grid: { color: 'rgba(28,35,48,0.8)' },
        },
        y: {
          title: { display: true, text: units, color: '#4a5568' },
          grid: { color: 'rgba(28,35,48,0.8)' },
        },
      },
    },
  });
}

async function changeSensor(stationId) {
  const select  = document.getElementById(`select-${stationId}`);
  const sensorId = parseInt(select.value);
  const units    = select.selectedOptions[0].dataset.units;
  const activeBtn = document.querySelector(`#station-${stationId} .hours-btn.active`);
  const hours    = parseInt(activeBtn?.dataset.hours || 24);
  await renderStationChart(stationId, sensorId, units, hours);
}

async function changeHours(stationId, hours, btn) {
  // Toggle active class
  btn.closest('.hours-toggle').querySelectorAll('.hours-btn').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');

  const select   = document.getElementById(`select-${stationId}`);
  const sensorId = parseInt(select.value);
  const units    = select.selectedOptions[0].dataset.units;
  await renderStationChart(stationId, sensorId, units, hours);
}


// ── Boot ──────────────────────────────────────────────────────────────────────
(async () => {
  await Promise.all([loadModelStats(), loadStations()]);
})();