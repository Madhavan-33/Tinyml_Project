// Client-Side Real-Time Inference & Vercel API Bridge
let modelData = null;
const NUM_SENSORS = 9;

// Physical sensor operating range for the tactile mesh system
const SENSOR_MIN = 1.60;    // V
const SENSOR_MAX = 2.28;    // V
const SENSOR_DEFAULT = 2.15; // V (dataset baseline)

// Default initial telemetry (realistic in-range values)
let sensorValues = [2.05, 1.95, 2.05, 2.00, 1.88, 2.00, 2.05, 1.95, 2.05];

// Preset Scenarios (physical range: 1.60V - 2.28V)
const PRESETS = {
  cube:          [2.00, 1.96, 2.01, 1.95, 1.88, 1.97, 2.02, 1.98, 2.01],
  cylinder:      [2.22, 1.88, 2.23, 2.21, 1.82, 2.22, 2.23, 1.89, 2.21],
  high_pressure: [1.75, 1.70, 1.75, 1.68, 1.62, 1.69, 1.74, 1.71, 1.75],
  light_touch:   [2.25, 2.22, 2.25, 2.24, 2.20, 2.24, 2.25, 2.22, 2.26]
};

// Initialize Application
document.addEventListener("DOMContentLoaded", async () => {
  renderSliderControls();
  setupCanvas();
  setupEventListeners();

  try {
    const res = await fetch("/model_weights.json");
    if (res.ok) {
      modelData = await res.json();
      console.log("Client-side TinyML model loaded successfully.");
    }
  } catch (err) {
    console.warn("Local model weights not found, will rely on serverless API:", err);
  }

  // Initial calculation
  evaluatePrediction();
  renderWaveform();
});

// Render 9 sensor sliders dynamically
function renderSliderControls() {
  const container = document.getElementById("sensorSlidersList");
  if (!container) return;

  container.innerHTML = "";

  for (let i = 1; i <= NUM_SENSORS; i++) {
    const val = Math.min(SENSOR_MAX, Math.max(SENSOR_MIN, sensorValues[i - 1]));
    sensorValues[i - 1] = val; // clamp stored value
    const row = document.createElement("div");
    row.className = "sensor-row";
    row.innerHTML = `
      <div class="sensor-row-header">
        <label for="slider_s${i}" class="sensor-label">
          <span class="sensor-label-badge">S${i}</span> Mesh Sensor ${i}
        </label>
        <input type="number" id="num_s${i}" class="sensor-number-input" 
               min="${SENSOR_MIN}" max="${SENSOR_MAX}" step="0.01" value="${val.toFixed(2)}" aria-label="Sensor ${i} Value">
      </div>
      <input type="range" id="slider_s${i}" min="${SENSOR_MIN}" max="${SENSOR_MAX}" step="0.01" value="${val}"
             aria-label="Sensor ${i} Range Slider">
      <div style="font-size:10px;color:#64748b;margin-top:2px">Range: ${SENSOR_MIN}V &ndash; ${SENSOR_MAX}V (physical operating range)</div>
    `;
    container.appendChild(row);

    const slider = row.querySelector(`#slider_s${i}`);
    const numInput = row.querySelector(`#num_s${i}`);

    slider.addEventListener("input", (e) => {
      const v = parseFloat(e.target.value);
      sensorValues[i - 1] = v;
      numInput.value = v.toFixed(2);
      onSensorDataChanged();
    });

    numInput.addEventListener("input", (e) => {
      let v = parseFloat(e.target.value);
      if (isNaN(v)) v = SENSOR_DEFAULT;
      v = Math.min(SENSOR_MAX, Math.max(SENSOR_MIN, v)); // clamp
      sensorValues[i - 1] = v;
      slider.value = v;
      onSensorDataChanged();
    });
  }
}

// When any slider or input changes
function onSensorDataChanged() {
  evaluatePrediction();
  renderWaveform();
  updateApiPreview();
}

// Client-Side Fast Inference Engine (<1ms)
function evaluatePrediction() {
  if (!modelData) {
    // If client weights not yet loaded, call serverless endpoint
    callServerlessApi();
    return;
  }

  const meta = modelData.metadata;
  const weights = modelData.weights;
  const classes = meta.shape_classes || ["cube", "cylinder"];

  // 1. Normalize via StandardScaler (z-score)
  const x_scaled = sensorValues.map((v, i) => (v - meta.sensor_means[i]) / meta.sensor_stds[i]);

  // 2. Dense 1 + ReLU
  const h1 = [];
  for (let j = 0; j < weights.b1.length; j++) {
    let sum = weights.b1[j];
    for (let i = 0; i < 9; i++) sum += x_scaled[i] * weights.w1[i][j];
    h1.push(Math.max(0, sum));
  }

  // 3. BatchNorm 1
  const bn1 = [];
  for (let j = 0; j < h1.length; j++) {
    const std = Math.sqrt(weights.bn_var[j] + weights.bn_eps);
    const norm = (h1[j] - weights.bn_mean[j]) / std;
    bn1.push(weights.bn_gamma[j] * norm + weights.bn_beta[j]);
  }

  // 4. Dense 2 + ReLU
  const h2_pre = [];
  for (let j = 0; j < weights.b2.length; j++) {
    let sum = weights.b2[j];
    for (let i = 0; i < bn1.length; i++) sum += bn1[i] * weights.w2[i][j];
    h2_pre.push(Math.max(0, sum));
  }

  // 5. BatchNorm 2
  let h2 = h2_pre;
  if (weights.bn2_gamma) {
    h2 = [];
    for (let j = 0; j < h2_pre.length; j++) {
      const std = Math.sqrt(weights.bn2_var[j] + weights.bn2_eps);
      const norm = (h2_pre[j] - weights.bn2_mean[j]) / std;
      h2.push(weights.bn2_gamma[j] * norm + weights.bn2_beta[j]);
    }
  }

  // 6. Dense 3 + ReLU
  const h3 = [];
  for (let j = 0; j < weights.b3.length; j++) {
    let sum = weights.b3[j];
    for (let i = 0; i < h2.length; i++) sum += h2[i] * weights.w3[i][j];
    h3.push(Math.max(0, sum));
  }

  // 7. Optional Dense 4 + ReLU
  let h_last = h3;
  if (weights.w4) {
    h_last = [];
    for (let j = 0; j < weights.b4.length; j++) {
      let sum = weights.b4[j];
      for (let i = 0; i < h3.length; i++) sum += h3[i] * weights.w4[i][j];
      h_last.push(Math.max(0, sum));
    }
  }

  // 8. Shape Head (Softmax)
  const logits = [];
  for (let j = 0; j < weights.b_shape.length; j++) {
    let sum = weights.b_shape[j];
    for (let i = 0; i < h_last.length; i++) sum += h_last[i] * weights.w_shape[i][j];
    logits.push(sum);
  }
  const maxL = Math.max(...logits);
  const expL = logits.map(l => Math.exp(l - maxL));
  const sumExp = expL.reduce((a, b) => a + b, 0);
  const probs = expL.map(e => e / sumExp);

  // 9. Dimension Head (Linear)
  let dim_mm = weights.b_dim[0];
  for (let i = 0; i < h_last.length; i++) dim_mm += h_last[i] * weights.w_dim[i][0];

  // Render results
  let bestIdx = 0;
  for (let i = 1; i < probs.length; i++) {
    if (probs[i] > probs[bestIdx]) bestIdx = i;
  }

  const probObj = {};
  classes.forEach((c, idx) => {
    probObj[c] = probs[idx];
  });

  renderResultsUI({
    predicted_shape: classes[bestIdx],
    confidence_percentage: (probs[bestIdx] * 100).toFixed(1),
    estimated_dimension_mm: Math.max(1.0, dim_mm).toFixed(2),
    probabilities: probObj,
    mode: "Client TinyML Engine (<1ms)"
  });
}

// Update UI Components
function renderResultsUI(data) {
  const shapeEl = document.getElementById("resShape");
  const confEl = document.getElementById("resConfidence");
  const sizeEl = document.getElementById("resSize");
  const latEl = document.getElementById("resLatency");

  if (shapeEl) shapeEl.textContent = (data.predicted_shape || "CUBE").toUpperCase();
  if (confEl) confEl.textContent = `Confidence: ${data.confidence_percentage}%`;
  if (sizeEl) sizeEl.textContent = `${data.estimated_dimension_mm} mm`;
  if (latEl) latEl.textContent = data.mode || "< 1 ms";

  // Handle probabilities
  const probs = data.shape_probabilities || data.probabilities || {};
  const pCube = (probs.cube !== undefined) ? (probs.cube * 100).toFixed(1) : ((probs.Cube !== undefined) ? (probs.Cube * 100).toFixed(1) : "100.0");
  const pCyl = (probs.cylinder !== undefined) ? (probs.cylinder * 100).toFixed(1) : ((probs.Cylinder !== undefined) ? (probs.Cylinder * 100).toFixed(1) : "0.0");

  const barCube = document.getElementById("barCube");
  const pctCube = document.getElementById("pctCube");
  if (barCube) barCube.style.width = `${pCube}%`;
  if (pctCube) pctCube.textContent = `${pCube}%`;

  const barCylinder = document.getElementById("barCylinder");
  const pctCylinder = document.getElementById("pctCylinder");
  if (barCylinder) barCylinder.style.width = `${pCyl}%`;
  if (pctCylinder) pctCylinder.textContent = `${pCyl}%`;
}

// Canvas Waveform / Radar Visualizer
let canvas, ctx;
function setupCanvas() {
  canvas = document.getElementById("sensorWaveCanvas");
  if (!canvas) return;
  ctx = canvas.getContext("2d");
  resizeCanvas();
  window.addEventListener("resize", resizeCanvas);
}

function resizeCanvas() {
  if (!canvas) return;
  const rect = canvas.getBoundingClientRect();
  canvas.width = rect.width * window.devicePixelRatio;
  canvas.height = rect.height * window.devicePixelRatio;
  ctx.scale(window.devicePixelRatio, window.devicePixelRatio);
  renderWaveform();
}

function renderWaveform() {
  if (!canvas || !ctx) return;
  const w = canvas.getBoundingClientRect().width;
  const h = canvas.getBoundingClientRect().height;

  ctx.clearRect(0, 0, w, h);

  // Background Grid Lines
  ctx.strokeStyle = "rgba(255, 255, 255, 0.05)";
  ctx.lineWidth = 1;
  for (let y = 20; y < h; y += 30) {
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(w, y);
    ctx.stroke();
  }

  // Draw 9-Sensor Waveform
  const paddingX = 35;
  const stepX = (w - paddingX * 2) / (NUM_SENSORS - 1);
  // Scale waveform to physical sensor range
  const waveMin = SENSOR_MIN;
  const waveMax = SENSOR_MAX;
  const waveRange = waveMax - waveMin;

  // Gradient fill under curve
  const grad = ctx.createLinearGradient(0, 0, 0, h);
  grad.addColorStop(0, "rgba(56, 189, 248, 0.35)");
  grad.addColorStop(1, "rgba(56, 189, 248, 0.0)");

  ctx.beginPath();
  sensorValues.forEach((val, i) => {
    const x = paddingX + i * stepX;
    const y = h - 25 - ((val - waveMin) / waveRange) * (h - 45);
    if (i === 0) ctx.moveTo(x, y);
    else {
      // Smooth bezier curves
      const prevX = paddingX + (i - 1) * stepX;
      const prevY = h - 25 - ((sensorValues[i - 1] - waveMin) / waveRange) * (h - 45);
      const cpX = (prevX + x) / 2;
      ctx.bezierCurveTo(cpX, prevY, cpX, y, x, y);
    }
  });

  // Stroke path
  ctx.strokeStyle = "#38bdf8";
  ctx.lineWidth = 3;
  ctx.shadowColor = "#38bdf8";
  ctx.shadowBlur = 10;
  ctx.stroke();

  // Draw node points
  sensorValues.forEach((val, i) => {
    const x = paddingX + i * stepX;
    const y = h - 25 - ((val - waveMin) / waveRange) * (h - 45);

    ctx.beginPath();
    ctx.arc(x, y, 5, 0, Math.PI * 2);
    ctx.fillStyle = "#0f172a";
    ctx.fill();
    ctx.strokeStyle = "#38bdf8";
    ctx.lineWidth = 2.5;
    ctx.stroke();

    // Text label
    ctx.shadowBlur = 0;
    ctx.fillStyle = "#94a3b8";
    ctx.font = "10px JetBrains Mono, monospace";
    ctx.textAlign = "center";
    ctx.fillText(`S${i + 1}`, x, h - 8);
  });
}

// Call Vercel Serverless Function (/api/predict)
async function callServerlessApi() {
  const btn = document.getElementById("btnTriggerApi");
  const responseBox = document.getElementById("apiResponseCode");
  if (btn) btn.textContent = "⏳ Invoking Serverless...";

  const t0 = performance.now();
  try {
    const res = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ sensors: sensorValues })
    });
    const t1 = performance.now();
    const data = await res.json();

    if (res.ok) {
      data.mode = `Vercel Lambda (${(t1 - t0).toFixed(0)} ms)`;
      renderResultsUI(data);
      if (responseBox) {
        responseBox.textContent = JSON.stringify(data, null, 2);
      }
    } else {
      if (responseBox) responseBox.textContent = JSON.stringify(data, null, 2);
    }
  } catch (err) {
    if (responseBox) responseBox.textContent = `Error connecting to /api/predict: ${err.message}`;
  } finally {
    if (btn) btn.textContent = "🚀 Send POST Request to /api/predict";
  }
}

// Update API cURL Code Block
function updateApiPreview() {
  const curlBox = document.getElementById("apiCurlCode");
  if (!curlBox) return;

  const jsonPayload = JSON.stringify({ sensors: sensorValues });
  curlBox.textContent = `curl -X POST https://your-deployment.vercel.app/api/predict \\
  -H "Content-Type: application/json" \\
  -d '${jsonPayload}'`;
}

// Preset and UI event listeners
function setupEventListeners() {
  // Preset buttons
  document.querySelectorAll("[data-preset]").forEach(btn => {
    btn.addEventListener("click", () => {
      const type = btn.getAttribute("data-preset");
      if (PRESETS[type]) {
        sensorValues = [...PRESETS[type]];
        renderSliderControls();
        onSensorDataChanged();
      }
    });
  });

  // Randomize button (keep values within physical sensor range)
  const btnRand = document.getElementById("btnRandomize");
  if (btnRand) {
    btnRand.addEventListener("click", () => {
      sensorValues = sensorValues.map(() => +(SENSOR_MIN + Math.random() * (SENSOR_MAX - SENSOR_MIN)).toFixed(2));
      renderSliderControls();
      onSensorDataChanged();
    });
  }

  // Trigger API Button
  const btnApi = document.getElementById("btnTriggerApi");
  if (btnApi) {
    btnApi.addEventListener("click", () => {
      callServerlessApi();
    });
  }
}
