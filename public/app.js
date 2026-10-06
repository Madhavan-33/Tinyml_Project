// Client-Side Real-Time Inference & Vercel API Bridge
let modelData = null;
const NUM_SENSORS = 9;

// Default initial telemetry
let sensorValues = [1.19, 4.47, 1.98, 2.90, 5.30, 3.68, 4.47, 6.14, 5.42];

// Preset Scenarios
const PRESETS = {
  beam: [1.19, 4.47, 1.98, 2.90, 5.30, 3.68, 4.47, 6.14, 5.42],
  cylinder: [2.34, 3.42, 1.66, 3.74, 4.66, 3.50, 4.95, 5.82, 5.40],
  sphere: [4.50, 4.20, 5.10, 5.80, 5.50, 6.20, 6.80, 7.10, 7.50],
  anomaly: [0.10, 0.20, 0.05, 9.80, 9.90, 0.15, 0.30, 8.90, 9.50]
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
    const val = sensorValues[i - 1];
    const row = document.createElement("div");
    row.className = "sensor-row";
    row.innerHTML = `
      <div class="sensor-row-header">
        <label for="slider_s${i}" class="sensor-label">
          <span class="sensor-label-badge">S${i}</span> Mesh Sensor ${i}
        </label>
        <input type="number" id="num_s${i}" class="sensor-number-input" 
               min="0" max="15" step="0.01" value="${val.toFixed(2)}" aria-label="Sensor ${i} Value">
      </div>
      <input type="range" id="slider_s${i}" min="0" max="15" step="0.05" value="${val}"
             aria-label="Sensor ${i} Range Slider">
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
      if (isNaN(v)) v = 0;
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
  const classes = meta.shape_classes;

  // 1. Normalize
  const x_scaled = sensorValues.map((v, i) => (v - meta.sensor_means[i]) / meta.sensor_stds[i]);

  // 2. Dense 1 + ReLU
  const h1 = [];
  for (let j = 0; j < weights.b1.length; j++) {
    let sum = weights.b1[j];
    for (let i = 0; i < 9; i++) {
      sum += x_scaled[i] * weights.w1[i][j];
    }
    h1.push(Math.max(0, sum));
  }

  // 3. Batch Normalization
  const bn1 = [];
  for (let j = 0; j < h1.length; j++) {
    const std = Math.sqrt(weights.bn_var[j] + weights.bn_eps);
    const norm = (h1[j] - weights.bn_mean[j]) / std;
    bn1.push(weights.bn_gamma[j] * norm + weights.bn_beta[j]);
  }

  // 4. Dense 2 + ReLU
  const h2 = [];
  for (let j = 0; j < weights.b2.length; j++) {
    let sum = weights.b2[j];
    for (let i = 0; i < bn1.length; i++) {
      sum += bn1[i] * weights.w2[i][j];
    }
    h2.push(Math.max(0, sum));
  }

  // 5. Dense 3 + ReLU
  const h3 = [];
  for (let j = 0; j < weights.b3.length; j++) {
    let sum = weights.b3[j];
    for (let i = 0; i < h2.length; i++) {
      sum += h2[i] * weights.w3[i][j];
    }
    h3.push(Math.max(0, sum));
  }

  // 6. Shape Head (Softmax)
  const logits = [];
  for (let j = 0; j < weights.b_shape.length; j++) {
    let sum = weights.b_shape[j];
    for (let i = 0; i < h3.length; i++) {
      sum += h3[i] * weights.w_shape[i][j];
    }
    logits.push(sum);
  }
  const maxL = Math.max(...logits);
  const expL = logits.map(l => Math.exp(l - maxL));
  const sumExp = expL.reduce((a, b) => a + b, 0);
  const probs = expL.map(e => e / sumExp);

  // 7. Dimension Head (Linear)
  let dim_mm = weights.b_dim[0];
  for (let i = 0; i < h3.length; i++) {
    dim_mm += h3[i] * weights.w_dim[i][0];
  }

  // Render results
  let bestIdx = 0;
  for (let i = 1; i < probs.length; i++) {
    if (probs[i] > probs[bestIdx]) bestIdx = i;
  }

  renderResultsUI({
    predicted_shape: classes[bestIdx],
    confidence_percentage: (probs[bestIdx] * 100).toFixed(1),
    estimated_dimension_mm: Math.max(1.0, dim_mm).toFixed(2),
    probabilities: {
      Beam: probs[classes.indexOf("Beam")] || 0,
      Cylinder: probs[classes.indexOf("Cylinder")] || 0,
      Sphere: probs[classes.indexOf("Sphere")] || 0
    },
    mode: "Client WebAssembly / JS Engine (<1ms)"
  });
}

// Update UI Components
function renderResultsUI(data) {
  document.getElementById("resShape").textContent = data.predicted_shape.toUpperCase();
  document.getElementById("resConfidence").textContent = `Confidence: ${data.confidence_percentage}%`;
  document.getElementById("resSize").textContent = `${data.estimated_dimension_mm} mm`;
  document.getElementById("resLatency").textContent = data.mode || "< 5 ms";

  // Progress Bars
  const pBeam = (data.probabilities.Beam * 100).toFixed(1);
  const pCylinder = (data.probabilities.Cylinder * 100).toFixed(1);
  const pSphere = (data.probabilities.Sphere * 100).toFixed(1);

  document.getElementById("barBeam").style.width = `${pBeam}%`;
  document.getElementById("pctBeam").textContent = `${pBeam}%`;

  document.getElementById("barCylinder").style.width = `${pCylinder}%`;
  document.getElementById("pctCylinder").textContent = `${pCylinder}%`;

  document.getElementById("barSphere").style.width = `${pSphere}%`;
  document.getElementById("pctSphere").textContent = `${pSphere}%`;
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
  const maxSensorVal = 15.0;

  // Gradient fill under curve
  const grad = ctx.createLinearGradient(0, 0, 0, h);
  grad.addColorStop(0, "rgba(56, 189, 248, 0.35)");
  grad.addColorStop(1, "rgba(56, 189, 248, 0.0)");

  ctx.beginPath();
  sensorValues.forEach((val, i) => {
    const x = paddingX + i * stepX;
    const y = h - 25 - (val / maxSensorVal) * (h - 45);
    if (i === 0) ctx.moveTo(x, y);
    else {
      // Smooth bezier curves
      const prevX = paddingX + (i - 1) * stepX;
      const prevY = h - 25 - (sensorValues[i - 1] / maxSensorVal) * (h - 45);
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
    const y = h - 25 - (val / maxSensorVal) * (h - 45);

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

  // Randomize button
  const btnRand = document.getElementById("btnRandomize");
  if (btnRand) {
    btnRand.addEventListener("click", () => {
      sensorValues = sensorValues.map(() => +(Math.random() * 10).toFixed(2));
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
