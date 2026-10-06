# 🚀 Deploying to Vercel: 9-Sensor TinyML Prediction Service

This project is now configured for **instant, high-performance deployment to Vercel**, providing both a **modern, real-time web dashboard** and a **high-speed Serverless API endpoint (`/api/predict`)**.

---

## 💡 Why Vercel is Different from Render

| Feature | Render (Current Deployment) | Vercel (New Deployment) |
| :--- | :--- | :--- |
| **Architecture** | Long-running container / Linux VM | Serverless edge functions & static CDN |
| **Streamlit Compatibility** | Native (persistent WebSockets) | Not natively supported (serverless timeouts) |
| **Package Size Limit** | Gigabytes | Strict **250 MB** uncompressed limit |
| **Cold Starts** | None (always on, unless free tier sleeps) | Sub-15ms cold start with lightweight serverless engine |
| **Cost & Scaling** | Monthly container cost / sleep on free | Generous **free serverless tier**, instant global CDN |

> [!IMPORTANT]
> Standard `tensorflow` (~500 MB+) will fail Vercel builds due to Lambda package size limits. We extracted the trained weights and normalization statistics into a lightweight bundle (`152 KB`) in `api/model_weights.json` and created a serverless function in [api/predict.py](file:///c:/Users/HP/Downloads/Anomalies_Prediction/api/predict.py) that executes inference in **under 5 milliseconds** using only `numpy`.

---

## 🛠️ Project Structure for Vercel

```text
Anomalies_Prediction/
├── vercel.json                 # Vercel routing & CORS header configuration
├── .vercelignore               # Prevents large data files/notebooks from bloating function bundle
├── api/
│   ├── predict.py              # Serverless function handling GET/POST/OPTIONS
│   ├── model_weights.json      # Trained network weights + sensor mean/std normalization
│   └── requirements.txt        # Lightweight requirements (numpy only)
├── public/
│   ├── index.html              # Modern glassmorphism web dashboard
│   ├── styles.css              # Custom responsive dark-theme design system
│   ├── app.js                  # Real-time slider events, canvas waveform & API tester
│   └── model_weights.json      # Client-side weights for instantaneous 60fps interaction
├── dev_server.py               # Local simulator for testing the Vercel app offline
├── requirements-render.txt     # Complete requirements preserved for Render (Streamlit + TF)
└── export_weights_for_vercel.py# Script to re-export weights whenever you retrain the model
```

---

## 🚀 Deployment Methods

### Method 1: Deploy via GitHub (Recommended)

1. **Commit and push** the new Vercel files to your GitHub repository:
   ```bash
   git add vercel.json .vercelignore api/ public/ export_weights_for_vercel.py requirements-render.txt DEPLOY_TO_VERCEL.md
   git commit -m "Configure Vercel serverless deployment and web dashboard"
   git push origin main
   ```

2. Open the [Vercel Dashboard](https://vercel.com/dashboard).
3. Click **"Add New..."** ➔ **"Project"**.
4. Select your GitHub repository: `Madhavan-33/Tinyml_Project`.
5. Under **Build and Output Settings**:
   - Leave the defaults as detected (the included `vercel.json` configures everything automatically).
6. Click **Deploy**.
7. In ~30 seconds, your site will be live at `https://<your-project>.vercel.app`!

---

### Method 2: Deploy via Vercel CLI

1. Install the Vercel CLI (if not already installed):
   ```bash
   npm install -g vercel
   ```

2. Run the deployment command in your terminal:
   ```bash
   vercel
   ```
   *(Follow the brief prompts: link to existing project? [N], project name? [enter], etc.)*

3. Deploy to production:
   ```bash
   vercel --prod
   ```

---

## 🔌 Using the Serverless REST API

Your Vercel deployment provides a public REST API at `/api/predict`.

### 1. Health & Metadata Check
```bash
curl -X GET "https://your-project.vercel.app/api/predict?meta=true"
```

### 2. Predict Object Shape & Size (POST)
Send a JSON payload containing the 9 sensor values (`S1` to `S9`):

```bash
curl -X POST "https://your-project.vercel.app/api/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "sensors": [1.19, 4.47, 1.98, 2.90, 5.30, 3.68, 4.47, 6.14, 5.42]
  }'
```

#### Sample Response:
```json
{
  "predicted_shape": "Beam",
  "confidence_percentage": 94.8,
  "estimated_dimension_mm": 24.11,
  "shape_probabilities": {
    "Beam": 0.948,
    "Cylinder": 0.031,
    "Sphere": 0.021
  },
  "sensor_inputs": [1.19, 4.47, 1.98, 2.90, 5.30, 3.68, 4.47, 6.14, 5.42],
  "latency_ms": 1.45
}
```

---

## 💻 Testing Locally Before Deploying

You can run the included local simulator:
```bash
python dev_server.py 3000
```
Then visit:
- **Interactive UI**: `http://localhost:3000`
- **Serverless API**: `http://localhost:3000/api/predict`

---

## 🔄 Retraining the Model in the Future

If you retrain your model with new telemetry data, run:
```bash
python export_weights_for_vercel.py
```
This updates `anomaly_model.tflite`, `model_data.h`, and re-generates the lightweight `api/model_weights.json` for Vercel automatically.
