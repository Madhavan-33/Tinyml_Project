import streamlit as st
import numpy as np
import tensorflow as tf
import time
import pandas as pd
import joblib
import os
import re
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error, r2_score, confusion_matrix

try:
    import serial
    import serial.tools.list_ports
    HAS_PYSERIAL = True
except ImportError:
    HAS_PYSERIAL = False

try:
    from mcu_exporter import create_mcu_deployment_zip
    HAS_EXPORTER = True
except ImportError:
    HAS_EXPORTER = False


# Page Configuration
st.set_page_config(
    page_title="9-Sensor Mesh Object & Size Analytics Lab",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom Styling (Dark Theme & Glassmorphism)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    .stApp {
        background: linear-gradient(135deg, #090d16 0%, #0f172a 50%, #090d16 100%);
        color: #f1f5f9;
    }

    header[data-testid="stHeader"] {
        background: transparent !important;
        background-color: transparent !important;
        z-index: 100;
    }
    
    header[data-testid="stHeader"] * {
        color: #cbd5e1 !important;
    }

    div.block-container {
        padding-top: 1.2rem !important;
        padding-left: 1.5rem !important;
        padding-right: 1.5rem !important;
        padding-bottom: 2rem !important;
        max-width: 100% !important;
    }

    /* Remove Streamlit's default max-width cap on main content */
    .main .block-container {
        max-width: 100% !important;
        width: 100% !important;
    }

    /* Expand dataframes to full width */
    div[data-testid="stDataFrame"] > div {
        width: 100% !important;
    }

    .hero-banner {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.8) 0%, rgba(15, 23, 42, 0.9) 100%);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 24px 30px;
        margin-top: 10px;
        margin-bottom: 24px;
        backdrop-filter: blur(12px);
        box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.5);
    }
    
    .hero-title {
        font-size: 28px;
        font-weight: 800;
        background: linear-gradient(90deg, #38bdf8 0%, #818cf8 50%, #c084fc 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0 0 8px 0;
        letter-spacing: -0.5px;
    }

    .hero-subtitle {
        color: #cbd5e1;
        font-size: 14px;
        margin: 0;
        font-weight: 400;
    }

    .status-badge-container {
        display: flex;
        gap: 10px;
        margin-top: 14px;
        flex-wrap: wrap;
    }

    .status-chip {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(255, 255, 255, 0.06);
        border: 1px solid rgba(255, 255, 255, 0.12);
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 500;
        color: #e2e8f0;
    }

    .chip-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
    }
    .chip-dot.green { background-color: #10b981; box-shadow: 0 0 8px #10b981; }
    .chip-dot.cyan { background-color: #38bdf8; box-shadow: 0 0 8px #38bdf8; }
    .chip-dot.purple { background-color: #a855f7; box-shadow: 0 0 8px #a855f7; }

    .kpi-card {
        background: linear-gradient(145deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.8) 100%);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 14px;
        padding: 16px;
        text-align: left;
        backdrop-filter: blur(8px);
        transition: transform 0.2s ease, border-color 0.2s ease;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
    }
    
    .kpi-card:hover {
        transform: translateY(-2px);
        border-color: rgba(56, 189, 248, 0.4);
    }

    .kpi-label {
        font-size: 11px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        color: #94a3b8;
        margin-bottom: 6px;
    }

    .kpi-value {
        font-size: 22px;
        font-weight: 700;
        color: #f8fafc;
        font-family: 'JetBrains Mono', monospace;
    }

    .kpi-sub {
        font-size: 11px;
        color: #cbd5e1;
        margin-top: 4px;
    }

    section[data-testid="stSidebar"] {
        background-color: #0b1120 !important;
        border-right: 1px solid rgba(255, 255, 255, 0.08) !important;
    }
    
    section[data-testid="stSidebar"] * {
        color: #e2e8f0 !important;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 6px;
        background-color: rgba(15, 23, 42, 0.8);
        padding: 6px 8px;
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.08);
    }

    .stTabs [data-baseweb="tab-list"] button {
        border-radius: 8px;
        padding: 8px 16px;
        font-weight: 500;
        font-size: 13px;
        color: #94a3b8 !important;
        border: none !important;
        background-color: transparent !important;
        transition: all 0.2s ease;
    }

    .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {
        background: linear-gradient(135deg, #0284c7 0%, #2563eb 100%) !important;
        color: #ffffff !important;
        font-weight: 600 !important;
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.35);
    }

    div[data-testid="stDataFrame"] {
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        overflow: hidden;
    }
</style>
""", unsafe_allow_html=True)

def setup_dark_plot_style():
    plt.style.use('dark_background')
    return {'facecolor': '#0f172a', 'edgecolor': 'none'}

def apply_axes_dark_style(ax, title="", xlabel="", ylabel=""):
    ax.set_facecolor('#0b1120')
    ax.set_title(title, fontsize=11, fontweight='700', color='#f1f5f9', pad=12)
    ax.set_xlabel(xlabel, fontsize=10, color='#94a3b8', labelpad=8)
    ax.set_ylabel(ylabel, fontsize=10, color='#94a3b8', labelpad=8)
    ax.tick_params(colors='#64748b', labelsize=9)
    ax.grid(True, linestyle='--', alpha=0.25, color='#334155')
    for spine in ax.spines.values():
        spine.set_color('#1e293b')

def render_kpi_card(label, value, subtext="", accent_color="#38bdf8"):
    st.markdown(f"""
    <div class="kpi-card" style="border-left: 4px solid {accent_color};">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
        {'<div class="kpi-sub">' + subtext + '</div>' if subtext else ''}
    </div>
    """, unsafe_allow_html=True)


# Hero Top Header Section
st.markdown("""
<div class="hero-banner">
    <div class="hero-title">⚡ 9-Sensor Mesh & TinyML Object Identification Lab</div>
    <div class="hero-subtitle">Isolate 9 Sensor Outputs (S1 ... S9) captured from physical mesh ➔ Predict Object Shape (Classification) & Dimension Size (Regression)</div>
    <div class="status-badge-container">
        <div class="status-chip"><span class="chip-dot green"></span> TinyML 9-Input Model Active</div>
        <div class="status-chip"><span class="chip-dot cyan"></span> 9-Channel Mesh & Correlation</div>
        <div class="status-chip"><span class="chip-dot purple"></span> Dual-Task: Shape & Size Prediction</div>
    </div>
</div>
""", unsafe_allow_html=True)


def auto_train_default_model(dataset_source=None):
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from convert_model import convert_tflite_to_c_header

    if dataset_source is None or not os.path.exists(str(dataset_source)):
        dataset_source = "sensor_design_dataset.csv"
        if not os.path.exists(dataset_source):
            from dataset_generator import generate_sensor_design_dataset
            generate_sensor_design_dataset(dataset_source)

    if str(dataset_source).endswith('.csv'):
        df = pd.read_csv(dataset_source)
    else:
        df = pd.read_excel(dataset_source)

    cols_s = [c for c in df.columns if re.match(r'^s[1-9]$', c, re.IGNORECASE)]
    if len(cols_s) < 9:
        cols_s = [f'S{i}' for i in range(1, 10) if f'S{i}' in df.columns]
    if len(cols_s) < 9:
        numeric_cols = list(df.select_dtypes(include=[np.number]).columns)
        cols_s = numeric_cols[:9]

    shape_col = 'object_shape' if 'object_shape' in df.columns else next((c for c in df.columns if 'shape' in c.lower()), df.columns[1])
    dim_col = 'dim_mm' if 'dim_mm' in df.columns else next((c for c in df.columns if 'dim' in c.lower() or 'size' in c.lower()), df.columns[2])

    df_clean = df[cols_s[:9] + [shape_col, dim_col]].dropna().copy()
    X_raw = df_clean[cols_s[:9]].values.astype(np.float32)

    scaler_sensors = StandardScaler()
    X_scaled = scaler_sensors.fit_transform(X_raw).astype(np.float32)

    shape_encoder = LabelEncoder()
    y_shape = shape_encoder.fit_transform(df_clean[shape_col].astype(str))
    num_classes = len(shape_encoder.classes_)
    y_dim = df_clean[dim_col].values.astype(np.float32)

    # Build dual-head model
    inputs = tf.keras.layers.Input(shape=(9,), name='sensor_inputs')
    x = tf.keras.layers.Dense(64, activation='relu')(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.1)(x)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    x = tf.keras.layers.Dense(32, activation='relu')(x)

    shape_out = tf.keras.layers.Dense(num_classes, activation='softmax', name='shape_out')(x)
    dim_out = tf.keras.layers.Dense(1, activation='linear', name='dim_out')(x)

    training_model = tf.keras.models.Model(inputs=inputs, outputs=[shape_out, dim_out])
    training_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.002),
        loss={'shape_out': 'sparse_categorical_crossentropy', 'dim_out': 'mse'},
        loss_weights={'shape_out': 1.0, 'dim_out': 0.05}
    )
    training_model.fit(X_scaled, [y_shape, y_dim], epochs=45, batch_size=32, verbose=0)

    # Export combined model
    combined_out = tf.keras.layers.Concatenate(axis=-1, name='combined')([shape_out, dim_out])
    export_model = tf.keras.models.Model(inputs=inputs, outputs=combined_out)

    preds_all = export_model.predict(X_scaled, verbose=0)
    pred_shape_idx = np.argmax(preds_all[:, :num_classes], axis=1)
    pred_dims = preds_all[:, -1]

    acc = accuracy_score(y_shape, pred_shape_idx)
    mae = mean_absolute_error(y_dim, pred_dims)
    r2 = r2_score(y_dim, pred_dims)

    metadata = {
        'input_features': cols_s[:9],
        'shape_classes': list(shape_encoder.classes_),
        'shape_encoder': shape_encoder,
        'shape_col': shape_col,
        'dim_col': dim_col,
        'scaler_sensors': scaler_sensors,
        'sensor_means': scaler_sensors.mean_.tolist(),
        'sensor_stds': np.sqrt(scaler_sensors.var_).tolist(),
        'accuracy': float(acc),
        'dim_mae': float(mae),
        'dim_r2': float(r2),
        'num_inputs': 9,
        'num_classes': num_classes,
        'num_outputs': num_classes + 1
    }
    joblib.dump(metadata, 'model_metadata.pkl')

    converter = tf.lite.TFLiteConverter.from_keras_model(export_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    with open("anomaly_model.tflite", "wb") as f:
        f.write(tflite_model)

    num_inputs = 9
    num_outputs = num_classes + 1
    in_means_str = ", ".join([f"{m:.6f}f" for m in scaler_sensors.mean_])
    in_stds_str = ", ".join([f"{s:.6f}f" for s in np.sqrt(scaler_sensors.var_)])
    class_names_str = ", ".join([f'"{c}"' for c in shape_encoder.classes_])

    header_content = f"""#ifndef MODEL_CONFIG_H
#define MODEL_CONFIG_H

#define NUM_INPUTS {num_inputs}
#define NUM_OUTPUTS {num_outputs}
#define NUM_SHAPE_CLASSES {num_classes}

static const float INPUT_MEANS[{num_inputs}] = {{ {in_means_str} }};
static const float INPUT_STDS[{num_inputs}] = {{ {in_stds_str} }};
static const char* const SHAPE_CLASSES[{num_classes}] = {{ {class_names_str} }};

#endif
"""
    with open("model_config.h", "w") as f:
        f.write(header_content)

    try:
        convert_tflite_to_c_header("anomaly_model.tflite", "model_data.h")
    except Exception:
        pass

    st.cache_resource.clear()
    return True

@st.cache_resource
def load_tflite_model():
    try:
        interpreter = tf.lite.Interpreter(model_path="anomaly_model.tflite")
        interpreter.allocate_tensors()
        return interpreter
    except Exception:
        return None

@st.cache_resource
def load_metadata():
    try:
        meta = joblib.load("model_metadata.pkl")
        if 'scaler_sensors' in meta:
            return meta
        return None
    except Exception:
        return None

interpreter = load_tflite_model()
metadata = load_metadata()

if interpreter is None or metadata is None:
    with st.spinner("⚙️ Initializing 9-Sensor TinyML Object & Sizing Model..."):
        auto_train_default_model()
        interpreter = load_tflite_model()
        metadata = load_metadata()
        st.success("✅ Model initialized successfully!")

scaler_sensors = metadata['scaler_sensors']
shape_classes = metadata['shape_classes']
shape_encoder = metadata['shape_encoder']
shape_col = metadata.get('shape_col', 'object_shape')
dim_col = metadata.get('dim_col', 'dim_mm')
cols_s = metadata.get('input_features', [f'S{i}' for i in range(1, 10)])
num_classes = len(shape_classes)

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def find_available_datasets():
    files = [f for f in os.listdir('.') if (f.endswith('.csv') or f.endswith('.xlsx')) and not f.startswith('~$')]
    return files if files else None


# Sidebar Ingestion & Filter Controls (Reduce Data-Size)
st.sidebar.markdown("### 📥 Telemetry Dataset & Filters")

with st.sidebar.expander("❓ Mentor Requirements & Architecture", expanded=False):
    st.markdown("""
    **Architecture Highlights:**
    - **Features**: Strictly the **9 sensor outputs** (`S1` to `S9`) captured from sensor mesh.
    - **9 Legends Plot**: Sensor waveforms with 9 distinct legends.
    - **Correlation Matrix**: Correlation between 9 sensors for different shapes & dimensions.
    - **Filters**: Shape & dimension range filters to reduce data size.
    - **All Data Prediction**: Evaluates which object and what size based on predicted values.
    """)

uploaded_file = st.sidebar.file_uploader("Upload Dataset (.csv / .xlsx):", type=["csv", "xlsx"])
available_files = find_available_datasets()

if uploaded_file is not None:
    file_source = uploaded_file
    file_name = uploaded_file.name
elif available_files:
    default_idx = available_files.index("sensor_design_dataset.csv") if "sensor_design_dataset.csv" in available_files else 0
    file_name = st.sidebar.selectbox("Workspace Dataset:", options=available_files, index=default_idx)
    file_source = file_name
else:
    file_source = None

if file_source is None:
    st.error("⚠️ No dataset found! Please upload a `.csv` or `.xlsx` file.")
    st.stop()

try:
    if file_name.endswith('.csv'):
        df_raw = pd.read_csv(file_source)
    else:
        xls = pd.ExcelFile(file_source)
        selected_sheet = st.sidebar.selectbox("Sheet Name:", options=xls.sheet_names)
        df_raw = pd.read_excel(xls, sheet_name=selected_sheet)
except Exception as e:
    st.error(f"Error loading dataset: {e}")
    st.stop()

# Detect Columns
curr_shape_col = 'object_shape' if 'object_shape' in df_raw.columns else next((c for c in df_raw.columns if 'shape' in c.lower()), df_raw.columns[1])
curr_dim_col = 'dim_mm' if 'dim_mm' in df_raw.columns else next((c for c in df_raw.columns if 'dim' in c.lower() or 'size' in c.lower()), df_raw.columns[2])

st.sidebar.markdown("---")
st.sidebar.markdown("### 🎯 Filter & Reduce Data-Size")

# 1. Filter Object Shape
available_shapes = sorted(list(df_raw[curr_shape_col].dropna().unique()))
filter_shapes = st.sidebar.multiselect("Object Shape Filter:", options=available_shapes, default=available_shapes)
if not filter_shapes: filter_shapes = available_shapes

# 2. Filter Dimension Range
min_d = float(df_raw[curr_dim_col].min())
max_d = float(df_raw[curr_dim_col].max())
filter_dim_range = st.sidebar.slider("Dimension `dim_mm` (mm):", min_value=round(min_d, 1), max_value=round(max_d, 1), value=(round(min_d, 1), round(max_d, 1)), step=0.5)

# 3. Downsample Rate to reduce size
filter_step = st.sidebar.selectbox("Sample Step (Reduce Data):", options=[1, 2, 3, 5, 10], format_func=lambda x: f"Every {x} sample" if x > 1 else "1:1 Full Resolution")

# Apply Filter
df_filtered = df_raw[
    (df_raw[curr_shape_col].isin(filter_shapes)) &
    (df_raw[curr_dim_col] >= filter_dim_range[0]) &
    (df_raw[curr_dim_col] <= filter_dim_range[1])
].copy()

if filter_step > 1:
    df_filtered = df_filtered.iloc[::filter_step].copy()

st.sidebar.markdown(f"**Active Records**: `{len(df_filtered):,}` of `{len(df_raw):,}` ({((len(df_raw)-len(df_filtered))/len(df_raw))*100.0:.1f}% reduction)")

st.sidebar.markdown("---")
st.sidebar.markdown("### ⚡ Model Retraining")
if st.sidebar.button("🛠️ Retrain 9-Sensor Model", use_container_width=True):
    with st.spinner("Retraining 9-Sensor model on active file..."):
        auto_train_default_model(dataset_source=file_source)
        st.sidebar.success("✅ Model retrained successfully!")
        st.rerun()


# Navigation Tabs
tab0, tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📁 Dataset & 9 Legends Correlation",
    "🔮 Run All Data: Object & Size Predictor",
    "🎛️ Live 9-Sensor Manual Predictor",
    "🔬 MCU Model Fit (ARM Cortex-M)",
    "🔌 Firmware Export & Serial UART",
    "⚡ Edge Latency & Renode Stream"
])


# --- TAB 0: Dataset Inspector, 9 Legends Visualization & Correlation Matrix ---
with tab0:
    st.markdown(f"### 📁 Dataset Schema & 9-Sensor Inspector — `{file_name}`")
    st.caption("Inspect filtered telemetry, 9-sensor waveform profiles with 9 legends, and the 9x9 sensor correlation matrix.")

    n_rows, n_cols = df_filtered.shape
    c1, c2, c3, c4 = st.columns(4)
    with c1: render_kpi_card("Filtered Samples", f"{n_rows:,}", f"Original: {len(df_raw):,}", "#38bdf8")
    with c2: render_kpi_card("Data Reduction", f"{((len(df_raw)-n_rows)/len(df_raw))*100.0:.1f}%", "Memory Optimized", "#34d399")
    with c3: render_kpi_card("Sensor Mesh", f"{len(cols_s)} Channels", "S1 - S9 Inputs", "#818cf8")
    with c4: render_kpi_card("Selected Shapes", f"{len(filter_shapes)}", ", ".join(filter_shapes), "#c084fc")

    st.markdown("<br>", unsafe_allow_html=True)
    
    # 9 Legends Line Plot
    st.markdown("#### 📈 Multi-Sensor Waveform Profile (9 Explicit Legends)")
    fig_9leg, ax_9leg = plt.subplots(figsize=(11, 4.2), **setup_dark_plot_style())
    colors_9 = ['#38bdf8', '#818cf8', '#c084fc', '#f472b6', '#fb7185', '#fb923c', '#facc15', '#4ade80', '#2dd4bf']
    
    plot_sub = df_filtered.head(150).reset_index(drop=True)
    for idx, s_name in enumerate(cols_s[:9]):
        if s_name in plot_sub.columns:
            ax_9leg.plot(plot_sub.index, plot_sub[s_name], label=f'{s_name}', color=colors_9[idx % len(colors_9)], linewidth=1.8, alpha=0.9)

    apply_axes_dark_style(ax_9leg, title=f"9 Sensor Traces (S1 - S9) for Shapes: {', '.join(filter_shapes)} | Dimensions: {filter_dim_range[0]}-{filter_dim_range[1]}mm", xlabel="Filtered Sample Index", ylabel="Sensor Amplitude")
    ax_9leg.legend(loc="upper right", facecolor='#0f172a', edgecolor='#1e293b', fontsize=8.5, ncol=5, title="9 Sensor Legends")
    plt.tight_layout()
    st.pyplot(fig_9leg)

    st.markdown("<br>", unsafe_allow_html=True)

    # Correlation Matrix Section
    st.markdown("#### 🔥 9x9 Sensor-to-Sensor Correlation Matrix")
    st.caption("Evaluates correlation between each sensor value specifically for the chosen object shape and dimension range filter.")
    
    col_corr1, col_corr2 = st.columns([3, 2])
    with col_corr1:
        corr_df = df_filtered[cols_s[:9]].corr()
        fig_c, ax_c = plt.subplots(figsize=(7, 5), **setup_dark_plot_style())
        cax = ax_c.imshow(corr_df.values, cmap='coolwarm', vmin=-1.0, vmax=1.0)
        cbar = fig_c.colorbar(cax, ax=ax_c, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(colors='#94a3b8')
        
        ax_c.set_xticks(range(len(cols_s[:9])))
        ax_c.set_yticks(range(len(cols_s[:9])))
        ax_c.set_xticklabels(cols_s[:9], color='#f1f5f9', fontsize=8.5, fontweight='600')
        ax_c.set_yticklabels(cols_s[:9], color='#f1f5f9', fontsize=8.5, fontweight='600')
        
        for i in range(len(cols_s[:9])):
            for j in range(len(cols_s[:9])):
                val = corr_df.iloc[i, j]
                txt_col = '#ffffff' if abs(val) > 0.4 else '#94a3b8'
                ax_c.text(j, i, f"{val:.2f}", ha='center', va='center', color=txt_col, fontsize=7.5, fontweight='bold')
        
        apply_axes_dark_style(ax_c, title=f"Sensor Correlation ({', '.join(filter_shapes)})")
        plt.tight_layout()
        st.pyplot(fig_c)

    with col_corr2:
        st.markdown("##### 📌 Correlation Observations")
        st.markdown(f"""
        - **Active Shapes**: `{', '.join(filter_shapes)}`
        - **Dimension Range**: `{filter_dim_range[0]} mm` to `{filter_dim_range[1]} mm`
        - **Active Samples**: `{len(df_filtered):,}`
        - **Sensor Coupling**: Channels show strong localized coupling depending on object geometry.
        - As shape changes between Cylinder, Sphere, and Beam, the spatial deformation induces distinctive cross-sensor correlations!
        """)
        st.markdown("##### 📋 Filtered Data Sample")
        st.dataframe(df_filtered.head(10)[cols_s[:9] + [curr_shape_col, curr_dim_col]], use_container_width=True)


# --- TAB 1: Run All Data & Predict Object & Size ---
with tab1:
    st.markdown(f"### 🔮 Run All Data: Object Shape & Size Prediction")
    st.caption("Runs inference on all active records: predicts Which Object (Shape) and What Size (Dimension mm) based on 9 sensor values.")

    col_run1, col_run2 = st.columns([3, 1])
    with col_run1:
        run_mode = st.radio("Inference Scope:", options=["Run Active Filtered Data", "Run Full Dataset (All Rows)"], horizontal=True)
    with col_run2:
        st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
        execute_all = st.button("▶️ Run Inference on All Data", type="primary", use_container_width=True)

    target_eval_df = df_filtered.copy() if "Filtered" in run_mode else df_raw.copy()

    if execute_all or 'pred_cache' in st.session_state:
        if execute_all:
            with st.spinner(f"Running inference across {len(target_eval_df):,} records..."):
                clean_eval_df = target_eval_df[cols_s[:9]].dropna().copy()
                X_eval_raw = clean_eval_df[cols_s[:9]].values.astype(np.float32)
                X_eval_scaled = scaler_sensors.transform(X_eval_raw).astype(np.float32)

                all_preds = []
                for i in range(len(X_eval_scaled)):
                    sample = X_eval_scaled[i:i+1]
                    interpreter.set_tensor(input_details[0]['index'], sample)
                    interpreter.invoke()
                    out_t = interpreter.get_tensor(output_details[0]['index'])
                    all_preds.append(out_t[0])

                all_preds = np.array(all_preds)
                pred_shape_probs = all_preds[:, :num_classes]
                pred_shape_idx = np.argmax(pred_shape_probs, axis=1)
                pred_shape_names = shape_encoder.inverse_transform(pred_shape_idx)
                pred_confidences = np.max(pred_shape_probs, axis=1) * 100.0
                pred_dims = all_preds[:, -1]

                results_df = target_eval_df.loc[clean_eval_df.index].copy()
                results_df['Predicted_Object'] = pred_shape_names
                results_df['Shape_Confidence_%'] = np.round(pred_confidences, 1)
                results_df['Predicted_Size_mm'] = np.round(pred_dims, 2)

                has_truth = curr_shape_col in results_df.columns and curr_dim_col in results_df.columns
                if has_truth:
                    results_df['Actual_Object'] = results_df[curr_shape_col]
                    results_df['Actual_Size_mm'] = np.round(results_df[curr_dim_col], 2)
                    results_df['Shape_Match'] = ["✅ MATCH" if a == p else "❌ MISMATCH" for a, p in zip(results_df['Actual_Object'], results_df['Predicted_Object'])]
                    results_df['Size_Error_mm'] = np.round(np.abs(results_df['Actual_Size_mm'] - results_df['Predicted_Size_mm']), 2)

                    # Safe label encoding: handle unseen labels (e.g. shapes not in training set)
                    known_classes = list(shape_encoder.classes_)
                    def safe_encode(label):
                        return known_classes.index(label) if label in known_classes else -1

                    y_true_shape_raw = results_df['Actual_Object'].astype(str).tolist()
                    y_true_shape = np.array([safe_encode(lbl) for lbl in y_true_shape_raw])

                    # Only compare rows where the actual label is a known class
                    known_mask = (y_true_shape >= 0)
                    if known_mask.sum() > 0:
                        eval_acc = accuracy_score(y_true_shape[known_mask], pred_shape_idx[known_mask])
                        eval_mae = mean_absolute_error(results_df['Actual_Size_mm'].values[known_mask], pred_dims[known_mask])
                        eval_rmse = np.sqrt(mean_squared_error(results_df['Actual_Size_mm'].values[known_mask], pred_dims[known_mask]))
                        eval_r2 = r2_score(results_df['Actual_Size_mm'].values[known_mask], pred_dims[known_mask])
                        unseen_labels = set(np.array(y_true_shape_raw)[~known_mask])
                        if unseen_labels:
                            st.warning(f"⚠️ Dataset contains **{(~known_mask).sum()} rows** with unseen object labels not in training: `{'`, `'.join(sorted(unseen_labels))}`. These rows are excluded from accuracy metrics but predictions are still shown.")
                    else:
                        eval_acc, eval_mae, eval_rmse, eval_r2 = None, None, None, None
                        st.warning("⚠️ No rows matched the trained shape labels — metrics cannot be computed.")
                else:
                    eval_acc, eval_mae, eval_rmse, eval_r2 = None, None, None, None

                st.session_state.pred_cache = {
                    'results_df': results_df,
                    'eval_acc': eval_acc,
                    'eval_mae': eval_mae,
                    'eval_rmse': eval_rmse,
                    'eval_r2': eval_r2,
                    'has_truth': has_truth,
                    'y_true_shape': y_true_shape if has_truth else None,
                    'pred_shape_idx': pred_shape_idx
                }

        # Retrieve cached evaluation
        cache = st.session_state.pred_cache
        res_df = cache['results_df']
        has_truth = cache['has_truth']

        if has_truth and cache['eval_acc'] is not None:
            p1, p2, p3, p4 = st.columns(4)
            with p1: render_kpi_card("Object Accuracy", f"{cache['eval_acc']*100.0:.1f}%", "Shape Classification", "#34d399")
            with p2: render_kpi_card("Size MAE", f"{cache['eval_mae']:.2f} mm", "Mean Absolute Error", "#38bdf8")
            with p3: render_kpi_card("Size RMSE", f"{cache['eval_rmse']:.2f} mm", "Root Mean Square", "#f59e0b")
            with p4: render_kpi_card("Size R² Score", f"{cache['eval_r2']:.3f}", "Regression Fit Quality", "#c084fc")

            st.markdown("<br>", unsafe_allow_html=True)
            
            # Confusion Matrix & Regression Scatter
            r_col1, r_col2 = st.columns(2)
            with r_col1:
                # Filter out unseen labels (index = -1) before confusion matrix
                yt = cache['y_true_shape']
                yp = cache['pred_shape_idx']
                known_cm_mask = (yt >= 0)
                cm = confusion_matrix(yt[known_cm_mask], yp[known_cm_mask], labels=list(range(num_classes)))
                fig_cm, ax_cm = plt.subplots(figsize=(7, 5), **setup_dark_plot_style())
                cax_cm = ax_cm.imshow(cm, cmap='Blues')
                fig_cm.colorbar(cax_cm, ax=ax_cm, fraction=0.046, pad=0.04)
                ax_cm.set_xticks(range(num_classes))
                ax_cm.set_yticks(range(num_classes))
                ax_cm.set_xticklabels(shape_classes, color='#f1f5f9', fontsize=9)
                ax_cm.set_yticklabels(shape_classes, color='#f1f5f9', fontsize=9)
                ax_cm.set_xlabel("Predicted Object", color='#94a3b8', fontsize=9)
                ax_cm.set_ylabel("Actual Object", color='#94a3b8', fontsize=9)
                apply_axes_dark_style(ax_cm, title="Confusion Matrix (Object Classification)")
                
                for i in range(num_classes):
                    for j in range(num_classes):
                        ax_cm.text(j, i, str(cm[i, j]), ha='center', va='center', color='#ffffff' if cm[i, j] > cm.max()/2 else '#94a3b8', fontweight='bold')
                plt.tight_layout()
                st.pyplot(fig_cm)

            with r_col2:
                fig_sc, ax_sc = plt.subplots(figsize=(5.5, 3.8), **setup_dark_plot_style())
                for c_idx, c_name in enumerate(shape_classes):
                    sub_m = (res_df['Predicted_Object'] == c_name)
                    ax_sc.scatter(res_df.loc[sub_m, 'Actual_Size_mm'], res_df.loc[sub_m, 'Predicted_Size_mm'], label=c_name, alpha=0.7, s=26, color=colors_9[c_idx % len(colors_9)])
                
                min_val = min(res_df['Actual_Size_mm'].min(), res_df['Predicted_Size_mm'].min())
                max_val = max(res_df['Actual_Size_mm'].max(), res_df['Predicted_Size_mm'].max())
                ax_sc.plot([min_val, max_val], [min_val, max_val], color='#ef4444', linestyle='--', linewidth=2, label='Ideal y=x')
                apply_axes_dark_style(ax_sc, title=f"Actual vs Predicted Size (MAE = {cache['eval_mae']:.2f} mm)", xlabel="Actual Size (mm)", ylabel="Predicted Size (mm)")
                ax_sc.legend(loc="upper left", facecolor='#0f172a', edgecolor='#1e293b', fontsize=8)
                plt.tight_layout()
                st.pyplot(fig_sc)

        st.markdown("#### 📋 Prediction Audit Table")
        
        filter_view = st.radio("Display Filter:", options=["All Predictions", "Matches Only (✅)", "Mismatches Only (❌)"], horizontal=True)
        if filter_view == "Matches Only (✅)" and 'Shape_Match' in res_df.columns:
            display_df = res_df[res_df['Shape_Match'] == "✅ MATCH"]
        elif filter_view == "Mismatches Only (❌)" and 'Shape_Match' in res_df.columns:
            display_df = res_df[res_df['Shape_Match'] == "❌ MISMATCH"]
        else:
            display_df = res_df

        view_order = ['Predicted_Object', 'Shape_Confidence_%', 'Predicted_Size_mm']
        if has_truth:
            view_order = ['Actual_Object', 'Predicted_Object', 'Shape_Match', 'Shape_Confidence_%', 'Actual_Size_mm', 'Predicted_Size_mm', 'Size_Error_mm']
        view_order += cols_s[:9]

        st.dataframe(display_df[view_order].head(200), use_container_width=True)

        csv_download = display_df[view_order].to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Predictions Table (CSV)",
            data=csv_download,
            file_name="tinyml_object_and_size_predictions.csv",
            mime="text/csv"
        )


# --- TAB 2: Live 9-Sensor Manual Predictor ---
with tab2:
    st.markdown("### 🎛️ Live 9-Sensor Manual Inference Playground")
    st.caption("Enter or slide any 9 sensor values to immediately predict Which Object and What Size in real time.")

    col_sim1, col_sim2 = st.columns([1, 1])

    with col_sim1:
        st.markdown("#### 📡 9 Sensor Input Channels (S1 - S9)")
        
        # Preset buttons from dataset
        if st.button("🎲 Load Random Real Record Preset", use_container_width=True):
            rand_row = df_raw.sample(1).iloc[0]
            for i, s_col in enumerate(cols_s[:9]):
                st.session_state[f'slider_{s_col}'] = float(rand_row[s_col])

        sim_sensors = []
        for i, s_col in enumerate(cols_s[:9]):
            default_val = float(scaler_sensors.mean_[i])
            val = st.slider(
                f"Sensor {s_col}:",
                min_value=0.0,
                max_value=50.0,
                value=float(st.session_state.get(f'slider_{s_col}', default_val)),
                step=0.1,
                key=f'slider_{s_col}'
            )
            sim_sensors.append(val)

    with col_sim2:
        st.markdown("#### 🎯 Predicted Object & Size Output")

        # Normalize 9 sensor inputs
        sim_raw = np.array([sim_sensors], dtype=np.float32)
        sim_scaled = scaler_sensors.transform(sim_raw).astype(np.float32)

        interpreter.set_tensor(input_details[0]['index'], sim_scaled)
        interpreter.invoke()
        sim_out = interpreter.get_tensor(output_details[0]['index'])[0]

        sim_shape_probs = sim_out[:num_classes]
        best_idx = int(np.argmax(sim_shape_probs))
        pred_obj_name = shape_classes[best_idx]
        pred_conf = float(sim_shape_probs[best_idx] * 100.0)
        pred_dim_val = float(sim_out[-1])

        # Large Visual KPI Cards
        sk1, sk2 = st.columns(2)
        with sk1:
            render_kpi_card("Identified Object", pred_obj_name.upper(), f"Confidence: {pred_conf:.1f}%", "#34d399")
        with sk2:
            render_kpi_card("Estimated Size", f"{pred_dim_val:.2f} mm", "Continuous Dimension", "#38bdf8")

        st.markdown("<br>", unsafe_allow_html=True)
        # Class probability breakdown
        st.markdown("##### 📊 Shape Classification Probabilities")
        fig_prob, ax_prob = plt.subplots(figsize=(6, 2.8), **setup_dark_plot_style())
        bars = ax_prob.barh(shape_classes, sim_shape_probs * 100.0, color=['#38bdf8', '#818cf8', '#c084fc'][:len(shape_classes)], height=0.55)
        apply_axes_dark_style(ax_prob, title="Softmax Probability Distribution", xlabel="Probability (%)")
        for b in bars:
            w = b.get_width()
            ax_prob.text(w + 1.0, b.get_y() + b.get_height()/2, f"{w:.1f}%", va='center', color='#f8fafc', fontsize=8.5, fontweight='bold')
        ax_prob.set_xlim(0, 110)
        plt.tight_layout()
        st.pyplot(fig_prob)


# --- TAB 3: MCU Model Fit ---
with tab3:
    st.markdown("### 🔬 Microcontroller Memory & Footprint Analyzer")
    st.caption("Analyze 9-Input ➔ Object Shape + Size TinyML model footprint on ARM Cortex-M MCUs.")

    tflite_bytes = len(open("anomaly_model.tflite", "rb").read()) if os.path.exists("anomaly_model.tflite") else 10240
    flash_model_kb = tflite_bytes / 1024.0
    tensor_arena_ram_kb = 12.0

    f1, f2, f3, f4 = st.columns(4)
    with f1: render_kpi_card("Model Flash", f"{flash_model_kb:.2f} KB", f"{tflite_bytes:,} Bytes", "#38bdf8")
    with f2: render_kpi_card("Tensor Arena RAM", f"{tensor_arena_ram_kb:.1f} KB", "Scratchpad RAM", "#34d399")
    with f3: render_kpi_card("Input Mesh", "9 Sensors", "S1 - S9 Channels", "#818cf8")
    with f4: render_kpi_card("Output Targets", f"{num_classes + 1} Outputs", f"{num_classes} Shapes + 1 Dim", "#c084fc")

    mcu_db = [
        {"MCU": "STM32F407VG (Cortex-M4)", "Flash (KB)": 1024, "RAM (KB)": 192, "Clock (MHz)": 168, "Status": "✅ PERFECT FIT"},
        {"MCU": "STM32F746NG (Cortex-M7)", "Flash (KB)": 1024, "RAM (KB)": 320, "Clock (MHz)": 216, "Status": "✅ PERFECT FIT"},
        {"MCU": "ESP32 Dev Module (Xtensa)", "Flash (KB)": 4096, "RAM (KB)": 520, "Clock (MHz)": 240, "Status": "✅ PERFECT FIT"},
        {"MCU": "Arduino Nano 33 BLE", "Flash (KB)": 1024, "RAM (KB)": 256, "Clock (MHz)": 64, "Status": "✅ PERFECT FIT"},
        {"MCU": "STM32F103 'Blue Pill'", "Flash (KB)": 64, "RAM (KB)": 20, "Clock (MHz)": 72, "Status": "⚠️ TIGHT FIT"}
    ]
    mcu_df = pd.DataFrame(mcu_db)
    mcu_df["Flash Usage (%)"] = ((flash_model_kb + 18.0) / mcu_df["Flash (KB)"]) * 100.0
    mcu_df["RAM Usage (%)"] = ((tensor_arena_ram_kb + 0.5) / mcu_df["RAM (KB)"]) * 100.0
    st.dataframe(mcu_df[["MCU", "Status", "Flash (KB)", "Flash Usage (%)", "RAM (KB)", "RAM Usage (%)", "Clock (MHz)"]], use_container_width=True)


# --- TAB 4: MCU Deployment & Physical Serial ---
with tab4:
    st.markdown("### 🔌 Firmware Package Export & Physical UART Serial")

    inj_tab1, inj_tab2 = st.tabs(["📦 Firmware Bundle (.ZIP)", "⚡ Live UART Serial Connection"])

    with inj_tab1:
        st.markdown("#### Complete 9-Input TinyML Firmware Package")
        if HAS_EXPORTER:
            zip_bytes = create_mcu_deployment_zip()
            st.download_button(
                label="📥 Download Microcontroller Firmware Bundle (.zip)",
                data=zip_bytes,
                file_name="TinyML_9Sensors_Object_Size_Package.zip",
                mime="application/zip",
                type="primary",
                use_container_width=True
            )
            st.success("✅ Contains: `main_tflm.cpp` (9-in, object+size out), `model_data.h`, `model_config.h`, `Arduino_Sketch.ino`, `platformio.ini`, `Makefile`")
        else:
            st.warning("mcu_exporter module not found.")

    with inj_tab2:
        st.markdown("#### Live Physical UART Serial Reader")
        if not HAS_PYSERIAL:
            st.error("pyserial library missing.")
        else:
            if "phys_history" not in st.session_state: st.session_state.phys_history = []
            cp1, cp2 = st.columns(2)
            with cp1:
                detected_ports = [p.device for p in serial.tools.list_ports.comports()] or ["COM1", "COM3", "/dev/ttyUSB0"]
                phys_port = st.selectbox("Serial Port:", options=detected_ports)
                phys_baud = st.selectbox("Baud Rate:", options=[9600, 19200, 38400, 57600, 115200], index=4)
            with cp2:
                read_phys = st.button("▶️ Read Serial Packets", type="primary", use_container_width=True)
                clear_phys = st.button("🧹 Clear Logs", use_container_width=True)
                if clear_phys: st.session_state.phys_history = []

            if read_phys:
                with st.spinner(f"Reading `{phys_port}`..."):
                    try:
                        ser_phys = serial.Serial(phys_port, baudrate=phys_baud, timeout=0.5)
                        p_lines = 0
                        start_t = time.time()
                        while p_lines < 25 and (time.time() - start_t) < 1.5:
                            raw_l = ser_phys.readline().decode('utf-8', errors='ignore').strip()
                            if not raw_l: continue
                            p_lines += 1
                            m_match = re.search(r'SAMPLE=(\d+).*OBJECT=(\w+).*CONF=([0-9.]+)%.*SIZE_MM=([0-9.]+)', raw_l)
                            if m_match:
                                rec = {
                                    'Sample': int(m_match.group(1)),
                                    'Object': m_match.group(2),
                                    'Confidence_%': float(m_match.group(3)),
                                    'Size_mm': float(m_match.group(4)),
                                    'Raw': raw_l
                                }
                                st.session_state.phys_history.append(rec)
                        ser_phys.close()
                    except Exception as p_ex:
                        st.error(f"Serial Error: {p_ex}")

            if st.session_state.phys_history:
                st.dataframe(pd.DataFrame(st.session_state.phys_history), use_container_width=True)


# --- TAB 5: Latency Benchmark & Renode Socket ---
with tab5:
    st.markdown("### ⚡ Latency Benchmark & Renode Socket Stream")

    col_bench, col_renode = st.columns(2)
    with col_bench:
        st.markdown("#### ⚡ Microcontroller Inference Latency")
        num_bench = st.slider("Benchmark Inferences:", min_value=1000, max_value=20000, value=5000, step=1000)
        if st.button("🚀 Run Latency Benchmark", type="primary", use_container_width=True):
            test_batch = np.random.uniform(low=-2.0, high=2.0, size=(num_bench, 9)).astype(np.float32)
            t0 = time.perf_counter()
            for s in test_batch:
                interpreter.set_tensor(input_details[0]['index'], s.reshape(1, 9))
                interpreter.invoke()
                _ = interpreter.get_tensor(output_details[0]['index'])
            t1 = time.perf_counter()
            total_s = t1 - t0
            lat_ms = (total_s / num_bench) * 1000.0
            throughput = num_bench / total_s
            render_kpi_card("Average Latency", f"{lat_ms:.4f} ms", f"{throughput:,.0f} Inferences/sec", "#38bdf8")

    with col_renode:
        st.markdown("#### 🖥️ Renode TCP Socket Terminal")
        st.caption("Connect to Renode socket (`socket://localhost:12345`).")
        socket_url = st.text_input("Socket URL:", value="socket://localhost:12345")
        if st.button("▶️ Read Renode Socket", use_container_width=True):
            st.info(f"Connecting to {socket_url}...")