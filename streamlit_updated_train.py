import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error, r2_score, confusion_matrix
import matplotlib.pyplot as plt
import joblib
import os
import re

# Page Configuration
st.set_page_config(
    page_title="9-Sensor Mesh Classifier & Sizing Trainer",
    page_icon="🔬",
    layout="wide"
)

# Custom Dark Styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .stApp { background: linear-gradient(135deg, #090d16 0%, #0f172a 50%, #090d16 100%); color: #f1f5f9; }

    /* Full-width layout — no side padding constraints */
    div.block-container {
        padding-top: 1.2rem !important;
        padding-left: 1.5rem !important;
        padding-right: 1.5rem !important;
        padding-bottom: 2rem !important;
        max-width: 100% !important;
    }
    .main .block-container {
        max-width: 100% !important;
        width: 100% !important;
    }
    /* Expand dataframes to full width */
    div[data-testid="stDataFrame"] > div { width: 100% !important; }

    .kpi-box {
        background: linear-gradient(145deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.8) 100%);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 14px 18px;
        backdrop-filter: blur(8px);
    }
    .kpi-label { font-size: 11px; text-transform: uppercase; color: #94a3b8; font-weight: 600; letter-spacing: 0.8px; }
    .kpi-val { font-size: 24px; font-weight: 700; color: #f8fafc; font-family: 'JetBrains Mono', monospace; }
    .kpi-sub { font-size: 11px; color: #cbd5e1; margin-top: 2px; }
</style>
""", unsafe_allow_html=True)

st.title("🔬 9-Sensor Mesh Classifier & Sizing Trainer")
st.markdown("""
**Mentor Specification Architecture**:
- **Inputs**: Strictly the **9 physical sensor outputs** (`S1` ... `S9`) captured from the sensor mesh. Auxiliary/extraneous columns are excluded from training.
- **Filters & Data-Size Reduction**: Interactive filtering by `object_shape`, dimension range (`dim_mm`), and sample downsampling to reduce dataset size.
- **9 Legends & Sensor Correlation**: Plot sensor curves with **9 distinct legends** and compute the $9 \\times 9$ sensor correlation matrix across chosen shapes and dimensions.
- **Predict Object & Size**: Deep TinyML multi-task neural network that predicts **Which Object** (`object_shape` classification) and **What Size** (`dim_mm` regression) when running over all data.
""")
st.markdown("---")

def find_available_datasets():
    files = [f for f in os.listdir('.') if (f.endswith('.xlsx') or f.endswith('.csv')) and not f.startswith('~$')]
    return files if files else None

def generate_c_header(cols_s, shape_classes, sensor_means, sensor_stds, filepath="model_config.h"):
    num_inputs = len(cols_s)
    num_classes = len(shape_classes)
    num_outputs = num_classes + 1
    
    in_means_str = ", ".join([f"{m:.6f}f" for m in sensor_means])
    in_stds_str = ", ".join([f"{s:.6f}f" for s in sensor_stds])
    class_names_str = ", ".join([f'"{c}"' for c in shape_classes])
    
    header_content = f"""/*
 * Auto-Generated TinyML Model Configuration Header
 * Target: STM32 Microcontroller / ARM Cortex-M Firmware
 * Architecture: 9 Sensor Inputs (S1-S9) -> Object Shape ({num_classes} classes) + Size (dim_mm)
 */

#ifndef MODEL_CONFIG_H
#define MODEL_CONFIG_H

#define NUM_INPUTS {num_inputs}
#define NUM_OUTPUTS {num_outputs}
#define NUM_SHAPE_CLASSES {num_classes}

/*
 * Sensor Normalization Parameters (S1 ... S9)
 */
static const float INPUT_MEANS[{num_inputs}] = {{ {in_means_str} }};
static const float INPUT_STDS[{num_inputs}] = {{ {in_stds_str} }};

/*
 * Shape Class Labels ({num_classes}):
 */
static const char* const SHAPE_CLASSES[{num_classes}] = {{ {class_names_str} }};

/*
 * Model Output Format ({num_outputs} floats):
 * - Indices [0 .. {num_classes - 1}]: Shape probabilities (Softmax: {", ".join(shape_classes)})
 * - Index [{num_classes}]: Predicted Dimension (dim_mm in mm)
 */

#endif /* MODEL_CONFIG_H */
"""
    with open(filepath, "w") as f:
        f.write(header_content)
    return header_content

# --- STEP 1: DATASET INGESTION & FILTERING (REDUCE DATA-SIZE) ---
st.subheader("1. Ingestion, Filtering & Data-Size Reduction")

col_d1, col_d2 = st.columns([1, 1])
with col_d1:
    uploaded_file = st.file_uploader("Upload Dataset (.csv / .xlsx):", type=["csv", "xlsx"])
    available_files = find_available_datasets()
    if uploaded_file is not None:
        file_source = uploaded_file
        file_name = uploaded_file.name
    elif available_files:
        default_file_idx = available_files.index("sensor_design_dataset.csv") if "sensor_design_dataset.csv" in available_files else 0
        file_name = st.selectbox("Select Workspace Dataset File:", options=available_files, index=default_file_idx)
        file_source = file_name
    else:
        file_source = None

    if file_source is None:
        st.error("No dataset found! Please upload a `.csv` or `.xlsx` file.")
        st.stop()

    try:
        if file_name.endswith('.csv'):
            df_raw = pd.read_csv(file_source)
        else:
            xls = pd.ExcelFile(file_source)
            selected_sheet = st.selectbox("Select Sheet:", options=xls.sheet_names)
            df_raw = pd.read_excel(xls, sheet_name=selected_sheet)
    except Exception as e:
        st.error(f"Error reading dataset: {e}")
        st.stop()

# Auto-detect strictly the 9 sensor output columns (S1 - S9)
cols_s = [c for c in df_raw.columns if re.match(r'^s[1-9]$', c, re.IGNORECASE)]
if len(cols_s) < 9:
    cols_s = [f'S{i}' for i in range(1, 10) if f'S{i}' in df_raw.columns]
if len(cols_s) < 9:
    numeric_cols = list(df_raw.select_dtypes(include=[np.number]).columns)
    cols_s = numeric_cols[:9]

# Detect Shape and Dimension columns
shape_col = 'object_shape' if 'object_shape' in df_raw.columns else next((c for c in df_raw.columns if 'shape' in c.lower()), df_raw.columns[1])
dim_col = 'dim_mm' if 'dim_mm' in df_raw.columns else next((c for c in df_raw.columns if 'dim' in c.lower() or 'size' in c.lower()), df_raw.columns[2])

with col_d2:
    st.markdown("#### 🎯 Filter & Reduce Data-Size Options")
    
    # 1. Filter by Object Shape
    all_shapes = sorted(list(df_raw[shape_col].dropna().unique()))
    selected_shapes = st.multiselect("Filter by Object Shape:", options=all_shapes, default=all_shapes)
    if not selected_shapes:
        st.warning("Please select at least one Object Shape.")
        st.stop()

    # 2. Filter by Dimension (dim_mm)
    min_dim = float(df_raw[dim_col].min())
    max_dim = float(df_raw[dim_col].max())
    dim_range = st.slider("Filter Dimension Range `dim_mm` (mm):", min_value=round(min_dim, 1), max_value=round(max_dim, 1), value=(round(min_dim, 1), round(max_dim, 1)), step=0.5)

    # 3. Reduce Data-Size (Downsampling & Max Sample Cap)
    col_sub1, col_sub2 = st.columns(2)
    with col_sub1:
        downsample_step = st.selectbox("Sample Step Rate (Reduce Size):", options=[1, 2, 3, 5, 10], format_func=lambda x: f"Every {x} sample" if x > 1 else "Full 1:1 (All samples)")
    with col_sub2:
        max_sample_limit = st.slider("Max Training Samples Cap:", min_value=50, max_value=len(df_raw), value=min(1500, len(df_raw)), step=50)

# Apply Filters & Reduction
df_filtered = df_raw[
    (df_raw[shape_col].isin(selected_shapes)) &
    (df_raw[dim_col] >= dim_range[0]) &
    (df_raw[dim_col] <= dim_range[1])
].copy()

if downsample_step > 1:
    df_filtered = df_filtered.iloc[::downsample_step].copy()

if len(df_filtered) > max_sample_limit:
    df_filtered = df_filtered.head(max_sample_limit).copy()

orig_count = len(df_raw)
filt_count = len(df_filtered)
reduction_pct = ((orig_count - filt_count) / orig_count) * 100.0

k1, k2, k3, k4 = st.columns(4)
with k1:
    st.markdown(f'<div class="kpi-box" style="border-left: 4px solid #38bdf8;"><div class="kpi-label">Filtered Data Size</div><div class="kpi-val">{filt_count:,}</div><div class="kpi-sub">Original: {orig_count:,}</div></div>', unsafe_allow_html=True)
with k2:
    st.markdown(f'<div class="kpi-box" style="border-left: 4px solid #34d399;"><div class="kpi-label">Data Reduction</div><div class="kpi-val">{reduction_pct:.1f}%</div><div class="kpi-sub">Memory & Train Optimized</div></div>', unsafe_allow_html=True)
with k3:
    st.markdown(f'<div class="kpi-box" style="border-left: 4px solid #818cf8;"><div class="kpi-label">Sensor Channels</div><div class="kpi-val">{len(cols_s)}</div><div class="kpi-sub">{", ".join(cols_s[:5])}...</div></div>', unsafe_allow_html=True)
with k4:
    st.markdown(f'<div class="kpi-box" style="border-left: 4px solid #c084fc;"><div class="kpi-label">Target Shapes</div><div class="kpi-val">{len(selected_shapes)}</div><div class="kpi-sub">{", ".join(selected_shapes)}</div></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# --- STEP 2: 9 SENSORS VISUALIZATION (9 LEGENDS) & CORRELATION ANALYSIS ---
st.subheader("2. Multi-Sensor Visualization (9 Legends) & Correlation Analysis")
st.caption("Visualizing the 9 physical sensors captured from the sensor mesh, with 9 legends and correlation between sensors for different shapes & dimensions.")

tab_viz1, tab_viz2 = st.tabs(["📈 9-Sensor Curves (9 Legends)", "🔥 Sensor-to-Sensor Correlation Matrix"])

colors_9 = ['#38bdf8', '#818cf8', '#c084fc', '#f472b6', '#fb7185', '#fb923c', '#facc15', '#4ade80', '#2dd4bf']

with tab_viz1:
    fig_sensors, ax_sens = plt.subplots(figsize=(18, 4.5), facecolor='#0f172a')
    ax_sens.set_facecolor('#0b1120')
    
    # Plot each sensor with its explicit legend label
    plot_df = df_filtered.head(150).reset_index(drop=True)
    for idx, col in enumerate(cols_s[:9]):
        ax_sens.plot(plot_df.index, plot_df[col], label=f'{col}', color=colors_9[idx % len(colors_9)], linewidth=1.8, alpha=0.9)

    ax_sens.set_title(f"9 Sensor Output Signals (S1 - S9) for Shapes: {', '.join(selected_shapes)} | Dims: {dim_range[0]}-{dim_range[1]}mm", fontsize=11, fontweight='bold', color='#f1f5f9', pad=12)
    ax_sens.set_xlabel("Filtered Sample Index", fontsize=10, color='#94a3b8')
    ax_sens.set_ylabel("Sensor Amplitude", fontsize=10, color='#94a3b8')
    ax_sens.tick_params(colors='#64748b')
    ax_sens.grid(True, linestyle='--', alpha=0.25, color='#334155')
    for spine in ax_sens.spines.values(): spine.set_color('#1e293b')
    
    # 9 Legends clearly displayed
    ax_sens.legend(loc="upper right", facecolor='#0f172a', edgecolor='#334155', fontsize=9, ncol=5, title="9 Sensor Legends")
    plt.tight_layout()
    st.pyplot(fig_sensors)

with tab_viz2:
    st.markdown(f"#### 🔍 Correlation Between 9 Sensor Values (`S1` ... `S9`)")
    corr_matrix = df_filtered[cols_s[:9]].corr()
    
    fig_corr, ax_corr = plt.subplots(figsize=(13, 7), facecolor='#0f172a')
    ax_corr.set_facecolor('#0b1120')
    
    cax = ax_corr.imshow(corr_matrix.values, cmap='coolwarm', vmin=-1.0, vmax=1.0)
    cbar = fig_corr.colorbar(cax, ax=ax_corr, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(colors='#94a3b8')
    
    ax_corr.set_xticks(range(len(cols_s[:9])))
    ax_corr.set_yticks(range(len(cols_s[:9])))
    ax_corr.set_xticklabels(cols_s[:9], color='#f1f5f9', fontsize=9, fontweight='600')
    ax_corr.set_yticklabels(cols_s[:9], color='#f1f5f9', fontsize=9, fontweight='600')
    
    for i in range(len(cols_s[:9])):
        for j in range(len(cols_s[:9])):
            val = corr_matrix.iloc[i, j]
            txt_color = '#ffffff' if abs(val) > 0.4 else '#cbd5e1'
            ax_corr.text(j, i, f"{val:.2f}", ha='center', va='center', color=txt_color, fontsize=8, fontweight='bold')

    ax_corr.set_title(f"Sensor-to-Sensor Correlation Matrix ({', '.join(selected_shapes)}, {dim_range[0]}-{dim_range[1]}mm)", fontsize=11, fontweight='bold', color='#f1f5f9', pad=12)
    plt.tight_layout()
    st.pyplot(fig_corr)

st.markdown("---")

# --- STEP 3: MODEL TRAINING (9 SENSORS -> OBJECT SHAPE & SIZE) ---
st.subheader("3. Train Multi-Task Model (9 Sensors ➔ Object Shape + Dimension Size)")

col_t1, col_t2 = st.columns([1, 1])
with col_t1:
    epochs = st.slider("Training Epochs:", min_value=15, max_value=120, value=45, step=5)
    batch_size = st.selectbox("Batch Size:", options=[16, 32, 64, 128], index=1)
    learning_rate = st.select_slider("Learning Rate:", options=[0.0005, 0.001, 0.002, 0.005], value=0.002)

with col_t2:
    st.markdown("#### Model Architecture Overview")
    st.markdown(f"""
    - **Input Vector (9)**: Sensor channels `{", ".join(cols_s[:9])}` (Normalized)
    - **Backbone**: Dense(64, ReLU) ➔ BatchNorm ➔ Dropout(0.1) ➔ Dense(64, ReLU) ➔ Dense(32, ReLU)
    - **Head 1 (Object Shape)**: Dense({len(selected_shapes)}, Softmax) ➔ Categorical Crossentropy
    - **Head 2 (Size / Dim)**: Dense(1, Linear) ➔ Mean Squared Error (dim_mm)
    """)
    train_clicked = st.button("🚀 Train Model & Run All Data Prediction", type="primary", use_container_width=True)

# Preprocessing arrays
df_train_clean = df_filtered[cols_s[:9] + [shape_col, dim_col]].dropna().copy()
X_raw = df_train_clean[cols_s[:9]].values.astype(np.float32)

scaler_sensors = StandardScaler()
X_scaled = scaler_sensors.fit_transform(X_raw).astype(np.float32)

shape_encoder = LabelEncoder()
y_shape = shape_encoder.fit_transform(df_train_clean[shape_col].astype(str))
num_classes = len(shape_encoder.classes_)
y_dim = df_train_clean[dim_col].values.astype(np.float32)

if train_clicked:
    with st.spinner("Training Dual-Task Deep Neural Network on 9 Sensor Inputs..."):
        # Build Dual-Head Keras Model
        inputs = layers.Input(shape=(9,), name='sensor_inputs')
        x = layers.Dense(64, activation='relu')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.1)(x)
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dense(32, activation='relu')(x)

        shape_out = layers.Dense(num_classes, activation='softmax', name='shape_output')(x)
        dim_out = layers.Dense(1, activation='linear', name='dim_output')(x)

        training_model = models.Model(inputs=inputs, outputs=[shape_out, dim_out])
        training_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss={'shape_output': 'sparse_categorical_crossentropy', 'dim_output': 'mse'},
            loss_weights={'shape_output': 1.0, 'dim_output': 0.05},
            metrics={'shape_output': ['accuracy'], 'dim_output': ['mae']}
        )

        history = training_model.fit(
            X_scaled, [y_shape, y_dim],
            validation_split=0.15,
            epochs=epochs,
            batch_size=batch_size,
            verbose=0
        )

        # Combined Export Model: [p_0 ... p_K-1, dim_mm]
        combined_out = layers.Concatenate(axis=-1, name='predictions')([shape_out, dim_out])
        export_model = models.Model(inputs=inputs, outputs=combined_out)

        # Run All Data Prediction
        preds_all = export_model.predict(X_scaled, verbose=0)
        pred_shape_probs = preds_all[:, :num_classes]
        pred_shape_idx = np.argmax(pred_shape_probs, axis=1)
        pred_shape_names = shape_encoder.inverse_transform(pred_shape_idx)
        pred_confidences = np.max(pred_shape_probs, axis=1) * 100.0
        pred_dims = preds_all[:, -1]

        # Performance Metrics
        acc = accuracy_score(y_shape, pred_shape_idx)
        mae = mean_absolute_error(y_dim, pred_dims)
        rmse = np.sqrt(mean_squared_error(y_dim, pred_dims))
        r2 = r2_score(y_dim, pred_dims)

        # Export TFLite Model
        converter = tf.lite.TFLiteConverter.from_keras_model(export_model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_model = converter.convert()
        with open("anomaly_model.tflite", "wb") as f:
            f.write(tflite_model)

        # Export Metadata
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
            'dim_rmse': float(rmse),
            'dim_r2': float(r2),
            'num_inputs': 9,
            'num_classes': num_classes,
            'num_outputs': num_classes + 1
        }
        joblib.dump(metadata, 'model_metadata.pkl')

        # Generate C Header
        generate_c_header(cols_s[:9], shape_encoder.classes_, scaler_sensors.mean_, np.sqrt(scaler_sensors.var_), "model_config.h")

        # Convert to model_data.h
        try:
            from convert_model import convert_tflite_to_c_header
            convert_tflite_to_c_header("anomaly_model.tflite", "model_data.h")
        except Exception:
            pass

        st.success(f"✅ Training Complete! Trained on 9 Sensor Inputs ➔ Exported TinyML Model ({len(tflite_model):,} bytes)!")

        # --- STEP 4: RUN ALL DATA & PREDICT OBJECT AND SIZE RESULTS ---
        st.markdown("---")
        st.subheader("4. Run All Data & Prediction Results (Object & Size)")
        st.caption("Predictions run across all samples: identifying object shape and dimension size from sensor values.")

        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.markdown(f'<div class="kpi-box" style="border-left: 4px solid #34d399;"><div class="kpi-label">Shape Accuracy</div><div class="kpi-val">{acc*100.0:.1f}%</div><div class="kpi-sub">Correct Object ID</div></div>', unsafe_allow_html=True)
        with m2:
            st.markdown(f'<div class="kpi-box" style="border-left: 4px solid #38bdf8;"><div class="kpi-label">Size MAE</div><div class="kpi-val">{mae:.2f} mm</div><div class="kpi-sub">Mean Abs Error</div></div>', unsafe_allow_html=True)
        with m3:
            st.markdown(f'<div class="kpi-box" style="border-left: 4px solid #f59e0b;"><div class="kpi-label">Size RMSE</div><div class="kpi-val">{rmse:.2f} mm</div><div class="kpi-sub">Root Mean Sq Error</div></div>', unsafe_allow_html=True)
        with m4:
            st.markdown(f'<div class="kpi-box" style="border-left: 4px solid #c084fc;"><div class="kpi-label">Size R² Score</div><div class="kpi-val">{r2:.3f}</div><div class="kpi-sub">Regression Fit</div></div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Visual Evaluation Plots
        p_col1, p_col2 = st.columns(2)
        with p_col1:
            # Confusion Matrix Heatmap
            cm = confusion_matrix(y_shape, pred_shape_idx)
            fig_cm, ax_cm = plt.subplots(figsize=(8, 5.5), facecolor='#0f172a')
            ax_cm.set_facecolor('#0b1120')
            cax_cm = ax_cm.imshow(cm, cmap='Blues')
            fig_cm.colorbar(cax_cm, ax=ax_cm, fraction=0.046, pad=0.04)
            ax_cm.set_xticks(range(num_classes))
            ax_cm.set_yticks(range(num_classes))
            ax_cm.set_xticklabels(shape_encoder.classes_, color='#f1f5f9', fontsize=9)
            ax_cm.set_yticklabels(shape_encoder.classes_, color='#f1f5f9', fontsize=9)
            ax_cm.set_xlabel("Predicted Object", color='#94a3b8', fontsize=9)
            ax_cm.set_ylabel("Actual Object", color='#94a3b8', fontsize=9)
            ax_cm.set_title("Confusion Matrix (Object Classification)", color='#f1f5f9', fontsize=10, fontweight='bold', pad=10)
            
            for i in range(num_classes):
                for j in range(num_classes):
                    ax_cm.text(j, i, str(cm[i, j]), ha='center', va='center', color='#ffffff' if cm[i, j] > cm.max()/2 else '#cbd5e1', fontweight='bold')
            plt.tight_layout()
            st.pyplot(fig_cm)

        with p_col2:
            # Scatter Plot: Actual Size vs Predicted Size
            fig_reg, ax_reg = plt.subplots(figsize=(8, 5.5), facecolor='#0f172a')
            ax_reg.set_facecolor('#0b1120')
            for cls_idx, cls_name in enumerate(shape_encoder.classes_):
                mask = (y_shape == cls_idx)
                ax_reg.scatter(y_dim[mask], pred_dims[mask], label=cls_name, alpha=0.75, s=28, color=colors_9[cls_idx % len(colors_9)])
            
            # Identity line
            min_v = min(y_dim.min(), pred_dims.min())
            max_v = max(y_dim.max(), pred_dims.max())
            ax_reg.plot([min_v, max_v], [min_v, max_v], color='#ef4444', linestyle='--', linewidth=2, label='Ideal y=x')
            ax_reg.set_xlabel("Actual Dimension (mm)", color='#94a3b8', fontsize=9)
            ax_reg.set_ylabel("Predicted Dimension (mm)", color='#94a3b8', fontsize=9)
            ax_reg.set_title(f"Dimension Prediction (R² = {r2:.3f}, MAE = {mae:.2f}mm)", color='#f1f5f9', fontsize=10, fontweight='bold', pad=10)
            ax_reg.legend(loc="upper left", facecolor='#0f172a', edgecolor='#334155', fontsize=8)
            ax_reg.grid(True, linestyle='--', alpha=0.25, color='#334155')
            ax_reg.tick_params(colors='#64748b')
            plt.tight_layout()
            st.pyplot(fig_reg)

        # Detailed Predictions Table
        st.markdown("#### 📋 Detailed Predictions Table (All Samples)")
        results_df = df_train_clean.copy()
        results_df['Actual_Object'] = df_train_clean[shape_col]
        results_df['Predicted_Object'] = pred_shape_names
        results_df['Shape_Match'] = ["✅ MATCH" if a == p else "❌ MISMATCH" for a, p in zip(results_df['Actual_Object'], results_df['Predicted_Object'])]
        results_df['Shape_Confidence_%'] = np.round(pred_confidences, 1)
        results_df['Actual_Size_mm'] = np.round(y_dim, 2)
        results_df['Predicted_Size_mm'] = np.round(pred_dims, 2)
        results_df['Size_Error_mm'] = np.round(np.abs(y_dim - pred_dims), 2)

        view_cols = ['Actual_Object', 'Predicted_Object', 'Shape_Match', 'Shape_Confidence_%', 'Actual_Size_mm', 'Predicted_Size_mm', 'Size_Error_mm'] + cols_s[:9]
        st.dataframe(results_df[view_cols], use_container_width=True)

        # Download Predictions CSV
        csv_data = results_df[view_cols].to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download All Predictions as CSV",
            data=csv_data,
            file_name="9sensors_object_and_size_predictions.csv",
            mime="text/csv"
        )