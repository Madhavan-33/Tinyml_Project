"""
Retrain TinyML Multi-Task Neural Network on 9-Sensor Tactile Mesh Dataset
========================================================================
Trains on the 9-sensor mesh dataset (S1-S9 physical range: 1.60V to 2.28V).
Exports:
  - anomaly_model.tflite       (TFLite for edge inference)
  - model_metadata.pkl         (scaler + encoder + stats)
  - model_weights.json         (weights for JS/serverless inference)
  - api/model_weights.json     (Vercel API copy)
  - public/model_weights.json  (browser-side JS copy)
  - model_config.h             (C header for STM32/Arduino firmware)
  - model_data.h               (C byte array for TFLM micro firmware)

Usage:
  python retrain_on_telemetry.py --dataset sensor_design_dataset.csv
"""

import argparse
import os
import re
import json
import joblib
import sys

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, mean_absolute_error, r2_score, confusion_matrix
from convert_model import convert_tflite_to_c_header

# ── Physical sensor operating range ─────────────────────────
SENSOR_PHYSICAL_MIN = 1.60   # V
SENSOR_PHYSICAL_MAX = 2.28   # V
SENSOR_PHYSICAL_MEAN = 2.15  # V

SENSOR_COLS_PATTERN = re.compile(r'^s[1-9]$', re.IGNORECASE)
EXPECTED_SENSOR_COUNT = 9


def find_sensor_columns(df):
    cols = [c for c in df.columns if SENSOR_COLS_PATTERN.match(c)]
    if len(cols) >= EXPECTED_SENSOR_COUNT:
        return cols[:EXPECTED_SENSOR_COUNT]
    cols = [f'S{i}' for i in range(1, 10) if f'S{i}' in df.columns]
    if len(cols) >= EXPECTED_SENSOR_COUNT:
        return cols[:EXPECTED_SENSOR_COUNT]
    numeric = list(df.select_dtypes(include=[np.number]).columns)
    if len(numeric) >= EXPECTED_SENSOR_COUNT:
        return numeric[:EXPECTED_SENSOR_COUNT]
    raise ValueError(
        f"Could not find {EXPECTED_SENSOR_COUNT} sensor columns. "
        f"Columns: {list(df.columns)}"
    )


def find_target_columns(df):
    shape_col = 'object_shape' if 'object_shape' in df.columns else next(
        (c for c in df.columns if 'shape' in c.lower()), None)
    dim_col = 'dim_mm' if 'dim_mm' in df.columns else next(
        (c for c in df.columns if 'dim' in c.lower() or 'size' in c.lower()), None)
    if shape_col is None or dim_col is None:
        raise ValueError(
            "Could not find shape or dimension columns. "
            f"Columns: {list(df.columns)}"
        )
    return shape_col, dim_col


def load_dataset(path_or_df):
    if isinstance(path_or_df, pd.DataFrame):
        return path_or_df.copy()
    if hasattr(path_or_df, 'read'):
        try:
            path_or_df.seek(0)
        except Exception:
            pass
        try:
            return pd.read_csv(path_or_df)
        except Exception:
            try:
                path_or_df.seek(0)
            except Exception:
                pass
            return pd.read_excel(path_or_df)
    if not isinstance(path_or_df, str) or not os.path.exists(path_or_df):
        raise FileNotFoundError(f"Dataset not found: {path_or_df}")
    if path_or_df.endswith('.csv'):
        return pd.read_csv(path_or_df)
    elif path_or_df.endswith(('.xlsx', '.xls')):
        return pd.read_excel(path_or_df)
    raise ValueError(f"Unsupported file format: {path_or_df}")


def generate_c_header(cols_s, shape_classes, sensor_means, sensor_stds,
                      filepath="model_config.h"):
    num_inputs = len(cols_s)
    num_classes = len(shape_classes)
    num_outputs = num_classes + 1
    in_means_str = ", ".join([f"{m:.6f}f" for m in sensor_means])
    in_stds_str = ", ".join([f"{s:.6f}f" for s in sensor_stds])
    class_names_str = ", ".join([f'"{c}"' for c in shape_classes])
    content = f"""/*
 * Auto-Generated TinyML Model Configuration Header
 * 9-Sensor Tactile Mesh (S1-S9: {SENSOR_PHYSICAL_MIN}V - {SENSOR_PHYSICAL_MAX}V)
 * Target: STM32 / ARM Cortex-M Firmware
 */
#ifndef MODEL_CONFIG_H
#define MODEL_CONFIG_H

#define NUM_INPUTS          {num_inputs}
#define NUM_OUTPUTS         {num_outputs}
#define NUM_SHAPE_CLASSES   {num_classes}

/* Sensor Normalization Parameters (StandardScaler fit on dataset) */
static const float INPUT_MEANS[{num_inputs}] = {{ {in_means_str} }};
static const float INPUT_STDS[{num_inputs}]  = {{ {in_stds_str} }};

/* Shape Class Labels */
static const char* const SHAPE_CLASSES[{num_classes}] = {{ {class_names_str} }};

/*
 * Output: [{num_outputs} floats]
 *   [0..{num_classes-1}] = Softmax probs ({', '.join(shape_classes)})
 *   [{num_classes}]      = Predicted dim_mm
 */
#endif /* MODEL_CONFIG_H */
"""
    with open(filepath, "w") as f:
        f.write(content)
    return content


def build_dual_head_model(num_classes):
    tf.random.set_seed(42)
    np.random.seed(42)

    inp = layers.Input(shape=(9,), name='sensor_inputs')

    # Backbone: 9 -> 128 -> BN -> 128 -> BN -> 64 -> 32
    x = layers.Dense(128, activation='relu', name='dense_1')(inp)
    x = layers.BatchNormalization(name='bn_1')(x)
    x = layers.Dense(128, activation='relu', name='dense_2')(x)
    x = layers.BatchNormalization(name='bn_2')(x)
    x = layers.Dense(64, activation='relu', name='dense_3')(x)
    x = layers.Dense(32, activation='relu', name='dense_4')(x)

    shape_out = layers.Dense(num_classes, activation='softmax', name='shape_output')(x)
    dim_out = layers.Dense(1, activation='linear', name='dim_output')(x)

    model = models.Model(inputs=inp, outputs=[shape_out, dim_out])
    return model, shape_out, dim_out


def train(dataset_file, epochs=50, batch_size=32, lr=0.002,
          max_samples=None, verbose=1):
    print(f"\n{'='*60}")
    print(f"  TinyML 9-Sensor Multi-Task Retraining Pipeline")
    print(f"{'='*60}")
    print(f"Dataset : {dataset_file}")
    print(f"Epochs  : {epochs}  |  Batch: {batch_size}  |  LR: {lr}")

    # 1. Load dataset
    df = load_dataset(dataset_file)
    print(f"Loaded {len(df):,} rows x {len(df.columns)} cols")

    cols_s = find_sensor_columns(df)
    shape_col, dim_col = find_target_columns(df)
    print(f"Sensor cols: {cols_s}")
    print(f"Shape col: '{shape_col}' | Dim col: '{dim_col}'")

    df_clean = df[cols_s + [shape_col, dim_col]].dropna().copy()
    print(f"Clean rows: {len(df_clean):,}")

    if max_samples and len(df_clean) > max_samples:
        df_clean = df_clean.sample(n=max_samples, random_state=42).reset_index(drop=True)
        print(f"Sampled to {len(df_clean):,} rows")

    # 2. Sensor statistics
    X_raw = df_clean[cols_s].values.astype(np.float32)
    overall_min = float(X_raw.min())
    overall_max = float(X_raw.max())
    overall_mean = float(X_raw.mean())
    print(f"Sensor overall: min={overall_min:.4f}V, max={overall_max:.4f}V, mean={overall_mean:.4f}V")

    # 3. Preprocessing
    scaler_sensors = StandardScaler()
    X_scaled = scaler_sensors.fit_transform(X_raw).astype(np.float32)

    shape_encoder = LabelEncoder()
    y_shape = shape_encoder.fit_transform(df_clean[shape_col].astype(str))
    num_classes = len(shape_encoder.classes_)
    y_dim = df_clean[dim_col].values.astype(np.float32)

    print(f"Classes ({num_classes}): {list(shape_encoder.classes_)}")
    print(f"Dimension Range: {y_dim.min():.1f}mm to {y_dim.max():.1f}mm (mean {y_dim.mean():.1f}mm)")

    # 4. Build and train
    training_model, shape_out, dim_out = build_dual_head_model(num_classes)
    training_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss={'shape_output': 'sparse_categorical_crossentropy', 'dim_output': 'mse'},
        loss_weights={'shape_output': 1.0, 'dim_output': 0.02},
        metrics={'shape_output': ['accuracy'], 'dim_output': ['mae']}
    )

    print(f"\nTraining model across {len(X_scaled):,} samples...")
    training_model.fit(
        X_scaled, [y_shape, y_dim],
        epochs=epochs,
        batch_size=batch_size,
        verbose=verbose
    )

    # 5. Evaluate
    combined_out = layers.Concatenate(axis=-1, name='predictions')([shape_out, dim_out])
    export_model = models.Model(inputs=training_model.input, outputs=combined_out)

    preds_all = export_model.predict(X_scaled, verbose=0)
    pred_shape_probs = preds_all[:, :num_classes]
    pred_shape_idx = np.argmax(pred_shape_probs, axis=1)
    pred_dims = preds_all[:, -1]

    acc = accuracy_score(y_shape, pred_shape_idx)
    mae = mean_absolute_error(y_dim, pred_dims)
    r2 = r2_score(y_dim, pred_dims)
    mismatches = int((y_shape != pred_shape_idx).sum())

    print(f"\n{'─'*50}")
    print(f"  Shape Accuracy  : {acc*100.0:.2f}% (Matches: {len(df_clean) - mismatches}/{len(df_clean)}, Mismatches: {mismatches})")
    print(f"  Size MAE        : {mae:.3f} mm")
    print(f"  Size R2         : {r2:.4f}")
    print(f"{'─'*50}")

    cm = confusion_matrix(y_shape, pred_shape_idx)
    print(f"\nConfusion Matrix ({list(shape_encoder.classes_)}):")
    print(cm)

    # 6. Extract weights for JS / Serverless
    d1 = training_model.get_layer('dense_1')
    bn1 = training_model.get_layer('bn_1')
    d2 = training_model.get_layer('dense_2')
    bn2_l = training_model.get_layer('bn_2')
    d3 = training_model.get_layer('dense_3')
    d4 = training_model.get_layer('dense_4')
    hs = training_model.get_layer('shape_output')
    hd = training_model.get_layer('dim_output')

    w1, b1 = d1.get_weights()
    g1, be1, m1, v1 = bn1.get_weights()
    e1 = float(bn1.epsilon)

    w2, b2 = d2.get_weights()
    g2, be2, m2, v2 = bn2_l.get_weights()
    e2 = float(bn2_l.epsilon)

    w3, b3 = d3.get_weights()
    w4, b4 = d4.get_weights()
    ws, bs = hs.get_weights()
    wd, bd = hd.get_weights()

    # Validate forward pass in NumPy
    def np_predict(X):
        h1 = np.maximum(0, np.dot(X, w1) + b1)
        h1b = g1 * (h1 - m1) / np.sqrt(v1 + e1) + be1
        h2 = np.maximum(0, np.dot(h1b, w2) + b2)
        h2b = g2 * (h2 - m2) / np.sqrt(v2 + e2) + be2
        h3 = np.maximum(0, np.dot(h2b, w3) + b3)
        h4 = np.maximum(0, np.dot(h3, w4) + b4)
        logits = np.dot(h4, ws) + bs
        e = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        probs = e / np.sum(e, axis=-1, keepdims=True)
        dim = np.dot(h4, wd) + bd
        return np.concatenate([probs, dim], axis=-1)

    n_check = min(200, len(X_scaled))
    pf = np_predict(X_scaled[:n_check])
    diff = float(np.max(np.abs(preds_all[:n_check] - pf)))
    assert diff < 1e-3, f"Forward-pass mismatch too large: {diff}"
    print(f"NumPy vs Keras forward pass difference: {diff:.2e} (Validated OK)")

    # 7. Save metadata
    sensor_means_list = scaler_sensors.mean_.tolist()
    sensor_stds_list = np.sqrt(scaler_sensors.var_).tolist()

    metadata = {
        'input_features': cols_s,
        'shape_classes': list(shape_encoder.classes_),
        'shape_encoder': shape_encoder,
        'shape_col': shape_col,
        'dim_col': dim_col,
        'scaler_sensors': scaler_sensors,
        'sensor_means': sensor_means_list,
        'sensor_stds': sensor_stds_list,
        'accuracy': float(acc),
        'dim_mae': float(mae),
        'dim_r2': float(r2),
        'num_inputs': 9,
        'num_classes': num_classes,
        'num_outputs': num_classes + 1,
        'sensor_physical_min': SENSOR_PHYSICAL_MIN,
        'sensor_physical_max': SENSOR_PHYSICAL_MAX,
        'sensor_physical_mean': SENSOR_PHYSICAL_MEAN,
        'dataset_sensor_min': overall_min,
        'dataset_sensor_max': overall_max,
        'dataset_sensor_mean': overall_mean,
    }
    joblib.dump(metadata, 'model_metadata.pkl')
    print("Saved model_metadata.pkl")

    # 8. Export TFLite
    converter = tf.lite.TFLiteConverter.from_keras_model(export_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_bytes = converter.convert()
    with open("anomaly_model.tflite", "wb") as f:
        f.write(tflite_bytes)
    print(f"Saved anomaly_model.tflite ({len(tflite_bytes):,} bytes)")

    # 9. Generate C header
    generate_c_header(cols_s, list(shape_encoder.classes_),
                      sensor_means_list, sensor_stds_list, "model_config.h")
    print("Saved model_config.h")

    # 10. Generate model_data.h
    try:
        convert_tflite_to_c_header("anomaly_model.tflite", "model_data.h")
        print("Saved model_data.h")
    except Exception as ex:
        print(f"model_data.h skipped: {ex}")

    # 11. Save weights for Vercel Serverless & Client-side JS
    vercel_payload = {
        "metadata": {
            "num_inputs": 9,
            "shape_classes": list(shape_encoder.classes_),
            "sensor_means": sensor_means_list,
            "sensor_stds": sensor_stds_list,
            "accuracy": float(acc),
            "dim_mae": float(mae),
            "dim_r2": float(r2),
            "sensor_physical_min": SENSOR_PHYSICAL_MIN,
            "sensor_physical_max": SENSOR_PHYSICAL_MAX,
            "sensor_physical_mean": SENSOR_PHYSICAL_MEAN,
        },
        "weights": {
            "w1": w1.tolist(), "b1": b1.tolist(),
            "bn_gamma": g1.tolist(), "bn_beta": be1.tolist(),
            "bn_mean": m1.tolist(), "bn_var": v1.tolist(), "bn_eps": e1,
            "w2": w2.tolist(), "b2": b2.tolist(),
            "bn2_gamma": g2.tolist(), "bn2_beta": be2.tolist(),
            "bn2_mean": m2.tolist(), "bn2_var": v2.tolist(), "bn2_eps": e2,
            "w3": w3.tolist(), "b3": b3.tolist(),
            "w4": w4.tolist(), "b4": b4.tolist(),
            "w_shape": ws.tolist(), "b_shape": bs.tolist(),
            "w_dim": wd.tolist(), "b_dim": bd.tolist()
        }
    }

    os.makedirs("api", exist_ok=True)
    os.makedirs("public", exist_ok=True)
    for tgt in ["model_weights.json", "api/model_weights.json", "public/model_weights.json"]:
        with open(tgt, "w") as f:
            json.dump(vercel_payload, f)
        print(f"Saved {tgt} ({os.path.getsize(tgt)/1024:.1f} KB)")

    print(f"\n✅ RETRAINING & EXPORT COMPLETE!")
    print(f"Accuracy: {acc*100.0:.2f}% | MAE: {mae:.2f}mm | Mismatches: {mismatches}")
    return metadata


if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Retrain TinyML model on sensor dataset.')
    p.add_argument('--dataset', default='sensor_design_dataset.csv')
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--lr', type=float, default=0.002)
    p.add_argument('--max_samples', type=int, default=None)
    args = p.parse_args()

    dataset_file = args.dataset
    if not os.path.exists(dataset_file):
        from dataset_generator import generate_sensor_design_dataset
        generate_sensor_design_dataset(dataset_file)

    train(dataset_file, epochs=args.epochs, batch_size=args.batch_size,
          lr=args.lr, max_samples=args.max_samples)
