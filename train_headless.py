import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error, r2_score
import joblib
import json
import os
import re
import sys
from convert_model import convert_tflite_to_c_header

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

def train_and_export_default(dataset_file=None):
    if dataset_file is None:
        dataset_file = "all_runs_n10-n19_SS960_ipc.csv" if os.path.exists("all_runs_n10-n19_SS960_ipc.csv") else "sensor_design_dataset.csv"
    if not os.path.exists(dataset_file):
        from dataset_generator import generate_sensor_design_dataset
        generate_sensor_design_dataset(dataset_file)

    if dataset_file.endswith('.csv'):
        df = pd.read_csv(dataset_file)
    else:
        df = pd.read_excel(dataset_file)

    # 1. Select strictly the 9 sensor outputs captured from the sensor mesh (S1 ... S9)
    cols_s = [c for c in df.columns if re.match(r'^s[1-9]$', c, re.IGNORECASE)]
    if len(cols_s) < 9:
        cols_s = [f'S{i}' for i in range(1, 10) if f'S{i}' in df.columns]
    if len(cols_s) < 9:
        raise ValueError(f"Could not find 9 sensor columns (S1-S9) in {dataset_file}. Found: {cols_s}")

    # Targets: Object Shape (classification) & Dimension (regression)
    shape_col = 'object_shape' if 'object_shape' in df.columns else next((c for c in df.columns if 'shape' in c.lower()), None)
    dim_col = 'dim_mm' if 'dim_mm' in df.columns else next((c for c in df.columns if 'dim' in c.lower() or 'size' in c.lower()), None)

    if shape_col is None or dim_col is None:
        raise ValueError(f"Could not detect target shape and dim columns in {dataset_file}")

    # Preprocessing
    df_clean = df[cols_s + [shape_col, dim_col]].dropna().copy()

    # Features: strictly the 9 sensor readings
    X_raw = df_clean[cols_s].values.astype(np.float32)
    scaler_sensors = StandardScaler()
    X_scaled = scaler_sensors.fit_transform(X_raw).astype(np.float32)

    # Targets
    shape_encoder = LabelEncoder()
    y_shape = shape_encoder.fit_transform(df_clean[shape_col].astype(str))
    num_classes = len(shape_encoder.classes_)
    y_dim = df_clean[dim_col].values.astype(np.float32)

    print(f"Training on {len(df_clean)} samples with 9 sensor inputs: {cols_s}")
    print(f"Target Object Classes ({num_classes}): {list(shape_encoder.classes_)}")
    print(f"Target Dimension: min={y_dim.min():.2f} mm, max={y_dim.max():.2f} mm, mean={y_dim.mean():.2f} mm")

    tf.random.set_seed(42)
    np.random.seed(42)

    # 2. Build Dual-Head Neural Network
    inputs = layers.Input(shape=(9,), name='sensor_inputs')
    x = layers.Dense(64, activation='relu', name='dense_1')(inputs)
    x = layers.BatchNormalization(name='bn_1')(x)
    x = layers.Dropout(0.05)(x)
    x = layers.Dense(64, activation='relu', name='dense_2')(x)
    x = layers.Dense(32, activation='relu', name='dense_3')(x)

    shape_out = layers.Dense(num_classes, activation='softmax', name='shape_output')(x)
    dim_out = layers.Dense(1, activation='linear', name='dim_output')(x)

    training_model = models.Model(inputs=inputs, outputs=[shape_out, dim_out])
    training_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.003),
        loss={
            'shape_output': 'sparse_categorical_crossentropy',
            'dim_output': 'mse'
        },
        loss_weights={
            'shape_output': 1.0,
            'dim_output': 0.01
        },
        metrics={
            'shape_output': ['accuracy'],
            'dim_output': ['mae']
        }
    )

    training_model.fit(
        X_scaled, [y_shape, y_dim],
        validation_split=0.15,
        epochs=60,
        batch_size=32,
        verbose=0
    )

    # 3. Create Combined Export Model: [p_0, ..., p_K-1, dim_mm]
    combined_out = layers.Concatenate(axis=-1, name='predictions')([shape_out, dim_out])
    export_model = models.Model(inputs=inputs, outputs=combined_out)

    # 4. Evaluate on All Data
    preds_all = export_model.predict(X_scaled, verbose=0)
    pred_shape_probs = preds_all[:, :num_classes]
    pred_shape_idx = np.argmax(pred_shape_probs, axis=1)
    pred_dim = preds_all[:, -1]

    acc = accuracy_score(y_shape, pred_shape_idx)
    mae = mean_absolute_error(y_dim, pred_dim)
    rmse = np.sqrt(mean_squared_error(y_dim, pred_dim))
    r2 = r2_score(y_dim, pred_dim)

    print(f"\n✅ Training Complete!")
    print(f"Object Shape Accuracy: {acc*100.0:.2f}%")
    print(f"Dimension Size MAE: {mae:.3f} mm (RMSE: {rmse:.3f} mm, R² = {r2:.4f})")

    # 5. Save Metadata
    metadata = {
        'input_features': cols_s,
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

    # 6. Export TFLite Model
    converter = tf.lite.TFLiteConverter.from_keras_model(export_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    with open("anomaly_model.tflite", "wb") as f:
        f.write(tflite_model)

    # 7. Generate C Header for STM32 / Renode / Arduino firmware
    num_inputs = 9
    num_outputs = num_classes + 1
    in_means_str = ", ".join([f"{m:.6f}f" for m in scaler_sensors.mean_])
    in_stds_str = ", ".join([f"{s:.6f}f" for s in np.sqrt(scaler_sensors.var_)])
    class_names_str = ", ".join([f'"{c}"' for c in shape_encoder.classes_])

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
 * Sensor Input Normalization Parameters (S1 ... S9)
 */
static const float INPUT_MEANS[{num_inputs}] = {{ {in_means_str} }};
static const float INPUT_STDS[{num_inputs}] = {{ {in_stds_str} }};

/*
 * Shape Class Labels ({num_classes}):
 */
static const char* const SHAPE_CLASSES[{num_classes}] = {{ {class_names_str} }};

/*
 * Model Output Format ({num_outputs} floats):
 * - Indices [0 .. {num_classes - 1}]: Shape probabilities (Softmax: {", ".join(shape_encoder.classes_)})
 * - Index [{num_classes}]: Predicted Dimension (dim_mm in mm)
 */

#endif /* MODEL_CONFIG_H */
"""
    with open("model_config.h", "w") as f:
        f.write(header_content)

    # Convert to model_data.h
    try:
        convert_tflite_to_c_header("anomaly_model.tflite", "model_data.h")
    except Exception as e:
        print(f"Note: TFLite header conversion: {e}")

    # 8. Export Weights for Vercel Serverless & Browser JS
    try:
        d1 = training_model.get_layer('dense_1')
        bn = training_model.get_layer('bn_1')
        d2 = training_model.get_layer('dense_2')
        d3 = training_model.get_layer('dense_3')
        hs = training_model.get_layer('shape_output')
        hd = training_model.get_layer('dim_output')

        w1, b1 = d1.get_weights()
        gamma, beta, mean, var = bn.get_weights()
        eps = float(bn.epsilon)

        w2, b2 = d2.get_weights()
        w3, b3 = d3.get_weights()
        w_shape, b_shape = hs.get_weights()
        w_dim, b_dim = hd.get_weights()

        vercel_model = {
            "metadata": {
                "num_inputs": 9,
                "shape_classes": list(shape_encoder.classes_),
                "sensor_means": [float(m) for m in scaler_sensors.mean_],
                "sensor_stds": [float(s) for s in np.sqrt(scaler_sensors.var_)],
                "accuracy": float(acc),
                "dim_mae": float(mae),
                "dim_rmse": float(rmse),
                "dim_r2": float(r2)
            },
            "weights": {
                "w1": w1.tolist(),
                "b1": b1.tolist(),
                "bn_gamma": gamma.tolist(),
                "bn_beta": beta.tolist(),
                "bn_mean": mean.tolist(),
                "bn_var": var.tolist(),
                "bn_eps": float(eps),
                "w2": w2.tolist(),
                "b2": b2.tolist(),
                "w3": w3.tolist(),
                "b3": b3.tolist(),
                "w_shape": w_shape.tolist(),
                "b_shape": b_shape.tolist(),
                "w_dim": w_dim.tolist(),
                "b_dim": b_dim.tolist()
            }
        }
        os.makedirs("api", exist_ok=True)
        os.makedirs("public", exist_ok=True)
        for target in ["model_weights.json", "api/model_weights.json", "public/model_weights.json"]:
            with open(target, "w") as f:
                json.dump(vercel_model, f)
    except Exception as e:
        print(f"Note: JSON weights export: {e}")

    print(f"Successfully exported 9-sensor TinyML model to anomaly_model.tflite ({len(tflite_model)} bytes), model_config.h, model_data.h, and model_weights.json!")
    return metadata

if __name__ == '__main__':
    train_and_export_default()
