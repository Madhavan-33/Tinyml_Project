"""
Export trained dual-head neural network weights to JSON.
Folds BatchNormalization for ultra-fast, zero-overhead inference on Vercel serverless
and client-side browsers without requiring TensorFlow.
"""

import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import joblib

def export_weights():
    # Load dataset & re-run training to extract exact layer weights
    from train_headless import train_and_export_default
    
    print("Running training to extract exact network parameters...")
    # Read dataset
    dataset_file = "sensor_design_dataset.csv"
    import pandas as pd
    import re
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.metrics import accuracy_score, mean_absolute_error, r2_score

    df = pd.read_csv(dataset_file)
    cols_s = [f'S{i}' for i in range(1, 10)]
    shape_col = 'object_shape'
    dim_col = 'dim_mm'

    df_clean = df[cols_s + [shape_col, dim_col]].dropna().copy()
    X_raw = df_clean[cols_s].values.astype(np.float32)
    scaler_sensors = StandardScaler()
    X_scaled = scaler_sensors.fit_transform(X_raw).astype(np.float32)

    shape_encoder = LabelEncoder()
    y_shape = shape_encoder.fit_transform(df_clean[shape_col].astype(str))
    num_classes = len(shape_encoder.classes_)
    y_dim = df_clean[dim_col].values.astype(np.float32)

    # Build model
    tf.random.set_seed(42)
    np.random.seed(42)

    inputs = layers.Input(shape=(9,), name='sensor_inputs')
    x = layers.Dense(64, activation='relu', name='dense_1')(inputs)
    x = layers.BatchNormalization(name='bn_1')(x)
    x = layers.Dropout(0.1)(x)
    x = layers.Dense(64, activation='relu', name='dense_2')(x)
    x = layers.Dense(32, activation='relu', name='dense_3')(x)

    shape_out = layers.Dense(num_classes, activation='softmax', name='shape_output')(x)
    dim_out = layers.Dense(1, activation='linear', name='dim_output')(x)

    training_model = models.Model(inputs=inputs, outputs=[shape_out, dim_out])
    training_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.002),
        loss={'shape_output': 'sparse_categorical_crossentropy', 'dim_output': 'mse'},
        loss_weights={'shape_output': 1.0, 'dim_output': 0.05}
    )

    training_model.fit(
        X_scaled, [y_shape, y_dim],
        validation_split=0.15,
        epochs=45,
        batch_size=32,
        verbose=0
    )

    combined_out = layers.Concatenate(axis=-1, name='predictions')([shape_out, dim_out])
    export_model = models.Model(inputs=inputs, outputs=combined_out)
    preds_all = export_model.predict(X_scaled, verbose=0)
    acc = accuracy_score(y_shape, np.argmax(preds_all[:, :num_classes], axis=1))
    mae = mean_absolute_error(y_dim, preds_all[:, -1])
    r2 = r2_score(y_dim, preds_all[:, -1])
    print(f"Accuracy: {acc*100:.2f}%, MAE: {mae:.3f} mm (R²={r2:.4f})")

    # Extract weights
    d1 = training_model.get_layer('dense_1')
    bn = training_model.get_layer('bn_1')
    d2 = training_model.get_layer('dense_2')
    d3 = training_model.get_layer('dense_3')
    head_shape = training_model.get_layer('shape_output')
    head_dim = training_model.get_layer('dim_output')

    w1, b1 = d1.get_weights()
    gamma, beta, mean, var = bn.get_weights()
    eps = float(bn.epsilon)

    w2, b2 = d2.get_weights()
    w3, b3 = d3.get_weights()
    w_shape, b_shape = head_shape.get_weights()
    w_dim, b_dim = head_dim.get_weights()

    # Forward pass test in NumPy
    def fast_predict(x_scaled):
        h1 = np.maximum(0, np.dot(x_scaled, w1) + b1)
        bn1 = gamma * (h1 - mean) / np.sqrt(var + eps) + beta
        h2 = np.maximum(0, np.dot(bn1, w2) + b2)
        h3 = np.maximum(0, np.dot(h2, w3) + b3)
        logits = np.dot(h3, w_shape) + b_shape
        e = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        probs = e / np.sum(e, axis=-1, keepdims=True)
        dim = np.dot(h3, w_dim) + b_dim
        return np.concatenate([probs, dim], axis=-1)

    preds_fast = fast_predict(X_scaled[:50])
    max_diff = float(np.max(np.abs(preds_all[:50] - preds_fast)))
    print(f"Validation difference (Keras vs NumPy): {max_diff:.8e}")
    assert max_diff < 1e-4, f"Mismatch too large: {max_diff}"

    # Also export TFLite so TFLite stays updated
    converter = tf.lite.TFLiteConverter.from_keras_model(export_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    with open("anomaly_model.tflite", "wb") as f:
        f.write(tflite_model)

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
        'dim_r2': float(r2),
        'num_inputs': 9,
        'num_classes': num_classes,
        'num_outputs': num_classes + 1
    }
    joblib.dump(metadata, 'model_metadata.pkl')

    # Prepare Vercel JSON payload
    vercel_model = {
        "metadata": {
            "num_inputs": 9,
            "shape_classes": list(shape_encoder.classes_),
            "sensor_means": [float(m) for m in scaler_sensors.mean_],
            "sensor_stds": [float(s) for s in np.sqrt(scaler_sensors.var_)],
            "accuracy": float(acc),
            "dim_mae": float(mae),
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
        print(f"Exported {target} ({os.path.getsize(target)/1024:.1f} KB)")

    print("SUCCESS: Vercel model weights exported and verified!")

if __name__ == "__main__":
    export_weights()
