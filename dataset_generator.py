import pandas as pd
import numpy as np

def generate_sensor_design_dataset(filename="sensor_design_dataset.csv", num_samples=1500):
    np.random.seed(42)
    
    materials = ['Silicone', 'TPU', 'Rubber']
    shapes = ['Cylinder', 'Sphere', 'Beam']
    
    mat_factors = {'Silicone': 1.5, 'TPU': 1.0, 'Rubber': 0.7}
    shape_factors = {'Cylinder': 1.2, 'Sphere': 0.9, 'Beam': 1.4}
    
    selected_mat = np.random.choice(materials, size=num_samples)
    selected_shape = np.random.choice(shapes, size=num_samples)
    dim_mm = np.random.uniform(10.0, 60.0, size=num_samples)
    time_step = np.random.uniform(0.1, 10.0, size=num_samples)
    pressure = np.random.uniform(5.0, 500.0, size=num_samples)
    
    # Generate S1 to S9 responses
    sensor_data = {}
    for i in range(1, 10):
        alpha_i = 0.5 + 0.15 * i
        beta_i = 0.2 + 0.08 * (10 - i)
        omega_i = 0.5 + 0.3 * (i % 3)
        
        e_factors = np.array([mat_factors[m] for m in selected_mat])
        g_factors = np.array([shape_factors[s] for s in selected_shape])
        
        # Physics-inspired continuous sensor response
        base_resp = alpha_i * (pressure / 100.0) * (e_factors / g_factors) * (50.0 / dim_mm)
        dynamic_resp = beta_i * np.sin(omega_i * time_step) * np.sqrt(pressure / 50.0)
        noise = np.random.normal(0, 0.05, size=num_samples)
        
        s_val = np.round(np.clip(base_resp + dynamic_resp + noise, 0.01, 50.0), 4)
        sensor_data[f'S{i}'] = s_val

    df = pd.DataFrame({
        'Material': selected_mat,
        'object_shape': selected_shape,
        'dim_mm': np.round(dim_mm, 2),
        'Time_step': np.round(time_step, 2),
        'Pressure': np.round(pressure, 2),
        **sensor_data
    })
    
    df.to_csv(filename, index=False)
    print(f"Generated '{filename}' with {num_samples} samples across 5 inputs & 9 sensor outputs (S1-S9)!")
    return df

if __name__ == '__main__':
    generate_sensor_design_dataset()