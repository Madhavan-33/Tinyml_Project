import pandas as pd
import numpy as np

def generate_sensor_design_dataset(filename="sensor_design_dataset.csv", num_samples=2500):
    """
    Generates a physically-grounded 9-sensor tactile mesh dataset for TinyML indenter
    shape classification ('Beam', 'Cylinder', 'Sphere') and dimension sizing ('dim_mm').
    
    The 9 sensors are arranged in a 3x3 planar tactile grid:
        Row 0: S1 (r=-1, c=-1), S2 (r=-1, c=0), S3 (r=-1, c=1)
        Row 1: S4 (r=0,  c=-1), S5 (r=0,  c=0), S6 (r=0,  c=1)
        Row 2: S7 (r=1,  c=-1), S8 (r=1,  c=0), S9 (r=1,  c=1)
        
    Sensor operating voltage range: 1.50V to 2.30V (typical mesh telemetry baseline: ~1.55V, max: ~2.28V).
    """
    np.random.seed(42)
    
    materials = ['Silicone', 'TPU', 'Rubber']
    shapes = ['Beam', 'Cylinder', 'Sphere']
    
    mat_factors = {'Silicone': 1.15, 'TPU': 1.0, 'Rubber': 0.88}
    
    selected_mat = np.random.choice(materials, size=num_samples)
    selected_shape = np.random.choice(shapes, size=num_samples)
    dim_mm = np.random.uniform(10.0, 60.0, size=num_samples)
    pressure = np.random.uniform(20.0, 200.0, size=num_samples)
    time_step = np.random.uniform(0.1, 10.0, size=num_samples)
    
    sensor_data = {}
    for i in range(1, 10):
        # 3x3 grid coordinates: r in {-1, 0, 1}, c in {-1, 0, 1}
        r = (i - 1) // 3 - 1
        c = (i - 1) % 3 - 1
        dist_center = np.sqrt(r**2 + c**2)
        
        # Indenter contact spread radius scales with object dimension
        spread = 0.6 + 1.6 * ((dim_mm - 10.0) / 50.0)
        
        # 1. Sphere: Hertzian point contact centered at S5 (r=0, c=0) with radial Gaussian roll-off
        sphere_pattern = np.exp(- (dist_center / spread)**2)
        
        # 2. Cylinder: Linear axial contact ridge along central column c=0 (S2, S5, S8)
        cyl_pattern = np.exp(- (abs(c) / spread)**2) * (0.85 + 0.15 * (1.0 - abs(r)))
        
        # 3. Beam: Flat rectangular punch contact across rows with perimeter/edge stress
        beam_pattern = 0.35 + 0.65 * np.exp(- (abs(r) / spread)**2)
        
        pattern = np.where(selected_shape == 'Sphere', sphere_pattern,
                  np.where(selected_shape == 'Cylinder', cyl_pattern, beam_pattern))
                  
        m_fac = np.array([mat_factors[m] for m in selected_mat])
        load_factor = (pressure / 200.0) * m_fac
        
        # Sensor response mapped into physical 1.50V - 2.30V operating range
        v = 1.55 + 0.68 * pattern * load_factor
        noise = np.random.normal(0, 0.005, size=num_samples)
        sensor_data[f'S{i}'] = np.clip(np.round(v + noise, 3), 1.50, 2.30)
        
    df = pd.DataFrame({
        'Material': selected_mat,
        'object_shape': selected_shape,
        'dim_mm': np.round(dim_mm, 2),
        'Time_step': np.round(time_step, 2),
        'Pressure': np.round(pressure, 2),
        **sensor_data
    })
    
    df.to_csv(filename, index=False)
    print(f"Generated '{filename}' with {num_samples} samples across 9 sensor outputs (S1-S9) in 1.5V-2.3V physical range!")
    return df

if __name__ == '__main__':
    generate_sensor_design_dataset()