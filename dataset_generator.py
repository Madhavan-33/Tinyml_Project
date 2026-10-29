import pandas as pd
import numpy as np

def generate_sensor_design_dataset(filename="sensor_design_dataset.csv", num_samples=3000):
    """
    Generates a physically-grounded 9-sensor tactile mesh dataset for TinyML indenter
    shape classification ('cube', 'cylinder') and dimension sizing ('dim_mm').
    
    The 9 sensors are arranged in a 3x3 planar tactile grid:
        Row 0: S1 (r=-1, c=-1), S2 (r=-1, c=0), S3 (r=-1, c=1)
        Row 1: S4 (r=0,  c=-1), S5 (r=0,  c=0), S6 (r=0,  c=1)
        Row 2: S7 (r=1,  c=-1), S8 (r=1,  c=0), S9 (r=1,  c=1)
        
    Physical sensor characteristics (piezoresistive mesh):
        - Unloaded baseline: ~2.27V
        - Under pressure/contact: voltage drops towards ~1.65V - 1.95V
        - Cube: 2D planar flat press across all 9 sensors
        - Cylinder: Linear axial contact ridge along center column (S2, S5, S8)
    """
    np.random.seed(42)
    
    materials = ['SS960', 'Silicone', 'TPU']
    shapes = ['cube', 'cylinder']
    mat_factors = {'SS960': 1.0, 'Silicone': 1.10, 'TPU': 0.95}
    
    # Balanced shapes
    n_per_shape = num_samples // 2
    selected_shape = np.array(['cube'] * n_per_shape + ['cylinder'] * n_per_shape)
    np.random.shuffle(selected_shape)
    
    selected_mat = np.random.choice(materials, size=num_samples)
    dim_mm = np.random.uniform(15.0, 60.0, size=num_samples)
    pressure = np.random.uniform(40.0, 180.0, size=num_samples)
    time_step = np.random.uniform(0.5, 10.0, size=num_samples)
    
    sensor_data = {}
    for i in range(1, 10):
        # 3x3 grid coordinates
        r = (i - 1) // 3 - 1
        c = (i - 1) % 3 - 1
        dist_center = np.sqrt(r**2 + c**2)
        spread = 0.8 + 1.2 * ((dim_mm - 15.0) / 45.0)
        
        # 1. Cube: Flat uniform planar contact across all sensors
        cube_contact = 0.52 + 0.48 * np.exp(- (dist_center / (spread * 1.5))**2)
        
        # 2. Cylinder: Axial contact along central column (c=0: S2, S5, S8) with rapid dropoff at sides (c=-1, 1)
        cyl_contact = np.exp(- (abs(c) / (spread * 0.65))**2) * (0.85 + 0.15 * (1.0 - abs(r)*0.5))
        
        pattern = np.where(selected_shape == 'cylinder', cyl_contact, cube_contact)
        m_fac = np.array([mat_factors[m] for m in selected_mat])
        load = (pressure / 180.0) * m_fac
        
        # Physical voltage response: baseline 2.27V dropping under load
        delta_v = 0.45 * pattern * load + 0.08 * (dim_mm / 60.0)
        v = 2.27 - delta_v
        noise = np.random.normal(0, 0.002, size=num_samples)
        sensor_data[f'S{i}'] = np.clip(np.round(v + noise, 3), 1.60, 2.28)
        
    df = pd.DataFrame({
        'Material': selected_mat,
        'object_shape': selected_shape,
        'dim_mm': np.round(dim_mm, 2),
        'Time_step': np.round(time_step, 2),
        'Pressure': np.round(pressure, 2),
        **sensor_data
    })
    
    df.to_csv(filename, index=False)
    print(f"Generated '{filename}' with {num_samples} samples across 9 sensors (S1-S9) in 1.60V-2.28V physical range!")
    return df

if __name__ == '__main__':
    generate_sensor_design_dataset()