import pandas as pd
import numpy as np
import os
import random

def augment_gait(df):
    """Applies biological variations to simulate a new person's walking pattern."""
    df_aug = df.copy()
    sensors = ['ax', 'ay', 'az', 'wx', 'wy', 'wz']
    
    # Warping (simulates heavier/lighter step)
    warp_factor = np.random.normal(1.0, 0.15, size=len(sensors))
    
    # Jitter (simulates sensor noise and subtle tremors)
    jitter = np.random.normal(0.0, 0.1, size=(len(df_aug), len(sensors)))
    
    # Time shift (simulates phase shift of the walk cycle)
    shift = np.random.randint(-20, 20)
    
    for i, col in enumerate(sensors):
        if col in df_aug.columns:
            df_aug[col] = (df_aug[col] * warp_factor[i]) + jitter[:, i]
            df_aug[col] = np.roll(df_aug[col].values, shift)
             
    return df_aug

def main():
    print("Scanning for original dataset files across all subfolders...")
    
    # 1. Dynamically find all CSV files inside all subfolders
    base_files_map = {}
    for root, dirs, files in os.walk('.'):
        for file in files:
            if file.endswith('.csv') and 'person' in file:
                # Store the exact path to the file (e.g., "person1/person1_walk1.csv")
                base_files_map[file] = os.path.join(root, file)
    
    if len(base_files_map) == 0:
        print("Error: Still couldn't find any CSV files. Please ensure you are running this in the RealWorldLive folder.")
        return
        
    print(f"Found {len(base_files_map)} original files!")
        
    # 2. Create a clean structure for the new data
    output_dir = "synthetic_data"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Created directory: {output_dir}/")
    print("Generating 100 new synthetic persons (300 files)...")

    # 3. Generate Person 11 through Person 110
    generated_count = 0
    for p in range(11, 111):
        # Pick a random biological base (Person 1 through 10)
        base_p = random.randint(1, 10)
        
        for w in range(1, 4):
            base_file_name = f"person{base_p}_walk{w}.csv"
            
            if base_file_name in base_files_map:
                try:
                    # Read the original file using its true hidden path
                    full_path = base_files_map[base_file_name]
                    df = pd.read_csv(full_path)
                    
                    # Apply the AI augmentation
                    df_aug = augment_gait(df)
                    
                    # Save it into the new synthetic_data folder
                    out_name = os.path.join(output_dir, f"person{p}_walk{w}.csv")
                    df_aug.to_csv(out_name, index=False)
                    generated_count += 1
                except Exception as e:
                    print(f"Error processing {base_file_name}: {e}")

    print(f"\nSuccess! Generated {generated_count} synthetic files directly into the '{output_dir}' folder.")

if __name__ == "__main__":
    main()