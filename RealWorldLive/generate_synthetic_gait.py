import pandas as pd
import numpy as np
import os
import random

def augment_gait(df):
    """Applies biological variations to simulate a new person's walking pattern."""
    df_aug = df.copy()
    # Updated to match the columns produced by your MIT App
    sensors = ['ax', 'ay', 'az', 'wx', 'wy', 'wz']
    
    # Warping (simulates heavier/lighter step or different pace)
    warp_factor = np.random.normal(1.0, 0.15, size=len(sensors))
    
    # Jitter (simulates sensor noise and subtle tremors)
    jitter = np.random.normal(0.0, 0.05, size=(len(df_aug), len(sensors)))
    
    # Time shift (simulates phase shift of the walk cycle)
    shift = np.random.randint(-20, 20)
    
    for i, col in enumerate(sensors):
        if col in df_aug.columns:
            # Apply biological warping and jitter
            df_aug[col] = (df_aug[col] * warp_factor[i]) + jitter[:, i]
            # Shift the cycle
            df_aug[col] = np.roll(df_aug[col].values, shift)
             
    return df_aug

def main():
    print("🚀 Initializing Synthetic Gait Generator...")
    
    # 1. Dynamically find all CSV files inside 'person' subfolders
    # This prevents the script from accidentally reading 'received_gait.csv'
    base_files_map = {}
    for root, dirs, files in os.walk('.'):
        # Only look in folders that contain 'person' in the name
        if "person" in root.lower() and "synthetic" not in root.lower():
            for file in files:
                if file.lower().endswith('.csv'):
                    base_files_map[file] = os.path.join(root, file)
    
    if len(base_files_map) == 0:
        print("❌ Error: No base person files found! Run this inside the 'RealWorldLive' folder.")
        return
        
    print(f"✅ Found {len(base_files_map)} original source files.")
        
    # 2. Create output directory
    output_dir = "synthetic_data"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"🧬 Generating 100 synthetic identities (Person 11 to 110)...")

    # 3. Generate Person 11 through Person 110
    generated_count = 0
    for p in range(11, 111):
        # Pick a random biological base from your 10 real users
        base_p = random.randint(1, 10)
        
        # We generate 3 walks for each synthetic person to maintain consistency
        for w in range(1, 4):
            base_file_name = f"person{base_p}_walk{w}.csv"
            
            if base_file_name in base_files_map:
                try:
                    full_path = base_files_map[base_file_name]
                    df = pd.read_csv(full_path)
                    
                    # AI Augmentation
                    df_aug = augment_gait(df)
                    
                    # Save to synthetic_data folder
                    out_name = os.path.join(output_dir, f"person{p}_walk{w}.csv")
                    df_aug.to_csv(out_name, index=False)
                    generated_count += 1
                except Exception as e:
                    print(f"⚠️ Warning: Could not process {base_file_name}: {e}")

    print(f"\n✨ Success! {generated_count} files generated in '{output_dir}/'.")
    print("These users can now be used for large-scale Siamese training.")

if __name__ == "__main__":
    main()