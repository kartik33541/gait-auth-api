import os
import sys
import pickle
import numpy as np
from tensorflow.keras.models import load_model

# Dynamic local imports
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.append(CURRENT_DIR)
from dataset_loader import load_dataset 

# Look one folder UP for RealWorldLive
PARENT_DIR = os.path.dirname(CURRENT_DIR)
BASE_PATH = os.path.join(PARENT_DIR, "RealWorldLive")
MODEL_PATH = os.path.join(BASE_PATH, "model")

print("📂 Loading Model and Data...")
encoder = load_model(os.path.join(MODEL_PATH, "encoder.keras"), compile=False)
person_data = load_dataset(BASE_PATH)

with open(os.path.join(MODEL_PATH, "scaler.pkl"), "rb") as f:
    scaler = pickle.load(f)

templates = {}
print("🧬 Generating Multi-Style Signatures for REAL employees...")

for p in person_data:
    try:
        user_num = int(p.lower().replace("person", ""))
        # ONLY ENROLL USERS 1 THROUGH 10
        if user_num > 10: continue 
    except ValueError: continue 

    windows = person_data[p]
    windows_norm = scaler.transform(windows.reshape(-1, 8)).reshape(windows.shape)
    emb = encoder.predict(windows_norm, verbose=0)
    
    # RICH ENROLLMENT: Split the data into 3 style templates per person
    chunks = np.array_split(emb, 3)
    for i, chunk in enumerate(chunks):
        if len(chunk) > 0:
            template_name = f"{p}_style{i+1}"
            templates[template_name] = np.mean(chunk, axis=0)

with open(os.path.join(MODEL_PATH, "templates.pkl"), "wb") as f:
    pickle.dump(templates, f)

print(f"✅ Enrollment Complete. {len(templates)} unique style templates stored securely.")
