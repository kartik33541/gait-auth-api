# import os
# import sys
# import pickle
# import numpy as np
# from tensorflow.keras.models import load_model

# # Ensure local imports work when called from Flask Server
# CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# if CURRENT_DIR not in sys.path:
#     sys.path.append(CURRENT_DIR)
# from dataset_loader import preprocess_csv, window_data

# # The model is one directory up inside RealWorldLive
# PARENT_DIR = os.path.dirname(CURRENT_DIR)
# MODEL_PATH = os.path.join(PARENT_DIR, "RealWorldLive", "model")

# encoder = load_model(os.path.join(MODEL_PATH, "encoder.keras"), compile=False)
# with open(os.path.join(MODEL_PATH, "scaler.pkl"), "rb") as f:
#     scaler = pickle.load(f)
# with open(os.path.join(MODEL_PATH, "templates.pkl"), "rb") as f:
#     templates = pickle.load(f)

# t_ids = list(templates.keys())
# t_matrix = np.array([templates[k] for k in t_ids])

# def predict_person(csv_path):
#     data = preprocess_csv(csv_path)
#     if data is None: return "INVALID_DATA"
    
#     windows = window_data(data)
#     if len(windows) == 0: return "TOO_SHORT"
    
#     windows_norm = scaler.transform(windows.reshape(-1, 8)).reshape(windows.shape)
#     emb = encoder.predict(windows_norm, verbose=0)
#     emb_mean = np.mean(emb, axis=0, keepdims=True)
    
#     similarity = np.matmul(emb_mean, t_matrix.T)
#     idx = np.argmax(similarity)
#     score = similarity[0][idx]

#     # Translate "Person9_style2" back to just "Person9"
#     best_template_name = t_ids[idx]
#     actual_person = best_template_name.split("_")[0]

#     # Access Threshold
#     if score >= 0.70:
#         return f"GRANTED: {actual_person} ({score:.2f})"
#     else:
#         return f"DENIED (Closest: {actual_person} at {score:.2f})"


import os
import sys
import pickle
import numpy as np

# ======================================================
# 1. TENSORFLOW MEMORY DIET (Must be before TF import!)
# Tells TensorFlow to use absolute minimum RAM and CPU
# ======================================================
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Force CPU only
os.environ['TF_NUM_INTRAOP_THREADS'] = '1' # Limit RAM threads
os.environ['TF_NUM_INTEROP_THREADS'] = '1' # Limit RAM threads

from tensorflow.keras.models import load_model

# Ensure local imports work when called from Flask Server
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.append(CURRENT_DIR)
from dataset_loader import preprocess_csv, window_data

# The model is one directory up inside RealWorldLive
PARENT_DIR = os.path.dirname(CURRENT_DIR)
MODEL_PATH = os.path.join(PARENT_DIR, "RealWorldLive", "model")

encoder = load_model(os.path.join(MODEL_PATH, "encoder.keras"), compile=False)
with open(os.path.join(MODEL_PATH, "scaler.pkl"), "rb") as f:
    scaler = pickle.load(f)
with open(os.path.join(MODEL_PATH, "templates.pkl"), "rb") as f:
    templates = pickle.load(f)

t_ids = list(templates.keys())
t_matrix = np.array([templates[k] for k in t_ids])

# ======================================================
# 2. THE WARM-UP ROUTINE
# Force the model to build its math engine during boot
# ======================================================
print("🧠 Warming up AI model to prevent memory crash...")
try:
    dummy_steps = encoder.input_shape[1] if encoder.input_shape[1] is not None else 128
    dummy_data = np.zeros((1, dummy_steps, 8))
    encoder.predict(dummy_data, verbose=0)
    print("✅ AI Warm-up complete! Ready for mobile requests.")
except Exception as e:
    print(f"⚠️ Warm-up skipped: {e}")
# ======================================================

def predict_person(csv_path):
    data = preprocess_csv(csv_path)
    if data is None: return "INVALID_DATA"
    
    windows = window_data(data)
    if len(windows) == 0: return "TOO_SHORT"
    
    windows_norm = scaler.transform(windows.reshape(-1, 8)).reshape(windows.shape)
    emb = encoder.predict(windows_norm, verbose=0)
    emb_mean = np.mean(emb, axis=0, keepdims=True)
    
    similarity = np.matmul(emb_mean, t_matrix.T)
    idx = np.argmax(similarity)
    score = similarity[0][idx]

    # Translate "Person9_style2" back to just "Person9"
    best_template_name = t_ids[idx]
    actual_person = best_template_name.split("_")[0]

    # Access Threshold
    if score >= 0.70:
        return f"GRANTED: {actual_person} ({score:.2f})"
    else:
        return f"DENIED (Closest: {actual_person} at {score:.2f})"