# import os
# import pickle
# import numpy as np
# from tensorflow.keras.models import load_model
# from production.cnn_engine.dataset_loader import preprocess_csv, window_data

# BASE_DIR = os.path.dirname(os.path.abspath(__file__)) 
# MODEL_PATH = os.path.join(BASE_DIR, "RealWorldLive", "model")

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