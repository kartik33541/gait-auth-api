import os
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
from sklearn.preprocessing import StandardScaler
import pickle

FS = 30
WINDOW = 256
STEP = 128

# NEW: Safe Butterworth Bandpass Filter (0.3 Hz to 12.0 Hz)
# Isolates pure human biomechanics and strips gravity/sensor rattle
def butter_bandpass(signal, lowcut=0.3, highcut=12.0, fs=FS, order=3):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    
    # Mathematical safety catch in case of weird sampling rates
    if high >= 1.0:
        high = 0.99 
        
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, signal)

def preprocess_csv(path):
    df = pd.read_csv(path)
    df.columns = df.columns.str.lower().str.strip()
    cols = ["ax","ay","az","wx","wy","wz"]
    if not all(c in df.columns for c in cols): return None
    df = df[cols].apply(pd.to_numeric, errors="coerce").dropna()
    if len(df) < WINDOW: return None

    for c in cols:
        # 1. Apply Bandpass Filter (Removes gravity AND static noise)
        df[c] = butter_bandpass(df[c].values)
        # 2. Center the signal
        df[c] = df[c] - np.mean(df[c])

    # 3. Add Magnitudes (Orientation-independent features)
    df["acc_mag"] = np.sqrt(df["ax"]**2 + df["ay"]**2 + df["az"]**2)
    df["gyro_mag"] = np.sqrt(df["wx"]**2 + df["wy"]**2 + df["wz"]**2)
    
    return df[["ax","ay","az","wx","wy","wz","acc_mag","gyro_mag"]].values

def window_data(data):
    windows = []
    for i in range(0, len(data) - WINDOW, STEP):
        windows.append(data[i:i+WINDOW])
    return np.array(windows)

def load_dataset(base_path):
    person_data = {}
    for item in os.listdir(base_path):
        item_path = os.path.join(base_path, item)
        if os.path.isdir(item_path) and item.lower().startswith("person"):
            all_w = []
            for f in os.listdir(item_path):
                if f.endswith(".csv"):
                    data = preprocess_csv(os.path.join(item_path, f))
                    if data is not None: all_w.extend(window_data(data))
            if all_w: person_data[item] = np.array(all_w)

    syn_path = os.path.join(base_path, "synthetic_data")
    if os.path.exists(syn_path):
        for f in os.listdir(syn_path):
            if f.endswith(".csv"):
                identity = f.split("_")[0]
                data = preprocess_csv(os.path.join(syn_path, f))
                if data is not None:
                    if identity not in person_data: person_data[identity] = []
                    person_data[identity].extend(window_data(data))
    
    for k in person_data: person_data[k] = np.array(person_data[k])
    return person_data

def fit_scaler(person_data, save_path):
    all_samples = np.vstack([person_data[p] for p in person_data])
    scaler = StandardScaler()
    scaler.fit(all_samples.reshape(-1, 8))
    for p in person_data:
        w = person_data[p]
        person_data[p] = scaler.transform(w.reshape(-1,8)).reshape(w.shape).astype(np.float32)
    with open(save_path, "wb") as f: pickle.dump(scaler, f)
    return person_data
