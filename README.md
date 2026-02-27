# 📦 Production – Real-Time Gait Authentication System

---

## 🧱 1️⃣ System Overview

This production system implements a real-time gait-based biometric authentication pipeline.

It performs:

Mobile Data Collection → HTTP Transmission → Server Processing → ML Prediction → Authentication Decision

Key characteristics:

- Real-time biometric authentication
- Smartphone-based motion sensing
- Random Forest classifier trained on real-world walking data
- Session-based authentication (not single-window)
- Majority voting across overlapping windows
- Threshold-based access control

This is a deployment-ready system — not experimental or research-only code.

---

## 📱 2️⃣ Mobile Application (Data Collection Layer For Real world)

**Platform:** MIT App Inventor  

**Sensors Used:**
- Linear Accelerometer (ax, ay, az)
- Gyroscope (wx, wy, wz)

**Sampling Method:**
- Clock-based polling
- Timer interval = 50 ms
- Effective frequency ≈ 20 Hz

**Session Duration:**
- ~15 seconds per walk

**CSV Format:**

timestamp,ax,ay,az,wx,wy,wz


Example row:

24,3.86,3.43,7.74,2.49,3.31,-12.49


Important Notes:

- Timestamp is recorded but NOT used in feature extraction.
- Each new session overwrites the previous file.
- The mobile app acts as a sensor front-end only (no ML on device).
- I didn't use Physics Toolbox Sensor Toolbox as i wanted that csv contain both linear_accelerometer data and gyroscope data in one file and that should sends to the   system in real time when person walks near to the system which is real world implication of the system.

---

## 🌐 3️⃣ Network Communication Layer

**Server Type:** Python `HTTPServer`  
**Port:** 8000  
**Requirement:** Phone and laptop must be on same WiFi network.

### Data Flow

Mobile App
↓ HTTP POST
Python Server (server.py)
↓ Save CSV
received_gait.csv
↓ Inference
infer_identity.py
↓ Return Decision
Mobile App


The server:

- Accepts POST requests
- Saves CSV as `received_gait.csv`
- Runs prediction
- Returns authentication result
- New Csv file overWrite the previous `recieved_gait.csv`

---

## 🧠 4️⃣ Model Training (Production Version)

**Training Script:**

train_and_save_model.py


**Dataset Location:**

production/data/RealWorldLive/


**Structure:**

Person1/
walk1.csv
walk2.csv
walk3.csv
Person2/
...


### Configuration

- Window size = 128 samples
- Step size = 32 samples
- Overlap = 75%                    ## THis is the assumption from research that increasing overlap will result in more accuracy 
- Sampling rate ≈ 20 Hz
- Window duration ≈ 6.4 seconds
- Features per window = 56

**Model:**

- RandomForestClassifier
- 800 trees
- StandardScaler normalization

**Saved Artifacts:**

model/rf_model.pkl
model/scaler.pkl


Production training uses **all available data** (no train/test split).

---

## ⚙️ 5️⃣ Inference Pipeline (Core Logic)

When a CSV is received:

1. Load CSV
2. Validate column format
3. Convert to numeric
4. Drop NaNs
5. Static detection  
   `motion_energy < 0.15 → deny`
6. Sliding window generation
7. Feature extraction (56 features)
8. Scaling
9. Window-level predictions
10. Majority voting
11. Threshold check (0.45)
12. Return decision

---

## 🔐 6️⃣ Authentication Decision Logic

Let:

vote_ratio = dominant_class_votes / total_windows


If:

vote_ratio ≥ 0.45


Return:

ACCESS_GRANTED (PersonX)


Else:

ACCESS_DENIED (closest: PersonX)


Other denial cases:

- ACCESS_DENIED (static)
- ACCESS_DENIED (insufficient data)
- ACCESS_DENIED (bad csv format)
- ACCESS_DENIED (file not found)

---

## 📊 7️⃣ Observed Production Performance

Real testing results:

- 7 live sessions tested
- 6 correct identifications
- 1 misclassification

Typical vote ratio: 0.73 – 1.0  
Windows per session: ~14–15  

Static detection correctly rejects non-walking sessions.

---

# 🏗 8️⃣ Production Folder Structure

```
production/
│
├── app/
│ ├── server.py               # HTTP server + inference trigger
│ ├── infer_identity.py       # Prediction logic
│
├── data/
│ └── RealWorldLive/       # Training dataset (Person folders) this data I have collected using my own app with 20Hz frequency which can be modified to 50Hz as per need
│
├── mobile_app/
│ ├── gait_app.aia # MIT App Inventor project
│ ├── gait_app.apk # Installable Android app
│ └── img          # images of app 
│
├── model/
│ ├── rf_model.pkl # Trained Random Forest Model
│ ├── scaler.pkl # StandardScaler
│
├── train_and_save_model.py # Model training script
├── received_gait.csv # Runtime session file
└── README.md # This file
```

---

## 🧪 9️⃣ How To Run The System

### 1️⃣ Install dependencies

pip install -r requirements.txt


### 2️⃣ Train the model

python train_and_save_model.py


### 3️⃣ Start the server

python app/server.py


### 4️⃣ Connect phone

- Connect phone and laptop to same WiFi
- Enter laptop IPv4 address in mobile app
- Format: `http://10.30.15.64:8000`

### 5️⃣ Record walk

- Walk naturally for ~15 seconds
- Observe server output

---

## ⚠️ 10️⃣ Known Limitations

- Requires walking motion (static rejected)
- Requires same WiFi network between mobile and Laptop(or system)
- Currently trained on small user base (5 users)
- Sensitive to phone placement
- Sampling rate 20 Hz (lower than UCI HAR 50 Hz) as I believe taking more readings in less time will increase the data which can increase the stability
- Not hardened against spoofing attacks

---
 
## ✅ Deployment Status

This system is suitable for:

- Academic demonstration
- Controlled-environment authentication
- Prototype biometric systems
- Research extension

It is not yet hardened for commercial security deployment.

---

---

## 🧩 11️⃣ Assumptions Made in Production Mode

During deployment, several practical engineering assumptions were made to balance stability, simplicity, and real-world constraints.

### 1️⃣ Sampling Rate ≈ 20 Hz (Instead of 50 Hz)

- The MIT App uses a 50 ms clock interval.
- This results in ~20 samples per second.
- UCI HAR dataset uses 50 Hz.

**Reasoning:**  
MIT App Inventor’s clock-based polling is less precise than native Android APIs.  
20 Hz is sufficient to capture human gait cycles (~1–1.5 sec per step) while keeping implementation simple and stable.

---

### 2️⃣ Window Size = 128 Samples

At 20 Hz:

128 / 20 = 6.4 seconds per window


**Reasoning:**  
Longer windows increase stability and reduce noise for small datasets.  
With limited users (5 persons), longer windows improve separability between individuals.

---

### 3️⃣ Step Size = 32 Samples (≈75% Overlap)

Overlap calculation:

Overlap = (128 - 32) / 128 = 0.75 = 75%


**Reasoning:**  
Higher overlap increases number of decision windows per session.  
This improves majority voting robustness without requiring longer recording time.

---

### 4️⃣ Session Duration ≈ 15 Seconds

**Reasoning:**  
15 seconds ensures:

- Multiple gait cycles captured
- ~14–15 overlapping windows
- Stable majority voting
- Reliable authentication decision

Shorter sessions reduce confidence.

---

### 5️⃣ Majority Voting Threshold = 0.45

vote_ratio ≥ 0.45 → ACCESS_GRANTED


**Reasoning:**  
A lower threshold prevents rejecting genuine users due to minor variability.  
With 14–15 windows, this still requires consistent dominance of one identity.

---

### 6️⃣ Static Detection Threshold (motion_energy < 0.15)

**Reasoning:**  
Prevents authentication when:

- Phone is stationary
- Device is placed on a table
- No walking motion exists

This avoids forced classification of non-gait data.

---

### 7️⃣ Training on Full Dataset (No Train/Test Split in Production)

**Reasoning:**  
Production model prioritizes maximum available learning per user.  
The system is designed for closed-set authentication, not benchmark evaluation.

---

### 8️⃣ Closed-Set Identification Assumption

The system assumes:

- All valid users are enrolled.
- Any unknown user may still be mapped to the closest known identity.

Mitigation used:

- Threshold-based majority voting
- Static filtering

True open-set biometric rejection is not implemented in this version.

---

### 9️⃣ Device Placement Consistency Assumption

It is assumed that:

- Users carry the phone in similar orientation and location (e.g., pocket).

Large variation in placement may reduce accuracy.

---

These assumptions were made to create a stable, reproducible, and deployment-ready prototype while working within the constraints of:

- MIT App Inventor
- Small real-world dataset
- Non-native Android sensor access
- Controlled WiFi environment

---