# 📦 Gait-Secure: Production-Grade Biometric Authentication
### AI-Powered Contactless Employee Security System via Siamese 1D-CNN

This repository contains the deployment-ready implementation of a contactless biometric security system. By analyzing smartphone-based motion signatures (Accelerometer + Gyroscope), the system identifies enrolled personnel with high precision using Deep Metric Learning.

---

## 🏗️ 1. System Architecture & Methodology

The production system has transitioned from a traditional research-based Random Forest classifier to a modern **Deep Metric Learning** pipeline designed for real-world enterprise scalability.

### 🔬 Core Methodology (The "How" and "Why")

#### A. Transition to Siamese 1D-CNN (Key Innovation)
* **The Problem:** Traditional classification models (like Random Forest) require retraining every time a new employee is added to the database.
* **The Solution:** The production system uses a **Siamese 1D-Convolutional Neural Network (1D-CNN)**. This architecture learns to map complex gait signals into a 128-dimensional embedding space.
* **The Benefit:** Authentication is performed by comparing the "Live Probe" embedding against a "Stored Template" using **Cosine Similarity**. This allows the system to scale infinitely; new users are simply "enrolled" without ever touching the underlying AI model.

#### B. Signal Processing & Preprocessing


[Image of Butterworth bandpass filter frequency response]

* **Butterworth Bandpass Filter:** All raw data is processed through a digital filter (0.5Hz – 3Hz). This isolates the rhythmic gait cycle while stripping away:
    1. **DC Bias:** Eliminates the constant 1G force of Earth's gravity.
    2. **High-Frequency Jitter:** Removes sensor noise and tremors.
* **Normalization Strategy:** Based on our research findings, we use **Selective Normalization**. Over-normalizing can strip away the micro-variations that act as unique biometric signatures. We maintain the intensity scales that define individual walking intensity.

#### C. Physics-Based Security: 3D Magnitude Energy Score
To defeat "Static Spoofing" (e.g., a user manually shaking or lifting/dropping a phone), we implemented a robust **Walk Energy Score**.
* **Formula:** $\text{Energy} = \text{std}(\sqrt{a_x^2 + a_y^2 + a_z^2})$
* **Threshold:** A score $\ge 1.0$ is strictly required. Any signal below this is rejected as a "Static/Fake Walk," preventing the AI from processing non-gait data.

---

## 📈 2. Dataset Scalability & AI Augmentation

### 🧬 Synthetic Data Expansion (Scaling to 110 Users)
Deep Learning models require massive datasets to prevent **Overfitting** (memorizing individuals rather than learning gait mechanics).
* **Real-World Foundation:** The real-world test group was expanded from **5 to 10 users** via custom collection sessions.
* **Synthetic Scaling:** I developed a Python-based **AI Data Generator** (`generate_synthetic_gait.py`) to derive **100 additional unique identities** (Person 11–110).
* **Purpose:** This provides the Siamese network with a massive "latent space," ensuring it learns to distinguish between subtle gait variations across a large population.

### 🧪 On-the-Fly Data Augmentation
During the training loop, the system applies biological variance injection:
* **Time Warping:** Simulates walking faster or slower.
* **Jittering:** Simulates sensor noise or tremors.
* **Phase Shifting:** Simulates starting the walk at different points in the gait cycle.

---

## 🌐 3. Robust Flask Server (Cloud-Ready Backend)

The backend is engineered for high-speed response and stability in memory-constrained environments:
* **AI Warm-up Routine:** Upon booting, the server runs a dummy prediction. This "warms up" the TensorFlow math engine, eliminating the ~10s lag usually seen during the first user request.
* **Memory Optimization:** Implemented a "Thread-Safe" inference engine that limits memory consumption to fit within standard cloud tiers (512MB–1GB RAM).
* **API Security:** All communication from the mobile app is validated via a custom **X-API-KEY** header.

---

## 📂 4. Project Folder Structure

```text
GAIT_AUTHENTICATION/
├── production/
│   ├── app/
│   │   └── flask_server.py        # Optimized API (Energy Check, Warm-up, Inference)
│   ├── cnn_engine/
│   │   ├── dataset_loader.py      # Signal Filtering & Augmentation logic
│   │   ├── build_encoder.py       # Siamese 1D-CNN Architecture
│   │   ├── train_siamese.py       # Triplet Loss training engine
│   │   ├── enroll_templates.py    # Generates .pkl gait signatures
│   │   └── infer_realtime.py      # Memory-diet prediction scripts
│   ├── RealWorldLive/
│   │   ├── Person1/ to Person10/  # Real-world base CSV datasets (~20Hz)
│   │   ├── synthetic_data/        # (Ignored by Git) Generated 100 identities
│   │   ├── model/                 # Saved .keras and .pkl artifacts
│   │   └── generate_synthetic.py  # AI-driven scaling script
│   ├── mobile_app/
│   │   ├── GaitAuth_Live.apk      # MIT App Inventor Android Client
│   │   └── GaitAuth_Flask.aia     # Source project file
│   ├── .gitignore                 # Configured to exclude massive synthetic data
│   ├── requirements.txt           # Production-specific dependencies
│   └── Documentation.md           # Full technical project report

```

# 🛠️ Setup & Execution Guide

## 1️⃣ Prepare Environment
```bash
pip install -r requirements.txt
```

## 2️⃣ Scale the Dataset (AI Generator)
Navigate to the `RealWorldLive` folder and run the scaling script to create the 110-user training pool:
```bash
python RealWorldLive/generate_synthetic_gait.py
```

## 3️⃣ Train & Enroll Identities
Train the 1D-CNN encoder and then generate the mathematical templates for authorized users:
```bash
python cnn_engine/train_siamese.py
defaults.py
```
Enroll templates:
```python
cnn_engine/enroll_templates.py
```

## 4️⃣ Launch the Robust Server
```bash
python app/flask_server.py
```

## 5️⃣ Mobile Connectivity
- Connect Phone and Laptop to the same Mobile Hotspot (to bypass campus firewall restrictions).
- Enter the Laptop IPv4 address in the app (e.g., [http://192.168.1.5:8000](http://192.168.1.5:8000)).
- Authenticate by walking naturally for 15 seconds.
- The app sends data, the server verifies the Energy Score, generates an Embedding, and returns the Cosine Similarity decision.

## 🔑 6. Authentication Decision Logic
The system returns decisions based on the Cosine Similarity score against enrolled templates:
| Response | Condition |
| --- | --- |
| **ACCESS GRANTED** | Similarity ≥ 0.70 with an enrolled template |
| **ACCESS DENIED** | Similarity < 0.70 (Unrecognized user) |
| **STATIC/FAKE DETECTED** | Energy Score < 1.0 (Fake walk attempt) |
| **INSUFFICIENT DATA** | CSV contains < 128 samples |
---

# 7. Dataset Engineering & Server Architecture

## 7.1 RealWorldLive Dataset: The Biological Foundation

The RealWorldLive dataset serves as the high-fidelity ground truth for the production system.
- **Collection Methodology:** Data was collected using a custom-built MIT App Inventor client that polls the 3D Accelerometer and 3D Gyroscope at a stabilized frequency of approximately 20 Hz.
- **Subject Count (N=10):** The dataset was expanded from 8 to 10 unique individuals to provide a more robust baseline for identity variance.
- **Why 10 People?:** In a Biometric Siamese framework, these 10 individuals act as the "Anchor" identities. The model does not need thousands of real people to learn; it only needs enough high-quality, real-world examples to understand the fundamental physics of a human step.
- **Environment:** Unlike the UCI HAR dataset, this data was collected in uncontrolled real-world environments (indoor corridors and outdoor walkways), ensuring the model is resilient to natural floor surface variations.

## 7.2 AI-Driven Synthetic Scaling (100+ Users)

To bridge the gap between a 10-person pilot and an enterprise-scale system, an AI-driven Synthetic Data Generator was implemented.
- **Generation Logic:** The script (`generate_synthetic_gait.py`) uses the 10 real users as "biological seeds". It mathematically derives 100 additional unique identities (Person 11 to 110).
- **Purpose of Synthetic Data:** Without this expansion, a Deep Learning model would overfit, effectively memorizing the 10 real people rather than learning generalized gait mechanics.
- **Augmentation Techniques:**
    - *Time Warping:* Simulates variations in walking pace and stride length.
    - *Jittering:* Injects Gaussian noise to simulate different sensor qualities across various smartphone brands.
    - *Phase Shifting:* Simulates the 15-second recording starting at different points in the gait cycle (e.g., mid-swing vs. heel-strike).

## 7.3 Robust Flask Server (`flask_server.py`)

The production backend is a hardened Flask implementation optimized for low-latency biometric inference.
- **AI Warm-up Routine:** Upon initialization, the server executes a "Dummy Prediction". This triggers TensorFlow graph and CUDA/CPU optimizations before user arrival, eliminating the common 10-second "Cold Start" lag.
- **The Physics Guard (Walk Energy Score):** Before AI 

---

## ✅ Deployment Readiness
- **Scalability:** Successfully handles 100+ users via Siamese embedding architecture.
- **Stability:** Optimized with TensorFlow warm-up routines for instant response.
- **Security:** 3D-Magnitude energy filtering defeats physical spoofing.