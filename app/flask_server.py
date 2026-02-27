# from flask import Flask, request, jsonify
# import os
# import time
# import logging
# import pandas as pd
# import sys

# # Ensure the server can see files in the parent 'production' folder
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# from cnn_engine.infer_realtime import predict_person

# app = Flask(__name__)

# # ================= CONFIG =================
# # Based on your image, received_gait.csv is in the 'production' folder
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# CSV_PATH = os.path.abspath(os.path.join(BASE_DIR, "..", "received_gait.csv"))

# MAX_REQUEST_SIZE = 2 * 1024 * 1024
# app.config["MAX_CONTENT_LENGTH"] = MAX_REQUEST_SIZE

# EXPECTED_DURATION_MS = 15000
# DURATION_TOLERANCE = 3000  # Increased slightly for network latency

# API_KEY = "GAIT_SECURE_2026"

# logging.basicConfig(
#     level=logging.INFO,
#     format="%(asctime)s | %(levelname)s | %(message)s"
# )

# def log_block():
#     logging.info("=" * 65)

# print("\n📁 CSV path:", CSV_PATH)
# print("🚀 Secure Biometric Flask server running on port 8000...\n")

# # ================= ROUTE =================
# @app.route("/predict", methods=["POST"])
# def predict():
#     start_time = time.time()
#     log_block()
#     logging.info("📥 New request received from MIT App")

#     try:
#         # 1. API KEY VERIFICATION
#         client_key = request.headers.get("X-API-KEY")
#         if client_key != API_KEY:
#             logging.warning("🚫 Unauthorized request - Invalid API key")
#             return jsonify({"result": "ACCESS_DENIED", "reason": "UNAUTHORIZED"}), 401

#         # 2. READ BODY
#         csv_text = request.get_data(as_text=True)
#         if not csv_text or not csv_text.strip():
#             logging.warning("❌ Empty payload")
#             return jsonify({"result": "ACCESS_DENIED", "reason": "EMPTY"}), 400

#         # 3. SAVE CSV (Forces data to disk for model to read)
#         with open(CSV_PATH, "w", encoding="utf-8", newline="") as f:
#             f.write(csv_text)
#             f.flush()
#             os.fsync(f.fileno())
#         logging.info(f"📄 CSV saved to disk | Size: {len(csv_text)} bytes")

#         # 4. SERVER TEST HANDLER (Compatible with your 'Test_Send' block)
#         if "," not in csv_text:
#             logging.info(f"🧪 Test Signal: {csv_text}")
#             return jsonify({"result": "SERVER_TEST_RECEIVED"}), 200

#         # 5. LOAD AND VALIDATE STRUCTURE
#         df = pd.read_csv(CSV_PATH)
#         if "timestamp" not in df.columns:
#             logging.warning("❌ Missing timestamp column")
#             return jsonify({"result": "ACCESS_DENIED", "reason": "BAD_FORMAT"}), 200

#         # 6. REPLAY PROTECTION (Duration Check)
#         duration = df["timestamp"].iloc[-1] - df["timestamp"].iloc[0]
#         logging.info(f"⏱ Recording duration: {duration} ms")

#         if not (EXPECTED_DURATION_MS - DURATION_TOLERANCE <= duration <= EXPECTED_DURATION_MS + DURATION_TOLERANCE):
#             logging.warning("🚫 Replay or manipulated recording detected")
#             return jsonify({"result": "ACCESS_DENIED", "reason": "INVALID_DURATION"}), 200

#         # 7. ROBUST STATIC & TRICK DETECTION (Energy Check)
#         # Combine X, Y, Z into a single 3D force vector (Magnitude)
#         df['acc_mag'] = (df['ax']**2 + df['ay']**2 + df['az']**2)**0.5
#         walk_energy = df['acc_mag'].std()
        
#         logging.info(f"📊 Walk Energy Score: {walk_energy:.4f}")

#         # Real walking usually produces an energy score well above 1.5.
#         # A fake "lift and drop" trick will struggle to pass 1.0.
#         if walk_energy < 1.0: 
#             logging.warning(f"🧍 Static/Fake walk detected (Energy: {walk_energy:.4f})")
#             return jsonify({"result": "ACCESS_DENIED", "reason": "STATIC_DETECTED"}), 200

#         # 8. NEW SIAMESE MODEL PREDICTION
#         # This calls the infer_realtime.py script we finalized earlier
#         result = predict_person(CSV_PATH)

#         response_time = round(time.time() - start_time, 3)
#         logging.info(f"🧠 Model Decision: {result}")
#         logging.info(f"⚡ Total Response time: {response_time} sec")
#         log_block()

#         return jsonify({
#             "result": result,
#             "response_time_sec": response_time
#         }), 200

#     except Exception as e:
#         logging.error(f"🔥 Server error: {str(e)}")
#         return jsonify({"result": "ERROR", "message": str(e)}), 500

# if __name__ == "__main__":
#     app.run(host="0.0.0.0", port=8000, threaded=True)

from flask import Flask, request, jsonify
import os
import time
import logging
import pandas as pd
import sys

# Ensure the server can see files in the parent 'production' folder
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from cnn_engine.infer_realtime import predict_person

app = Flask(__name__)

# ================= CONFIG =================
# Pathing adjusted for Hugging Face Docker structure
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.abspath(os.path.join(BASE_DIR, "..", "received_gait.csv"))

MAX_REQUEST_SIZE = 2 * 1024 * 1024
app.config["MAX_CONTENT_LENGTH"] = MAX_REQUEST_SIZE

EXPECTED_DURATION_MS = 15000
DURATION_TOLERANCE = 3000  

API_KEY = "GAIT_SECURE_2026"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

def log_block():
    logging.info("=" * 65)

# ================= WARM-UP ROUTINE =================
def warmup_model():
    """Triggers the TensorFlow engine to prevent lag on first request."""
    logging.info("🧠 Warming up AI model to prevent memory crash...")
    try:
        # Create a dummy file to trigger the first inference
        dummy_df = pd.DataFrame([[0]*7]*150, columns=['timestamp','ax','ay','az','wx','wy','wz'])
        dummy_df.to_csv(CSV_PATH, index=False)
        predict_person(CSV_PATH)
        logging.info("✅ Model Warm-up Complete. Ready for authentication.")
    except Exception as e:
        logging.error(f"⚠️ Warm-up failed: {e}")

# ================= ROUTE =================
@app.route("/", methods=["GET"])
def health_check():
    return "Gait-Secure API is running on Hugging Face!", 200

@app.route("/predict", methods=["POST"])
def predict():
    start_time = time.time()
    log_block()
    logging.info("📥 New request received from MIT App")

    try:
        # 1. API KEY VERIFICATION
        client_key = request.headers.get("X-API-KEY")
        if client_key != API_KEY:
            logging.warning("🚫 Unauthorized request - Invalid API key")
            return jsonify({"result": "ACCESS_DENIED", "reason": "UNAUTHORIZED"}), 401

        # 2. READ BODY
        csv_text = request.get_data(as_text=True)
        if not csv_text or not csv_text.strip():
            logging.warning("❌ Empty payload")
            return jsonify({"result": "ACCESS_DENIED", "reason": "EMPTY"}), 400

        # 3. SAVE CSV 
        with open(CSV_PATH, "w", encoding="utf-8", newline="") as f:
            f.write(csv_text)
            f.flush()
            os.fsync(f.fileno())
        logging.info(f"📄 CSV saved to disk | Size: {len(csv_text)} bytes")

        # 4. SERVER TEST HANDLER 
        if "," not in csv_text:
            logging.info(f"🧪 Test Signal: {csv_text}")
            return jsonify({"result": "SERVER_TEST_RECEIVED"}), 200

        # 5. LOAD AND VALIDATE STRUCTURE
        df = pd.read_csv(CSV_PATH)
        if "timestamp" not in df.columns:
            logging.warning("❌ Missing timestamp column")
            return jsonify({"result": "ACCESS_DENIED", "reason": "BAD_FORMAT"}), 200

        # 6. REPLAY PROTECTION (Duration Check)
        duration = df["timestamp"].iloc[-1] - df["timestamp"].iloc[0]
        logging.info(f"⏱ Recording duration: {duration} ms")

        if not (EXPECTED_DURATION_MS - DURATION_TOLERANCE <= duration <= EXPECTED_DURATION_MS + DURATION_TOLERANCE):
            logging.warning("🚫 Replay or manipulated recording detected")
            return jsonify({"result": "ACCESS_DENIED", "reason": "INVALID_DURATION"}), 200

        # 7. ROBUST STATIC & TRICK DETECTION (Energy Check)
        df['acc_mag'] = (df['ax']**2 + df['ay']**2 + df['az']**2)**0.5
        walk_energy = df['acc_mag'].std()
        
        logging.info(f"📊 Walk Energy Score: {walk_energy:.4f}")

        if walk_energy < 1.0: 
            logging.warning(f"🧍 Static/Fake walk detected (Energy: {walk_energy:.4f})")
            return jsonify({"result": "ACCESS_DENIED", "reason": "STATIC_DETECTED"}), 200

        # 8. NEW SIAMESE MODEL PREDICTION
        result = predict_person(CSV_PATH)

        response_time = round(time.time() - start_time, 3)
        logging.info(f"🧠 Model Decision: {result}")
        logging.info(f"⚡ Total Response time: {response_time} sec")
        log_block()

        return jsonify({
            "result": result,
            "response_time_sec": response_time
        }), 200

    except Exception as e:
        logging.error(f"🔥 Server error: {str(e)}")
        return jsonify({"result": "ERROR", "message": str(e)}), 500

if __name__ == "__main__":
    warmup_model()
    # Hugging Face MUST use port 7860
    app.run(host="0.0.0.0", port=7860, threaded=True)