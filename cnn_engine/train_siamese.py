import os
import sys
import random
import numpy as np
import tensorflow as tf

# Bulletproof path resolving so imports work anywhere
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.append(CURRENT_DIR)

from build_encoder import build_encoder
from dataset_loader import load_dataset, fit_scaler

# Look one folder UP for the RealWorldLive data
PARENT_DIR = os.path.dirname(CURRENT_DIR)
BASE_PATH = os.path.join(PARENT_DIR, "RealWorldLive")
MODEL_PATH = os.path.join(BASE_PATH, "model")
os.makedirs(MODEL_PATH, exist_ok=True)

print(f"🚀 Starting High-Speed 1D CNN Training with Semi-Hard Mining...")
person_data = load_dataset(BASE_PATH)
person_data = fit_scaler(person_data, os.path.join(MODEL_PATH, "scaler.pkl"))

persons = list(person_data.keys())
neg_pool = {p: [x for x in persons if x != p] for p in persons}

# ================= ULTRA-FAST AUGMENTATION =================
def augment_gait(window):
    aug_window = np.copy(window)
    
    if random.random() > 0.5:
        aug_window *= np.random.uniform(0.90, 1.10)
        
    if random.random() > 0.5:
        aug_window += np.random.normal(0, 0.015, aug_window.shape)
        
    # Time-Warping (Optimized to run in microseconds using np.interp)
    if random.random() > 0.5:
        speed_factor = np.random.uniform(0.85, 1.15)
        orig_steps = np.arange(aug_window.shape[0])
        sample_points = np.linspace(0, aug_window.shape[0]-1, aug_window.shape[0]) * speed_factor
        
        warped = np.zeros_like(aug_window)
        for i in range(aug_window.shape[1]):
            warped[:, i] = np.interp(sample_points, orig_steps, aug_window[:, i])
        aug_window = warped
        
    return aug_window
# ===========================================================

encoder = build_encoder()
optimizer = tf.keras.optimizers.Adam(0.0005)

MARGIN, EPOCHS, BATCH, STEPS = 0.5, 40, 64, 50

for epoch in range(EPOCHS):
    losses = []
    for _ in range(STEPS):
        a_batch_raw, p_batch_raw, n_pool_raw = [], [], []
        
        for _ in range(BATCH):
            p = random.choice(persons)
            a_batch_raw.append(random.choice(person_data[p]))
            p_batch_raw.append(augment_gait(random.choice(person_data[p])))
            
            neg_candidates = random.sample(neg_pool[p], min(5, len(neg_pool[p])))
            for n_c in neg_candidates:
                n_pool_raw.append(augment_gait(random.choice(person_data[n_c])))

        a_batch_np = np.array(a_batch_raw)
        p_batch_np = np.array(p_batch_raw)
        n_pool_np = np.array(n_pool_raw)

        # 2. SEMI-HARD NEGATIVE MINING
        a_emb_frozen = encoder(a_batch_np, training=False).numpy()
        p_emb_frozen = encoder(p_batch_np, training=False).numpy()
        n_emb_frozen = encoder(n_pool_np, training=False).numpy()
        
        semi_hard_n_batch_raw = []
        
        for i in range(BATCH):
            d_ap = np.sum(np.square(a_emb_frozen[i] - p_emb_frozen[i]))
            
            candidates_emb = n_emb_frozen[i*5 : (i+1)*5]
            d_an = np.sum(np.square(candidates_emb - a_emb_frozen[i]), axis=1)
            
            valid_indices = np.where(d_an > d_ap)[0]
            
            if len(valid_indices) > 0:
                best_idx = valid_indices[np.argmin(d_an[valid_indices])]
            else:
                best_idx = random.randint(0, 4)
                
            semi_hard_n_batch_raw.append(n_pool_np[i*5 + best_idx])

        semi_hard_n_batch_np = np.array(semi_hard_n_batch_raw)

        # 3. ACTUAL TRAINING STEP
        with tf.GradientTape() as tape:
            a_emb = encoder(a_batch_np, training=True)
            p_emb = encoder(p_batch_np, training=True)
            n_emb = encoder(semi_hard_n_batch_np, training=True)
            
            pos_dist = tf.reduce_sum(tf.square(a_emb - p_emb), axis=1)
            neg_dist = tf.reduce_sum(tf.square(a_emb - n_emb), axis=1)
            loss = tf.reduce_mean(tf.maximum(pos_dist - neg_dist + MARGIN, 0.0))
        
        grads = tape.gradient(loss, encoder.trainable_weights)
        optimizer.apply_gradients(zip(grads, encoder.trainable_weights))
        losses.append(loss.numpy())
    
    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {np.mean(losses):.4f}")

encoder.save(os.path.join(MODEL_PATH, "encoder.keras"))
print("✅ High-Speed Model with Semi-Hard Mining Saved Successfully.")
